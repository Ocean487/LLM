import torch
import torch._dynamo
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from huggingface_hub import login
import pynvml
import psutil
import threading
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import gc

login()

# ================= 路徑與環境變數設定 =================
BASE_DIR = "/home/s3734/heng_project"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
OFFLOAD_DIR = os.path.join(BASE_DIR, "offload_weights")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

MODELS_TO_TEST = {
    "1": ("gpt-oss-20b", "openai/gpt-oss-20b"),
    "2": ("gpt-oss-120b", "openai/gpt-oss-120b"),
    "3": ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    "4": ("Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct"),
    "5": ("translategemma-12b", "google/translategemma-12b-it"),
    "6": ("translategemma-27b", "google/translategemma-27b-it"),
    "7": ("Qwen1.5-7B", "Qwen/Qwen1.5-7B"),
    "8": ("Qwen1.5-72B", "Qwen/Qwen1.5-72B"),
    "9": ("Qwen1.5-110B", "Qwen/Qwen1.5-110B")
}

TEST_RUNS = 5
MAX_NEW_TOKENS = 1024 
PROMPT_ZH = "請撰寫一篇約 2000 字的深度分析文章，探討量子計算在未來十年內，將如何顛覆製藥和材料科學這兩個領域。文章需包含基本原理介紹、潛在應用案例，以及目前面臨的主要技術挑戰。"
USE_PROMPT = PROMPT_ZH 

# ================= 量化=================
class CustomCenterQuantizerLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True, compute_dtype=torch.bfloat16, bit_width=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.bit_width = bit_width
        
        self.q_max = (1 << (bit_width - 1)) - 1 

        self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("epsilon", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("gamma", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("scale", torch.zeros(1, dtype=torch.float32))
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=self.compute_dtype))
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def quantize_from_float(self, weight_float, bias_float=None):
        abs_w = torch.abs(weight_float)
        sign_w = torch.sign(weight_float)
        
        std_w = torch.std(weight_float).clamp(min=1e-7)
        eps = 0.1 * std_w  
        gam = 1.5 * std_w  
        
        self.epsilon.copy_(eps)
        self.gamma.copy_(gam)
        
        y = torch.zeros_like(weight_float)
        
        mask_core = (abs_w >= eps) & (abs_w < gam)
        mask_tail = (abs_w >= gam)
        
        y[mask_core] = sign_w[mask_core] * ((abs_w[mask_core] - eps) / (gam - eps))
        y[mask_tail] = sign_w[mask_tail] * (1.0 + torch.log(abs_w[mask_tail] / gam))
        
        max_y = torch.max(torch.abs(y)).clamp(min=1e-7)
        scale = self.q_max / max_y
        self.scale.copy_(scale)
        
        quantized_weight = torch.clamp(torch.round(y * scale), -self.q_max, self.q_max).to(torch.int8)
        self.weight_q.copy_(quantized_weight)
        
        if bias_float is not None and self.bias is not None:
            self.bias.copy_(bias_float.to(self.compute_dtype))

    def forward(self, x):
        current_dtype = x.dtype
        eps = self.epsilon.to(current_dtype)
        gam = self.gamma.to(current_dtype)
        scale = self.scale.to(current_dtype)
        
        y_hat = self.weight_q.to(current_dtype) / scale
        
        abs_y = torch.abs(y_hat)
        sign_y = torch.sign(y_hat)
        
        core_val = sign_y * (eps + abs_y * (gam - eps))
        
        tail_val = sign_y * gam * torch.exp(abs_y - 1.0)
        
        x_hat = torch.where(abs_y > 1.0, tail_val, core_val)
        
        x_hat = torch.where(abs_y == 0.0, torch.zeros_like(x_hat), x_hat)
        
        del abs_y, sign_y, y_hat, core_val, tail_val
        
        current_bias = self.bias.to(current_dtype) if self.bias is not None else None
        
        return nn.functional.linear(x, x_hat, current_bias)


def replace_and_quantize_model(model, current_path=""):
 
    for name, module in model.named_children():
        full_name = f"{current_path}.{name}" if current_path else name
        
        if isinstance(module, nn.Linear):
            if "lm_head" in full_name:
                continue
                
            quantized_layer = CustomCenterQuantizerLinear(
                in_features=module.in_features, 
                out_features=module.out_features, 
                bias=(module.bias is not None),
                compute_dtype=module.weight.dtype,
                bit_width=8
            )
            quantized_layer.quantize_from_float(module.weight.data, getattr(module, 'bias', None))
            setattr(model, name, quantized_layer)
        else:
            replace_and_quantize_model(module, full_name)
            
# ================= 硬體監控 =================
class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.put_count = 0
        self.first_token_time = None

    def put(self, value):
        if self.put_count == 1: 
            self.first_token_time = time.time()
        self.put_count += 1

    def end(self):
        pass

class HardwareMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.keep_running = False
        self.metrics = []
        self.last_disk_io = psutil.disk_io_counters()
        self.last_swap = psutil.swap_memory()
        self.last_time = time.time()
        self.last_read_time = getattr(self.last_disk_io, 'read_time', 0)

    def _safe_get_temp(self, sensor_name):
        try:
            temps = psutil.sensors_temperatures()
            if sensor_name in temps:
                return temps[sensor_name][0].current
            if temps: return list(temps.values())[0][0].current
        except: pass
        return 45.0

    def start(self):
        self.keep_running = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def _monitor_loop(self):
        while self.keep_running:
            current_time = time.time()
            time_diff = current_time - self.last_time
            if time_diff <= 0: time_diff = 0.001

            try:
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 
                gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 ** 2)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = util_rates.gpu
                gpu_mem_util = util_rates.memory 
                pcie_rx_mb = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
                pcie_tx_mb = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
            except:
                gpu_temp, gpu_power, gpu_mem_mb, gpu_util, gpu_mem_util, pcie_rx_mb, pcie_tx_mb = 0, 0, 0, 0, 0, 0, 0

            cpu_util = psutil.cpu_percent()
            cpu_mem_mb = psutil.virtual_memory().used / (1024 ** 2)
            cpu_temp = self._safe_get_temp('coretemp')
            cpu_power = 15.0 + (cpu_util * 0.8) 

            current_swap = psutil.swap_memory()
            swap_in_mb_s = ((current_swap.sin - self.last_swap.sin) / (1024**2)) / time_diff
            swap_out_mb_s = ((current_swap.sout - self.last_swap.sout) / (1024**2)) / time_diff

            current_disk_io = psutil.disk_io_counters()
            read_bytes = current_disk_io.read_bytes - self.last_disk_io.read_bytes
            read_count = current_disk_io.read_count - self.last_disk_io.read_count
            current_read_time = getattr(current_disk_io, 'read_time', 0)
            delta_read_time = current_read_time - self.last_read_time
            
            read_mb_s = (read_bytes / (1024**2)) / time_diff
            read_iops = read_count / time_diff
            latency_ms = (delta_read_time / read_count) if read_count > 0 else 0
            queue_depth = read_iops * (latency_ms / 1000.0)
            ssd_temp = self._safe_get_temp('nvme')
            ssd_power = 0.5 + (read_mb_s / 7300.0) * 6.0 

            self.last_time = current_time
            self.last_disk_io = current_disk_io
            self.last_swap = current_swap
            self.last_read_time = current_read_time

            self.metrics.append({
                "GPU_Temp": gpu_temp, "GPU_Power": gpu_power, "GPU_Mem_MB": gpu_mem_mb, 
                "GPU_Util": gpu_util, "GPU_Mem_Util(%)": gpu_mem_util, 
                "PCIe_RX_MB_s": pcie_rx_mb, "PCIe_TX_MB_s": pcie_tx_mb,
                "CPU_Temp": cpu_temp, "CPU_Power": cpu_power, "CPU_Mem_MB": cpu_mem_mb, 
                "CPU_Util": cpu_util, "RAM_Swap_In_MB_s": swap_in_mb_s, "RAM_Swap_Out_MB_s": swap_out_mb_s,
                "SSD_Temp": ssd_temp, "SSD_Power": ssd_power, "SSD_Read_MB_s": read_mb_s, 
                "SSD_Read_IOPS": read_iops, "SSD_Read_Latency_ms": latency_ms, "SSD_Queue_Depth": queue_depth
            })
            time.sleep(0.5)

    def stop(self):
        self.keep_running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        if not self.metrics: return {}
        return {k: sum(d[k] for d in self.metrics)/len(self.metrics) for k in self.metrics[0]}

def update_plot():
    if not os.path.exists(LOG_FILE): return
    df = pd.read_csv(LOG_FILE)
    if len(df) == 0: return
    
    metrics_to_plot = [
        "Total_Duration(s)", "Eval_Rate(t/s)", "GPU_Power(W)", "GPU_Mem_MB", "GPU_Util(%)",
        "RAM_Swap_In_MB_s", "SSD_Read_MB_s"
    ]
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            pivot_df = df.pivot(index='Run', columns='Model', values=metric)
            for column in pivot_df.columns:
                plt.plot(pivot_df.index, pivot_df[column], marker='o', linewidth=2, label=column)
            plt.xlabel('Run')
            plt.ylabel(metric)
            plt.title(f'{metric} Across Tests')
            plt.xticks(range(1, TEST_RUNS + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
            if not pivot_df.empty: plt.legend()
            plt.tight_layout()
            safe_filename = metric.replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')
            plt.savefig(os.path.join(BASE_DIR, f"chart_{safe_filename}.png"))
            plt.close()

# ================= 主程式 =================
def main():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Run", "Total_Duration(s)", "Load_Duration(s)", "Prompt_Eval_Count", 
                "Prompt_Eval_Duration(s)", "Prompt_Eval_Rate(t/s)", "Eval_Count", "Eval_Duration(s)", 
                "Eval_Rate(t/s)", "GPU_Temp(C)", "GPU_Power(W)", "GPU_Mem_MB", "GPU_Util(%)", 
                "GPU_Mem_Util(%)", "PCIe_RX_MB_s", "PCIe_TX_MB_s", "CPU_Temp(C)", "CPU_Power(W)", 
                "CPU_Mem_MB", "CPU_Util(%)", "RAM_Swap_In_MB_s", "RAM_Swap_Out_MB_s", "SSD_Temp(C)", 
                "SSD_Power(W)", "SSD_Read_MB_s", "SSD_Read_IOPS", "SSD_Read_Latency_ms", "SSD_Queue_Depth"
            ])

    print("\n" + "="*40)
    for key, (name, repo) in MODELS_TO_TEST.items():
        print(f"[{key}] {name}")
    print("="*40)
    import torch._dynamo
    
    choice = input("請輸入測試編號 (q 離開): ")
    if choice.lower() == 'q' or choice not in MODELS_TO_TEST: return
    model_name, repo_id = MODELS_TO_TEST[choice]

    monitor = HardwareMonitor()
    try:
        load_start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            add_eos_token=True,
            legacy=False
        )
        
        print("正在將原始模型載入系統 CPU RAM ")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            device_map="cpu", 
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        print("量化")
        replace_and_quantize_model(model)
        
         gc.collect()
        
        print("量化完成。正在將推入 GPU")
        model = model.to("cuda")
        
        
        torch._dynamo.config.suppress_errors = True 
        model = torch.compile(model)
        
        load_duration = time.time() - load_start
        print(f"載入與自定義量化總耗時: {load_duration:.2f} 秒")

        inputs = tokenizer(USE_PROMPT, return_tensors="pt").to("cuda")
        prompt_eval_count = inputs.input_ids.shape[1]

        for run in range(1, TEST_RUNS + 1):
            print(f"  執行第 {run}/{TEST_RUNS} 次推論測試...")
            streamer = TimingStreamer()
            monitor.start()
            gen_start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer, pad_token_id=tokenizer.eos_token_id
                )
            
            gen_end_time = time.time()
            avg_hw = monitor.stop()

            total_duration = gen_end_time - gen_start_time
            prompt_eval_duration = streamer.first_token_time - gen_start_time if streamer.first_token_time else total_duration
            eval_duration = total_duration - prompt_eval_duration
            eval_count = outputs.shape[1] - prompt_eval_count
            prompt_eval_rate = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
            eval_rate = eval_count / eval_duration if eval_duration > 0 else 0

            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name, run, round(total_duration, 4), round(load_duration, 4), prompt_eval_count, 
                    round(prompt_eval_duration, 4), round(prompt_eval_rate, 2), eval_count, round(eval_duration, 4), 
                    round(eval_rate, 2), round(avg_hw["GPU_Temp"], 1), round(avg_hw["GPU_Power"], 1), 
                    round(avg_hw["GPU_Mem_MB"], 1), round(avg_hw["GPU_Util"], 1), round(avg_hw["GPU_Mem_Util(%)"], 2),
                    round(avg_hw["PCIe_RX_MB_s"], 2), round(avg_hw["PCIe_TX_MB_s"], 2), round(avg_hw["CPU_Temp"], 1), 
                    round(avg_hw["CPU_Power"], 1), round(avg_hw["CPU_Mem_MB"], 1), round(avg_hw["CPU_Util"], 1),
                    round(avg_hw["RAM_Swap_In_MB_s"], 2), round(avg_hw["RAM_Swap_Out_MB_s"], 2), round(avg_hw["SSD_Temp"], 1), 
                    round(avg_hw["SSD_Power"], 2), round(avg_hw["SSD_Read_MB_s"], 2), round(avg_hw["SSD_Read_IOPS"], 2),
                    round(avg_hw["SSD_Read_Latency_ms"], 2), round(avg_hw["SSD_Queue_Depth"], 2)
                ])
            print(f"    生成速率: {eval_rate:.2f} t/s")

    except Exception as e:
        print(f"執行錯誤: {e}")
        monitor.stop()

    finally:
        print("\n清理記憶體資源...")
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        gc.collect()
        update_plot()

if __name__ == "__main__":
    main()
