import torch
import torch._dynamo
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_max_memory
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

# ================= 路徑與環境變數設定 =================
BASE_DIR = "/home/s3734/heng_project"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
OFFLOAD_DIR = os.path.join(BASE_DIR, "offload_weights")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
if os.path.exists(OFFLOAD_DIR):
    shutil.rmtree(OFFLOAD_DIR)
os.makedirs(OFFLOAD_DIR)
    
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
USE_PROMPT = "請撰寫一篇約 2000 字的深度分析文章，探討量子計算在未來十年內，將如何顛覆製藥和材料科學這兩個領域。文章需包含基本原理介紹、潛在應用案例，以及目前面臨的主要技術挑戰。"

# ================= 工具函式 =================
def get_decoder_layer_class(model):
    model_type = getattr(model.config, "model_type", "").lower()
    if "llama" in model_type:
        return ["LlamaDecoderLayer"]
    elif "qwen" in model_type:
        return ["Qwen2DecoderLayer"]
    elif "gemma" in model_type:
        return ["GemmaDecoderLayer"]
    elif "gpt" in model_type:
        return ["GPTJBlock", "GPT2Block", "ParallelBlock"]
    return None

# ================= 量化 =================
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
        eps, gam = 0.1 * std_w, 1.5 * std_w
        
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
        eps, gam, scale = self.epsilon.to(current_dtype), self.gamma.to(current_dtype), self.scale.to(current_dtype)
        y_hat = self.weight_q.to(current_dtype) / scale
        abs_y, sign_y = torch.abs(y_hat), torch.sign(y_hat)
        
        core_val = sign_y * (eps + abs_y * (gam - eps))
        tail_val = sign_y * gam * torch.exp(abs_y - 1.0)
        x_hat = torch.where(abs_y > 1.0, tail_val, core_val)
        x_hat = torch.where(abs_y == 0.0, torch.zeros_like(x_hat), x_hat)
        
        current_bias = self.bias.to(current_dtype) if self.bias is not None else None
        return nn.functional.linear(x, x_hat, current_bias)

def replace_and_quantize_model(model, current_path=""):
    for name, module in model.named_children():
        full_name = f"{current_path}.{name}" if current_path else name
        if isinstance(module, nn.Linear):
            if "lm_head" in full_name: continue
            quantized_layer = CustomCenterQuantizerLinear(
                in_features=module.in_features, out_features=module.out_features, 
                bias=(module.bias is not None), compute_dtype=module.weight.dtype
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
        if self.put_count == 1: self.first_token_time = time.time()
        self.put_count += 1
    def end(self): pass

class HardwareMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.handle = None
        self.keep_running = False
        self.metrics = []
        self.last_disk_io = psutil.disk_io_counters()

    def start(self):
        self.keep_running = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def _monitor_loop(self):
        while self.keep_running:
            try:
                gpu_mem, gpu_util, gpu_temp, gpu_power = 0, 0, 0, 0
                if self.handle:
                    info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    gpu_mem, gpu_util = info.used/(1024**2), util.gpu
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(self.handle, 0)
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000

                self.metrics.append({
                    "GPU_Temp": gpu_temp, "GPU_Power": gpu_power, "GPU_Mem_MB": gpu_mem,
                    "GPU_Util": gpu_util, "CPU_Util": psutil.cpu_percent(),
                    "CPU_Mem_MB": psutil.virtual_memory().used/(1024**2),
                    "SSD_Read_MB_s": (psutil.disk_io_counters().read_bytes - self.last_disk_io.read_bytes)/(1024**2)
                })
                self.last_disk_io = psutil.disk_io_counters()
                time.sleep(0.5)
            except: pass

    def stop(self):
        self.keep_running = False
        if hasattr(self, 'thread'): self.thread.join()
        if not self.metrics: return {k:0 for k in ["GPU_Temp", "GPU_Power", "GPU_Mem_MB", "GPU_Util", "CPU_Util", "CPU_Mem_MB", "SSD_Read_MB_s"]}
        return {k: sum(d[k] for d in self.metrics)/len(self.metrics) for k in self.metrics[0]}

# ================= 主程式 =================
def main():
    login()
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Model", "Run", "Total_Duration(s)", "Eval_Rate(t/s)", "GPU_Mem_MB", "SSD_Read_MB_s"])

    print("\n" + "="*40)
    for key, (name, _) in MODELS_TO_TEST.items(): print(f"[{key}] {name}")
    choice = input("輸入測試編號: ")
    if choice not in MODELS_TO_TEST: return
    model_name, repo_id = MODELS_TO_TEST[choice]

    monitor = HardwareMonitor()

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        input_tokens = tokenizer(USE_PROMPT, return_tensors="pt")
        prompt_eval_count = input_tokens.input_ids.shape[1]

        print("載入模型至 CPU RAM")
        model = AutoModelForCausalLM.from_pretrained(repo_id, device_map="cpu", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        
        print("執行量化")
        replace_and_quantize_model(model)
        gc.collect()

        print("自動分配權重")
        sys_max_mem = get_max_memory()
        max_memory = {k: (int(v * 0.80) if isinstance(k, int) else v) 
            for k, v in sys_max_mem.items()}
        
        layer_class = get_decoder_layer_class(model)
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=layer_class)
        
        model = dispatch_model(
            model, 
            device_map=device_map, 
            offload_dir=OFFLOAD_DIR, 
            offload_buffers=True
        )
        
        print(f"分配狀態: {model.hf_device_map}")
        
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int): first_device = f"cuda:{first_device}"
        model_inputs = {k: v.to(first_device) for k, v in input_tokens.items()}

        load_duration = time.time() - load_start

        for run in range(1, TEST_RUNS + 1):
            print(f"測試次數 {run}/{TEST_RUNS}...")
            streamer = TimingStreamer()
            monitor.start()
            start_t = time.time()
            
            with torch.no_grad():
                outputs = model.generate(**model_inputs, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer, pad_token_id=tokenizer.eos_token_id)
            
            duration = time.time() - start_t
            avg_hw = monitor.stop()
            eval_count = outputs.shape[1] - prompt_eval_count
            
            eval_time = duration - (streamer.first_token_time - start_t if streamer.first_token_time else 0)
            eval_rate = eval_count / eval_time if eval_time > 0 else 0

            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([model_name, run, round(duration, 2), round(eval_rate, 2), round(avg_hw["GPU_Mem_MB"], 1), round(avg_hw["SSD_Read_MB_s"], 2)])
            print(f"完成。速率: {eval_rate:.2f} t/s")

    except Exception as e:
        print(f"執行中發生錯誤: {e}")
        if monitor: monitor.stop()
    finally:
        print("清理資源...")
        if 'model' in locals(): del model
        if os.path.exists(OFFLOAD_DIR):
            shutil.rmtree(OFFLOAD_DIR)
            os.makedirs(OFFLOAD_DIR)
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
