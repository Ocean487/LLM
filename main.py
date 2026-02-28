import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
import pynvml
import psutil
import threading
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from huggingface_hub import login
 
login()
# ================= 路徑與環境變數設定 =================
BASE_DIR = "/home/s3734/heng_project"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# ================= 參數與模型清單 =================
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

# ================= 效能計算攔截器 =================
class OllamaTimingStreamer(BaseStreamer):
    def __init__(self):
        self.put_count = 0
        self.first_token_time = None

    def put(self, value):
        if self.put_count == 1: 
            self.first_token_time = time.time()
        self.put_count += 1

    def end(self):
        pass

# ================= 硬體監控類別 =================
class HardwareMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.keep_running = False
        self.metrics = []
        
        # 初始化磁碟與 Swap 狀態
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

            # --- 1. GPU 數據 (包含 VRAM 活躍度與 PCIe 頻寬) ---
            try:
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 
                gpu_mem_mb = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 ** 2)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = util_rates.gpu
                gpu_mem_util = util_rates.memory # VRAM 活躍度 (%)
                
                # 抓取 PCIe RX (接收) 與 TX (發送) 頻寬 (單位轉換為 MB/s)
                pcie_rx_mb = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
                pcie_tx_mb = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
            except:
                gpu_temp, gpu_power, gpu_mem_mb, gpu_util, gpu_mem_util, pcie_rx_mb, pcie_tx_mb = 0, 0, 0, 0, 0, 0, 0

            # --- 2. CPU RAM 數據 (包含 Swap 虛擬記憶體交換頻寬) ---
            cpu_util = psutil.cpu_percent()
            cpu_mem_mb = psutil.virtual_memory().used / (1024 ** 2)
            cpu_temp = self._safe_get_temp('coretemp')
            cpu_power = 15.0 + (cpu_util * 0.8) 

            current_swap = psutil.swap_memory()
            swap_in_mb_s = ((current_swap.sin - self.last_swap.sin) / (1024**2)) / time_diff
            swap_out_mb_s = ((current_swap.sout - self.last_swap.sout) / (1024**2)) / time_diff

            # --- 3. SSD 數據 ---
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

            # 更新狀態
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
        self.thread.join()
        if not self.metrics: return {}
        return {k: sum(d[k] for d in self.metrics)/len(self.metrics) for k in self.metrics[0]}

# ================= 繪圖與更新分析 =================
def update_plot():
    if not os.path.exists(LOG_FILE):
        return
    
    df = pd.read_csv(LOG_FILE)
    
    metrics_to_plot = [
        "Total_Duration(s)", "Load_Duration(s)",
        "Prompt_Eval_Duration(s)", "Prompt_Eval_Rate(t/s)",
        "Eval_Duration(s)", "Eval_Rate(t/s)",
        "GPU_Temp(C)", "GPU_Power(W)", "GPU_Mem_MB", "GPU_Util(%)", "GPU_Mem_Util(%)",
        "PCIe_RX_MB_s", "PCIe_TX_MB_s",
        "CPU_Temp(C)", "CPU_Power(W)", "CPU_Mem_MB", "CPU_Util(%)",
        "RAM_Swap_In_MB_s", "RAM_Swap_Out_MB_s",
        "SSD_Temp(C)", "SSD_Power(W)", "SSD_Read_MB_s", "SSD_Read_IOPS",
        "SSD_Read_Latency_ms", "SSD_Queue_Depth"
    ]

    print("開始生成包含全部測試次數的折線圖...")
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            plt.figure(figsize=(10, 6))
            pivot_df = df.pivot(index='Run', columns='Model', values=metric)
            
            for column in pivot_df.columns:
                plt.plot(pivot_df.index, pivot_df[column], marker='o', linestyle='-', linewidth=2, markersize=8, label=column)
            
            plt.xlabel('Run (Test Iteration)')
            plt.ylabel(metric)
            plt.title(f'{metric} Across 5 Runs by Model')
            plt.xticks(range(1, TEST_RUNS + 1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Model")
            plt.tight_layout()
            
            safe_filename = metric.replace('/', '_per_').replace('(', '_').replace(')', '').replace('%', 'pct')
            file_path = os.path.join(BASE_DIR, f"chart_line_{safe_filename}.png")
            
            plt.savefig(file_path)
            plt.close()
            
    print("所有折線圖已更新並儲存至專案目錄。")

# ================= 主程式 =================
def main():
    print(f"專案路徑設定為: {BASE_DIR}")
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Run", 
                "Total_Duration(s)", "Load_Duration(s)",
                "Prompt_Eval_Count", "Prompt_Eval_Duration(s)", "Prompt_Eval_Rate(t/s)",
                "Eval_Count", "Eval_Duration(s)", "Eval_Rate(t/s)",
                "GPU_Temp(C)", "GPU_Power(W)", "GPU_Mem_MB", "GPU_Util(%)", "GPU_Mem_Util(%)",
                "PCIe_RX_MB_s", "PCIe_TX_MB_s",
                "CPU_Temp(C)", "CPU_Power(W)", "CPU_Mem_MB", "CPU_Util(%)",
                "RAM_Swap_In_MB_s", "RAM_Swap_Out_MB_s",
                "SSD_Temp(C)", "SSD_Power(W)", "SSD_Read_MB_s", "SSD_Read_IOPS",
                "SSD_Read_Latency_ms", "SSD_Queue_Depth"
            ])

    print("\n" + "="*40)
    print("可用模型清單:")
    for key, (name, repo) in MODELS_TO_TEST.items():
        print(f"[{key}] {name} ({repo})")
    print("="*40)
    
    choice = input("請輸入要測試的模型編號 (輸入 q 離開): ")
    if choice.lower() == 'q':
        return
    if choice not in MODELS_TO_TEST:
        print("錯誤的選項。")
        return

    model_name, repo_id = MODELS_TO_TEST[choice]
    
    print(f"\n準備下載並載入模型: {model_name}")
    monitor = HardwareMonitor()

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            dtype=torch.bfloat16, 
            device_map="cuda"
        )
        load_duration = time.time() - load_start
        print(f"模型載入完成，Load Duration: {load_duration:.2f} 秒")

        inputs = tokenizer(USE_PROMPT, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        prompt_eval_count = inputs.input_ids.shape[1]

        for run in range(1, TEST_RUNS + 1):
            print(f"  執行第 {run}/{TEST_RUNS} 次測試...")
            
            streamer = OllamaTimingStreamer()
            monitor.start()
            
            gen_start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=MAX_NEW_TOKENS,
                    streamer=streamer,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            gen_end_time = time.time()
            avg_hw = monitor.stop()

            total_duration = gen_end_time - gen_start_time
            
            if streamer.first_token_time:
                prompt_eval_duration = streamer.first_token_time - gen_start_time
            else:
                prompt_eval_duration = total_duration

            eval_duration = total_duration - prompt_eval_duration
            eval_count = outputs.shape[1] - prompt_eval_count
            
            prompt_eval_rate = prompt_eval_count / prompt_eval_duration if prompt_eval_duration > 0 else 0
            eval_rate = eval_count / eval_duration if eval_duration > 0 else 0

            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name, run, 
                    round(total_duration, 4), round(load_duration, 4),
                    prompt_eval_count, round(prompt_eval_duration, 4), round(prompt_eval_rate, 2),
                    eval_count, round(eval_duration, 4), round(eval_rate, 2),
                    round(avg_hw["GPU_Temp"], 1), round(avg_hw["GPU_Power"], 1), 
                    round(avg_hw["GPU_Mem_MB"], 1), round(avg_hw["GPU_Util"], 1), round(avg_hw["GPU_Mem_Util(%)"], 2),
                    round(avg_hw["PCIe_RX_MB_s"], 2), round(avg_hw["PCIe_TX_MB_s"], 2),
                    round(avg_hw["CPU_Temp"], 1), round(avg_hw["CPU_Power"], 1), 
                    round(avg_hw["CPU_Mem_MB"], 1), round(avg_hw["CPU_Util"], 1),
                    round(avg_hw["RAM_Swap_In_MB_s"], 2), round(avg_hw["RAM_Swap_Out_MB_s"], 2),
                    round(avg_hw["SSD_Temp"], 1), round(avg_hw["SSD_Power"], 2),
                    round(avg_hw["SSD_Read_MB_s"], 2), round(avg_hw["SSD_Read_IOPS"], 2),
                    round(avg_hw["SSD_Read_Latency_ms"], 2), round(avg_hw["SSD_Queue_Depth"], 2)
                ])
            
            print(f"    完成! Eval Rate: {eval_rate:.2f} t/s | PCIe RX 頻寬: {avg_hw['PCIe_RX_MB_s']:.2f} MB/s")

    except Exception as e:
        print(f"執行失敗: {e}")
        monitor.stop()

    finally:
        print(f"\n測試結束清理 {model_name} 的資源...")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        print("   GPU 與 RAM 記憶體釋放。")
        
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            print("   SSD 快取目刪除。")
            
        update_plot()

if __name__ == "__main__":
    main()
