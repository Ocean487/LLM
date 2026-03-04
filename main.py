import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
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
OFFLOAD_DIR = os.path.join(BASE_DIR, "offload_weights")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
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
PROMPT_ZH = "請撰寫一篇約 2000 字的深度分析文章，探討量子計算在未來十年內，將如何顛覆製藥和材料科學這兩個領域。"

# ================= 效能計算攔截器 =================
class OllamaTimingStreamer(BaseStreamer):
    def __init__(self):
        self.put_count = 0
        self.first_token_time = None
    def put(self, value):
        if self.put_count == 1: self.first_token_time = time.time()
        self.put_count += 1
    def end(self): pass

# ================= 硬體監控類別 =================
class HardwareMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except: self.handle = None
        self.keep_running = False
        self.metrics = []
        self.thread = None
        self.last_disk_io = psutil.disk_io_counters()
        self.last_time = time.time()

    def start(self):
        self.keep_running = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def _monitor_loop(self):
        while self.keep_running:
            current_time = time.time()
            time_diff = max(current_time - self.last_time, 0.001)
            try:
                gpu_mem, gpu_util = 0, 0
                if self.handle:
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024**2)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                curr_io = psutil.disk_io_counters()
                read_mb_s = ((curr_io.read_bytes - self.last_disk_io.read_bytes) / (1024**2)) / time_diff
                self.metrics.append({
                    "GPU_Mem_MB": gpu_mem, "GPU_Util": gpu_util,
                    "CPU_Mem_MB": psutil.virtual_memory().used/(1024**2),
                    "CPU_Util": psutil.cpu_percent(),
                    "SSD_Read_MB_s": read_mb_s
                })
                self.last_time, self.last_disk_io = current_time, curr_io
            except: pass
            time.sleep(0.5)

    def stop(self):
        self.keep_running = False
        if self.thread and self.thread.is_alive(): self.thread.join()
        if not self.metrics: return {}
        return {k: sum(d[k] for d in self.metrics)/len(self.metrics) for k in self.metrics[0]}

def update_plot():
    if not os.path.exists(LOG_FILE): return
    df = pd.read_csv(LOG_FILE)
    if df.empty: return
    metrics = ["Duration(s)", "Eval_Rate(t/s)", "GPU_Mem_MB", "SSD_Read_MB_s"]
    for m in metrics:
        if m in df.columns:
            plt.figure(figsize=(10, 6))
            pivot = df.pivot(index='Run', columns='Model', values=m)
            pivot.plot(marker='o')
            plt.title(f"{m} Benchmark")
            plt.savefig(os.path.join(BASE_DIR, f"chart_{m.replace('/', '_')}.png"))
            plt.close()

# ================= 主程式 =================
def main():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Model", "Run", "Duration(s)", "Eval_Rate(t/s)", "GPU_Mem_MB", "SSD_Read_MB_s"])

    print("\n可用模型清單:")
    for k, v in MODELS_TO_TEST.items(): print(f"[{k}] {v[0]}")
    choice = input("請輸入編號: ")
    if choice not in MODELS_TO_TEST: return
    model_name, repo_id = MODELS_TO_TEST[choice]
    quant_config = Mxfp4Config() 

    monitor = HardwareMonitor()
    try:
        print(f"載入模型 {model_name}...")
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # 傳入正確的 quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            quantization_config=quant_config,
            device_map="auto",
            max_memory={0: "12GiB"}, 
            offload_folder=OFFLOAD_DIR,
            low_cpu_mem_usage=True
        )
        print(f"載入完成，耗時: {time.time() - load_start:.2f}s")

        inputs = tokenizer(PROMPT_ZH, return_tensors="pt").to("cuda")

        for run in range(1, TEST_RUNS + 1):
            print(f"執行第 {run} 次測試...")
            streamer = OllamaTimingStreamer()
            monitor.start()
            start_t = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer)
            avg_hw = monitor.stop()
            total_dur = time.time() - start_t
            eval_count = outputs.shape[1] - inputs.input_ids.shape[1]
            eval_rate = eval_count / total_dur

            with open(LOG_FILE, 'a', newline='') as f:
                csv.writer(f).writerow([model_name, run, round(total_dur, 2), round(eval_rate, 2), 
                                       round(avg_hw.get("GPU_Mem_MB", 0), 1), round(avg_hw.get("SSD_Read_MB_s", 0), 2)])
            print(f"  完成! 速度: {eval_rate:.2f} t/s")

    except Exception as e:
        print(f"錯誤: {e}")
        if monitor.thread: monitor.stop()
    finally:
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        update_plot()

if __name__ == "__main__":
    main()
