import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import bitsandbytes as bnb
from accelerate import infer_auto_device_map, dispatch_model
import pynvml
import psutil
import threading
import time
import csv
import os
import shutil
import traceback

# 環境變數：優化顯存
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ================= 配置設定 =================
BASE_DIR = "/home/s3734/heng_project"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
OFFLOAD_DIR = os.path.join(BASE_DIR, "offload_weights")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

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
def get_no_split_modules(model):
    # 針對不同架構定義不分割層（防止一個 Layer 被拆在 CPU 和 GPU 導致報錯）
    if "Gemma2" in model.__class__.__name__:
        return ["Gemma2DecoderLayer"]
    elif "Llama" in model.__class__.__name__:
        return ["LlamaDecoderLayer"]
    elif "Qwen2" in model.__class__.__name__:
        return ["Qwen2DecoderLayer"]
    return getattr(model, "_no_split_modules", None)
# ================= 工具類別 =================
def save_to_csv(data):
    fieldnames = ["Timestamp", "Model", "Run", "TPS", "GPU_MB", "CPU_Util", "LoadTime_Sec"]
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: data.get(k, "N/A") for k in fieldnames})

class HardwareMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.keep_running = False
        self.metrics = []
    def start(self):
        self.keep_running = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
    def _monitor_loop(self):
        while self.keep_running:
            try:
                res = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.metrics.append({
                    "GPU_Util": res.gpu,
                    "GPU_Mem_MB": mem.used / (1024**2),
                    "CPU_Util": psutil.cpu_percent()
                })
            except: pass
            time.sleep(0.5)
    def stop(self):
        self.keep_running = False
        if hasattr(self, 'thread'): self.thread.join()
        if not self.metrics: return {"GPU_Util":0, "GPU_Mem_MB":0, "CPU_Util":0}
        return {k: sum(d[k] for d in self.metrics)/len(self.metrics) for k in self.metrics[0]}

# ================= 手動 NF4 核心邏輯 =================
def convert_to_nf4_manual(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # 排除掉某些不應量化的層 (如 lm_head)
            if "lm_head" in name or "embed_tokens" in name:
                continue
                
            new_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                compress_statistics=True,
                device=torch.device("cpu")
            )
            
            # 確保權重是實體張量後才封裝
            if module.weight.device.type == 'meta':
                raise RuntimeError(f"層 {name} 仍為 Meta Tensor，請確認 low_cpu_mem_usage 設定。")
                
            new_layer.weight = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                quant_type="nf4",
                compress_statistics=True
            ).to("cpu")
            
            if module.bias is not None:
                new_layer.bias = nn.Parameter(module.bias.data).to("cpu")
            
            setattr(model, name, new_layer)
        else:
            convert_to_nf4_manual(module)
def main():
    print("\n" + "="*50)
    print("手動 NF4 測試工具 - RTX 5070 Ti 16GB")
    print("="*50)
    for key, (name, _) in MODELS_TO_TEST.items():
        print(f"[{key}] {name}")
    
    choice = input("選擇編號: ").strip()
    if choice not in MODELS_TO_TEST: return
    model_name, repo_id = MODELS_TO_TEST[choice]

    torch.cuda.empty_cache()
    monitor = HardwareMonitor()

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        
        # 修正：12B 以上模型建議開啟 low_cpu_mem_usage=True 但需配合緩慢加載
        # 如果 device_map=None 還是報 Meta Tensor 錯誤，請將其設為 "cpu"
        print(f"[系統] 正在將模型載入至 RAM...")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="cpu",  # 明確指定載入到 CPU，避免 Meta Tensor
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, # Gemma 2 必須開啟此項以節省 RAM
            trust_remote_code=True
        )

        print("[系統] 正在進行 NF4 權重封裝...")
        convert_to_nf4_manual(model)

        # 針對 Gemma 2 優化 Device Map
        no_split = get_no_split_modules(model)
        
        # 顯存保留：gemma-12b 建議保留更多顯存給運算
        max_mem = {0: "11GiB", "cpu": "128GiB"} 
        
        device_map = infer_auto_device_map(
            model,
            max_memory=max_mem,
            no_split_module_classes=no_split
        )
        
        print("[系統] 正在派發模型至 GPU...")
        model = dispatch_model(model, device_map=device_map, offload_dir=OFFLOAD_DIR)

        load_duration = time.time() - load_start
        print(f"[成功] 模型載入並量化完成，耗時: {load_duration:.2f}s")

        # 測試推論
        inputs = tokenizer("請用繁體中文介紹什麼是人工智慧。", return_tensors="pt").to("cuda")
        
        for run in range(1, 6):
            monitor.start()
            gen_start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_end = time.time()
            avg_hw = monitor.stop()

            tps = (output.shape[1] - inputs.input_ids.shape[1]) / (gen_end - gen_start)
            save_to_csv({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Model": model_name,
                "Run": run,
                "TPS": round(tps, 2),
                "GPU_MB": round(avg_hw['GPU_Mem_MB'], 0),
                "CPU_Util": round(avg_hw['CPU_Util'], 1),
                "LoadTime_Sec": round(load_duration, 2)
            })
            print(f"Run {run}: {tps:.2f} t/s | 顯存: {avg_hw['GPU_Mem_MB']:.0f} MB")

    except Exception as e:
        print(f"\n[錯誤] 執行失敗: {e}")
        traceback.print_exc()
    finally:
        if 'model' in locals(): del model
        torch.cuda.empty_cache()
        if os.path.exists(OFFLOAD_DIR):
            shutil.rmtree(OFFLOAD_DIR)
            os.makedirs(OFFLOAD_DIR, exist_ok=True)
        print("[系統] 資源已釋放。")

if __name__ == "__main__":
    main()
