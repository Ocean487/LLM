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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
prompt = "請撰寫一篇約 2000 字的深度分析文章，探討量子計算在未來十年內，將如何顛覆製藥和材料科學這兩個領域。文章需包含基本原理介紹、潛在應用案例，以及目前面臨的主要技術挑戰。"
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
    if "Gemma2" in model.__class__.__name__:
        return ["Gemma2DecoderLayer"]
    elif "Llama" in model.__class__.__name__:
        return ["LlamaDecoderLayer"]
    elif "Qwen2" in model.__class__.__name__:
        return ["Qwen2DecoderLayer"]
    return getattr(model, "_no_split_modules", None)

# ================= 工具類別 =================
def save_to_csv(data):
    fieldnames = ["Timestamp", "Model", "Run", "Total_Duration(s)", "GPU_MB", "CPU_Util", "LoadTime_Sec"]
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

# =================  NF4 =================
def convert_to_nf4_manual(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
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
            if module.weight.device.type == 'meta':
                
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
    print(" NF4 量化")
    print("="*50)
    for key, (name, _) in MODELS_TO_TEST.items():
        print(f"[{key}] {name}")
    
    choice = input("選擇編號: ").strip()
    if choice not in MODELS_TO_TEST: return
    model_name, repo_id = MODELS_TO_TEST[choice]

    torch.cuda.empty_cache()
    monitor = HardwareMonitor()

    
    current_model_tag = f"{model_name}-Manual-NF4"

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        
        print(f"正在模型載入")
        
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="cpu", 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        print(f"正在執行NF4")
        convert_to_nf4_manual(model)

        
        no_split = get_no_split_modules(model)
        
        max_mem = {0: "12GiB", "cpu": "128GiB"} 
        device_map = infer_auto_device_map(
            model,
            max_memory=max_mem,
            no_split_module_classes=no_split
        )
        
        print("正在將模型轉至 GPU")
        model = dispatch_model(model, device_map=device_map, offload_dir=OFFLOAD_DIR)

        load_duration = time.time() - load_start
        print(f"[成功] {current_model_tag} 載入完成，耗時: {load_duration:.2f}s")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
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

            gen_duration = gen_end - gen_start
            
            save_to_csv({
                "Model": current_model_tag,
                "Run": run,
                "Total_Duration(s)": round(gen_duration, 4),
                "GPU_MB": round(avg_hw['GPU_Mem_MB'], 0),
                "CPU_Util": round(avg_hw['CPU_Util'], 1),
                "LoadTime_Sec": round(load_duration, 2)
            })
            print(f"Run {run}: 耗時 {gen_duration:.2f} 秒 | 模式:  NF4 | 顯存: {avg_hw['GPU_Mem_MB']:.0f} MB")

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

