import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

# ================= 路徑與環境變數設定 =================
BASE_DIR = "/home/s3734/heng_project"
CACHE_DIR = os.path.join(BASE_DIR, "hf_cache")
LOG_FILE = os.path.join(BASE_DIR, "hardware_benchmark_log.csv")

os.makedirs(BASE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# ================= 量化配置 (核心修改) =================
# 使用 4-bit 量化 (NF4)，這能將模型體積縮小約 4 倍
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # 計算時使用 bfloat16 提升速度
    bnb_4bit_quant_type="nf4",             # 使用 Normal Float 4，精準度較佳
    bnb_4bit_use_double_quant=True,       # 第二次量化量化常數，進一步省空間
)

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
USE_PROMPT = "請撰寫一篇約 2000 字的深度分析文章，探討量子計算在未來十年內的應用。"

# (其餘 HardwareMonitor 與 OllamaTimingStreamer 類別保持不變，此處略過以節省空間)

# ================= 主程式修改 =================
def main():
    # ... (前面的 LOG 檔案初始化邏輯保持不變) ...

    choice = input("請輸入要測試的模型編號 (輸入 q 離開): ")
    if choice.lower() == 'q' or choice not in MODELS_TO_TEST:
        return

    model_name, repo_id = MODELS_TO_TEST[choice]
    print(f"\n準備以 4-bit 量化模式載入模型: {model_name}")
    
    monitor = HardwareMonitor()

    try:
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        # 核心載入邏輯變更
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            quantization_config=quantization_config, # 注入量化配置
            device_map="auto",                       # 自動分配 GPU 資源
            trust_remote_code=True
        )
        
        load_duration = time.time() - load_start
        print(f"量化模型載入完成，耗時: {load_duration:.2f} 秒")

        # 確保輸入數據在正確的設備上
        inputs = tokenizer(USE_PROMPT, return_tensors="pt").to(model.device)
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

            # ... (後續計算 Duration 與寫入 CSV 的邏輯保持不變) ...
            # 建議在 CSV 記錄中標註此為量化版本

    except Exception as e:
        print(f"執行失敗: {e}")
        if 'monitor' in locals(): monitor.stop()

    finally:
        # 清理資源
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        # ... (其餘清理邏輯) ...

if __name__ == "__main__":
    main()
