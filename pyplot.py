import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. 定義原始數據
raw_data = """Model Run Total_Duration(s)
gpt-oss-20b-MXFP4 1 3.2315
gpt-oss-20b-MXFP4 2 2.8171
gpt-oss-20b-MXFP4 3 2.7708
gpt-oss-20b-MXFP4 4 2.7785
gpt-oss-20b-MXFP4 5 2.7445
gpt-oss-20b-Manual-NF4 1 3.7399
gpt-oss-20b-Manual-NF4 2 3.3219
gpt-oss-20b-Manual-NF4 3 3.3368
gpt-oss-20b-Manual-NF4 4 3.2607
gpt-oss-20b-Manual-NF4 5 3.2901"""

# 2. 將文字數據轉換為 Pandas DataFrame
# sep='\s+' 表示使用空格（一個或多個）作為分隔符
df = pd.read_csv(io.StringIO(raw_data), sep='\s+')

# 3. 設定繪圖風格
sns.set_theme(style="whitegrid") # 設定白色網格背景

# 4. 建立圖表圖層
plt.figure(figsize=(12, 7)) # 設定圖表大小 (寬, 高)

# 5. 使用 Seaborn 繪製折線圖
# x: X軸數據 (Run)
# y: Y軸數據 (Total_Duration(s))
# hue: 分組依據，不同 Model 不同顏色
# data: 數據來源
# marker='o': 在數據點上加上圓圈標記
chart = sns.lineplot(
    x='Run', 
    y='Total_Duration(s)', 
    hue='Model', 
    data=df, 
    marker='o', 
    linewidth=2.5,
    markersize=8
)

# 6. 進一步優化圖表細節
plt.title('GPT-OSS-20B Total Duration', fontsize=16, fontweight='bold')
plt.xlabel('Run', fontsize=12)
plt.ylabel('Total_Duration (s)', fontsize=12)

# 確保 X 軸只顯示整數 1 到 5
plt.xticks(df['Run'].unique())

# 調整圖例位置（放在右下角，避免遮擋線條）
plt.legend(title='model', loc='upper right')

# 7. 顯示圖表
plt.tight_layout() # 自動調整佈局，防止標籤被切到
plt.show()
