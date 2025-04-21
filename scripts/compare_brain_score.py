# Step 8：对比原始 GPT-2 与微调 GPT-2 的 Brain Score 表现
import pandas as pd
import matplotlib.pyplot as plt
import os

# 加载两个 CSV 文件
df_raw = pd.read_csv("results/brain_score_raw.csv").rename(columns={"brain_score": "raw_score"})
df_sft = pd.read_csv("results/brain_score_sft.csv").rename(columns={"brain_score": "sft_score"})

# 合并数据
df = pd.merge(df_raw, df_sft, on=["subject", "story"])
df["delta"] = df["sft_score"] - df["raw_score"]

# 按差值排序
df.sort_values("delta", ascending=False, inplace=True)
df.to_csv("results/brain_score_comparison.csv", index=False)
print("✅ 差值对比表已保存：results/brain_score_comparison.csv")

# 画图
plt.figure(figsize=(18, 6))  # 更宽画布
labels = df["subject"] + "-" + df["story"]
plt.bar(labels, df["delta"], color="skyblue")
plt.axhline(0, color="gray", linestyle="--")
plt.ylabel("Δ Brain Score (SFT - Raw)", fontsize=12)
plt.title("GPT-2 SFT 微调后的脑得分提升效果", fontsize=14)
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig("results/brain_score_comparison.png", dpi=300)
plt.show()
