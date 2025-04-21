import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from tqdm import tqdm

# === 配置路径 ===
manifest_path = "data_manifest.csv"
activation_dir = "activations_sft"
transcript_dir = "transcripts"
result_path = "results/brain_score_sft.csv"
os.makedirs("results", exist_ok=True)

# === 加载 manifest ===
manifest = pd.read_csv(manifest_path)
TR = 1.5  # repetition time
results = []

for i, row in tqdm(manifest.iterrows(), total=len(manifest)):
    subj = row['subject']
    story = row['story_name']
    bold_path = row['bold_path']
    transcript_path = os.path.join(transcript_dir, f"{story}_transcripts.tsv")
    activation_path = os.path.join(activation_dir, f"{story}.npy")

    if not os.path.exists(bold_path) or not os.path.exists(transcript_path) or not os.path.exists(activation_path):
        continue

    try:
        # 1. fMRI 全脑平均信号
        fmri = nib.load(bold_path).get_fdata()
        if fmri.size == 0:
            print(f"[跳过] {subj}-{story}: 空的 fMRI 文件")
            continue
        ts = fmri.reshape(-1, fmri.shape[-1]).mean(axis=0)

        # 2. 对齐时序信息
        df = pd.read_csv(transcript_path, sep='\t')
        if df.empty:
            print(f"[跳过] {subj}-{story}: transcript.tsv 为空")
            continue
        X = np.load(activation_path)
        if X.size == 0:
            print(f"[跳过] {subj}-{story}: GPT2 向量为空")
            continue

        df = df.iloc[:X.shape[0]]  # 限制行数与 GPT-2 输出一致
        onsets = (df["onset"] // TR).astype(int)
        valid = onsets < len(ts)
        onsets = onsets[valid]
        X = X[valid.values]
        Y = ts[onsets]

        if len(Y) < 10 or len(X) < 10:
            print(f"[跳过] {subj}-{story}: 有效样本不足")
            continue

        # 4. RidgeCV 预测 + Brain Score
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5))
        model.fit(X, Y)
        Y_pred = model.predict(X)
        r, _ = pearsonr(Y, Y_pred)

        results.append({"subject": subj, "story": story, "brain_score": round(r, 4)})

    except Exception as e:
        print(f"[跳过] {subj}-{story}: {e}")

# 保存结果
pd.DataFrame(results).to_csv(result_path, index=False)
print(f"✅ Brain Score 已保存到 {result_path}")