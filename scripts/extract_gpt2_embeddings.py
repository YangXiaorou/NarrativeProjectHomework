# Step 4：使用原始 GPT-2 提取每个 story 的特征表示
import os
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch
from tqdm import tqdm

transcript_dir = "transcripts"
output_dir = "activations_raw"
os.makedirs(output_dir, exist_ok=True)

# 使用原始 GPT-2 模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
model.eval()

for fname in tqdm(os.listdir(transcript_dir)):
    if not fname.endswith("_transcripts.tsv"):
        continue

    story = fname.replace("_transcripts.tsv", "")
    try:
        df = pd.read_csv(os.path.join(transcript_dir, fname), sep="\t")
        words = df["stimulus"].astype(str).tolist()

        # 拼成长句并 tokenize
        full_text = " ".join(words)
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            out = model(**tokens)
            layer8 = out.hidden_states[8].squeeze(0)  # (seq_len, 768)

        # 限制最大长度为 N 词（避免错位）
        vecs = layer8[:len(words)].cpu().numpy()
        np.save(os.path.join(output_dir, f"{story}.npy"), vecs)
        print(f"[保存] {story}.npy → shape: {vecs.shape}")

    except Exception as e:
        print(f"[错误] {story}: {e}")
