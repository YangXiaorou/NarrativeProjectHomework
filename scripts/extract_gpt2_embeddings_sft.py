# Step 7：使用 SFT 后的 GPT-2 重新提取特征并建模脑评分
import os
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch
from tqdm import tqdm

transcript_dir = "transcripts"
output_dir = "activations_sft"
model_dir = "gpt2_sft_model"
os.makedirs(output_dir, exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2Model.from_pretrained(model_dir, output_hidden_states=True)
model.eval()

for fname in tqdm(os.listdir(transcript_dir)):
    if not fname.endswith("_transcripts.tsv"):
        continue

    story = fname.replace("_transcripts.tsv", "")
    try:
        df = pd.read_csv(os.path.join(transcript_dir, fname), sep="\t")
        words = df["stimulus"].astype(str).tolist()

        text = " ".join(words)
        tokens = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**tokens)
            hidden = outputs.hidden_states[8].squeeze(0)

        hidden = hidden[:len(words)].cpu().numpy()
        np.save(os.path.join(output_dir, f"{story}.npy"), hidden)
        print(f"[保存] {story}.npy → shape: {hidden.shape}")

    except Exception as e:
        print(f"[错误] {story}: {e}")
