# Step 6：微调 GPT-2（SFT）并重新提取语言特征
# 6.1 脚本：生成训练语料 train_narratives.txt
import os
import pandas as pd

transcript_dir = "transcripts"
output_path = "train_narratives.txt"
lines = []

for fname in os.listdir(transcript_dir):
    if fname.endswith("_transcripts.tsv"):
        df = pd.read_csv(os.path.join(transcript_dir, fname), sep="\t")
        words = df["stimulus"].astype(str).tolist()
        line = " ".join(words)
        lines.append(line)

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"训练语料已生成：{output_path}")


# 6.2 脚本：微调 GPT-2 模型 scripts/

import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 路径配置
model_name = "gpt2"
save_path = "gpt2_sft_model"
corpus_path = "train_narratives.txt"
os.makedirs(save_path, exist_ok=True)

# 加载模型和 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# 准备训练数据
with open(corpus_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=128)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
labels = input_ids.clone()

# 训练参数
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 3
batch_size = 2

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for i in range(0, len(input_ids), batch_size):
        batch_ids = input_ids[i:i+batch_size]
        batch_mask = attention_mask[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(f"  step {i//batch_size+1}: loss = {loss.item():.4f}")

# 保存
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("微调完成，模型保存在 gpt2_sft_model/")

