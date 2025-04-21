# GPT-2 脑-语言建模复现项目
核心目标：探索 GPT-2 模型在sft微调前后对人脑神经反应的预测能力变化。

## 项目结构
```
NarrativeProjectHomework/
├── activations_raw/            # 原始 GPT-2 提取的激活向量 (.npy)
├── activations_sft/            # 微调后 GPT-2 提取的激活向量 (.npy)
├── data/
│   └── ds002345-mini/          # 提取后的 BIDS 格式 fMRI 子集（sub-001 ~ sub-005）
├── gpt2_sft_model/             # SFT 微调后的 GPT-2 模型
├── results/                    # 模型对比结果与图表（.csv, .png 等）
├── scripts/                    # 所有核心处理脚本
│   ├── create_manifest.py                      # 生成 data_manifest 索引表
│   ├── alignments_transcript.py                # 使用 gentle 对齐转录信息
│   ├── convert_gentle_to_transcript.py         # gentle 输出转为 transcript.tsv（如需）
│   ├── extract_gpt2_embeddings_raw.py          # 提取原始 GPT-2 激活
│   ├── extract_gpt2_embeddings_sft.py          # 提取微调 GPT-2 激活
│   ├── compute_brain_score_raw.py              # RidgeCV 建模，原始模型
│   ├── compute_brain_score_sft.py              # RidgeCV 建模，微调模型
│   ├── compare_brain_scores.py                 # 新旧模型对比分析
│   └── sft_train_gpt2.py                        # SFT 微调脚本
├── transcripts/                # 所有转录文本（.tsv）
├── data_manifest.csv           # 主索引表：subject × story × fMRI路径等
├── train_narratives.txt        # GPT-2 微调使用的训练文本
├── total.ipynb                 # 汇总全部流程的主 notebook
└── README.md                   # 当前文档
```

## 实验流程

1. **数据准备**  
   下载 Narratives 数据集，构建 data_manifest 索引。
2. **转录构建**  
   基于 `.wav` 和 `.txt` 文本，使用 Gentle 对齐生成 `.json` 文件。
3. **语言模型激活提取**  
   分别对原始与微调后的 GPT-2 提取第 8 层激活向量。
4. **fMRI 映射建模**  
   使用 RidgeCV 回归，预测每个 TR 的全脑平均 BOLD 信号，计算 brain score。
5. **模型微调与比较**  
   利用 Narratives 中的文本进行 SFT 微调，对比新旧模型的对齐表现。
6. **可视化与结果分析**  
   绘制 Δ brain score 分布图，统计提升样本数、平均提升等指标。

## 环境依赖
```bash
transformers==4.x
nibabel
scikit-learn
tqdm
numpy
pandas
torch
```

使用 Python 3.8+ 及 Conda 虚拟环境进行部署。

## 参考文献
- Nastase, S. A., et al. (2023). *Evidence of a predictive coding hierarchy in the human brain listening to speech*. **Nature Human Behaviour**. [DOI:10.1038/s41562-022-01516-2](https://doi.org/10.1038/s41562-022-01516-2)