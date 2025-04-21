# 提取 manifest（受试者 × 故事 × 路径表）
import os
import csv

base_dir = "F:/NarrativesDataset/ds002345-download"
manifest = []

# 获取所有 sub- 开头的受试者文件夹
subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("sub-")])

for subj in subjects:
    func_dir = os.path.join(base_dir, subj, "func")
    if not os.path.exists(func_dir):
        continue
    for fname in os.listdir(func_dir):
        if fname.endswith("_bold.nii.gz"):
            bold_path = os.path.join(func_dir, fname)
            events_name = fname.replace("_bold.nii.gz", "_events.tsv")
            events_path = os.path.join(func_dir, events_name)
            if not os.path.exists(events_path):
                continue
            story = fname.replace(f"{subj}_task-", "").replace("_bold.nii.gz", "")
            manifest.append([subj, story, bold_path, events_path])

# 保存
out_path = "data_manifest.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "story_name", "bold_path", "events_path"])
    writer.writerows(manifest)

print(f"✅ 提取完毕，共 {len(manifest)} 条记录，已保存到：{out_path}")
