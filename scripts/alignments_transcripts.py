import os
import json
import pandas as pd

gentle_dir = "F:/NarrativesDataset/ds002345-download/stimuli/alignments"
output_dir = "transcripts"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(gentle_dir):
    if not fname.endswith("_align.json"):
        continue

    story = fname.replace("_align.json", "")
    json_path = os.path.join(gentle_dir, fname)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = data.get("words", [])
    rows = []

    for w in words:
        if w.get("case") != "success":
            continue
        start = round(w["start"], 3)
        end = round(w["end"], 3)
        dur = round(end - start, 3)
        rows.append({"onset": start, "duration": dur, "stimulus": w["word"]})

    if not rows:
        print(f"[跳过] 无有效对齐：{story}")
        continue

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{story}_transcripts.tsv"), sep="\t", index=False)
    print(f"[完成] {story}_transcripts.tsv → {len(df)} 词")
