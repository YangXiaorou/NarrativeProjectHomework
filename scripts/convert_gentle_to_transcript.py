# 基于 Gentle 输出构建标准 transcript 文件
import os
import webbrowser
import pyperclip  # pip install pyperclip

text_dir = "F:/NarrativesDataset/ds002345-download/stimuli/transcripts"
audio_dir = "F:/NarrativesDataset/ds002345-download/stimuli/audio"

for fname in os.listdir(text_dir):
    if not fname.endswith(".txt"):
        continue

    story = fname.replace(".txt", "")
    txt_path = os.path.join(text_dir, fname)
    wav_path = os.path.join(audio_dir, f"{story}_audio.wav")

    if not os.path.exists(wav_path):
        print(f"[跳过] 缺少音频文件：{story}")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read()
        pyperclip.copy(transcript)  # 自动复制内容

    print(f"\n🧠 对齐：{story}")
    print(f"   - ⌨️ 文本已复制，直接 Ctrl + V 粘贴到 Transcript 框")
    print(f"   - 🎵 上传音频：{wav_path}")
    print("   - 🖱️ 点击 Align，完成后右键另存为 JSON 文件")

    webbrowser.open("http://localhost:8765")
    input("⏸️ 回车继续下一个 >>> ")


import os

alignments_path = r"F:\NarrativesDataset\ds002345-download\stimuli\alignments"
audio_path = r"F:\NarrativesDataset\ds002345-download\stimuli\audio"

for root, dirs, files in os.walk(alignments_path):
    for file in files:
        if file.endswith("_align.json"):
            base_name = file[:-len("_align.json")]
            audio_file = base_name + "_audio.wav"
            audio_file_path = os.path.join(audio_path, audio_file)
            if not os.path.exists(audio_file_path):
                print(f"未找到对应的音频文件: {audio_file_path}")