#!/usr/bin/env python3
import os
import whisper
import torch

# 1. Pick device and load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Loading model on {device}…")
model = whisper.load_model("tiny", device=device)

# 2. Point to your audio file
audio_file = "/home/gao/transcription_whisper_speech_to_text/mymp3.mp3"
if not os.path.exists(audio_file):
    raise FileNotFoundError(f"Audio file not found: {audio_file!r}")

print(f"⏳ Starting transcription of {audio_file}…")

# 3. Run transcription (returns full text + segment list)
result = model.transcribe(audio_file, language="zh",verbose=True)

# 4. Print each segment as it's decoded
for segment in result["segments"]:
    print(f"[{segment.start:.1f}s → {segment.end:.1f}s] {segment.text}")

# 5. Save the full transcription to outputs/<basename>.txt
full_text = result["text"]
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
base = os.path.splitext(os.path.basename(audio_file))[0]
out_path = os.path.join(out_dir, f"{base}.txt")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ Transcription saved to {out_path}")
