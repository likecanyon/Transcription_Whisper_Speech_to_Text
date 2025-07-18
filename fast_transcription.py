#!/usr/bin/env python3
import os
from faster_whisper import WhisperModel

# 1. Select device
device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", None) or True else "cpu"
print(f"⏳ Loading faster-whisper model on {device}…")

# 2. Load the model (you can choose "tiny", "small", "medium", "large", etc.)
model = WhisperModel("large", device=device, compute_type="float16")


audio_file = "/home/gao/transcription_whisper_speech_to_text/demo.mp3"
if not os.path.isfile(audio_file):
    raise FileNotFoundError(f"Couldn't find audio at {audio_file!r}")

print(f"⏳ Starting transcription of {audio_file}…")
segments_iter, info = model.transcribe(
    audio_file,
    beam_size=5,
    language="zh",
    without_timestamps=False,
)

# Stream to console & accumulate
segments = []
print("⏳ Decoding…")
for seg in segments_iter:
    print(f"[{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")
    segments.append(seg)


full_text = "".join(seg.text for seg in segments)
print("Full text length:", len(full_text))

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "demo.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ Transcription saved to {out_path}")
