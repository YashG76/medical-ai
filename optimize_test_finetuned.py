import time
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np

processor = WhisperProcessor.from_pretrained("ygotrijiya/whisper-small-gujarati-medscript")
model = WhisperForConditionalGeneration.from_pretrained("ygotrijiya/whisper-small-gujarati-medscript")
model.eval()

SAMPLE_RATE = 16000
DURATION = 10

input("Press Enter to record 10 seconds...")
print("🎙️ Speak Gujarati for 10 seconds!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
sd.wait()
print("Done! Running speed tests...\n")

audio_input = audio.flatten()
input_features = processor.feature_extractor(
    audio_input, sampling_rate=SAMPLE_RATE, return_tensors="pt"
).input_features

# ── Test 1: Baseline fp32 ─────────────────────────────────
print("Test 1 — Baseline fp32...")
times = []
for i in range(3):
    start = time.time()
    with torch.no_grad():
        ids = model.generate(input_features, language="gujarati", task="transcribe")
    times.append(time.time() - start)
baseline = sum(times) / len(times)
text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
print(f"  Average : {baseline:.2f}s")
print(f"  RTF     : {baseline/DURATION:.2f}x")
print(f"  Output  : {text}\n")

# ── Test 2: torch.compile (PyTorch 2.0+) ─────────────────
print("Test 2 — torch.compile...")
try:
    compiled_model = torch.compile(model)
    # warmup run
    with torch.no_grad():
        compiled_model.generate(input_features, language="gujarati", task="transcribe")
    times = []
    for i in range(3):
        start = time.time()
        with torch.no_grad():
            ids = compiled_model.generate(input_features, language="gujarati", task="transcribe")
        times.append(time.time() - start)
    compiled_time = sum(times) / len(times)
    text2 = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"  Average : {compiled_time:.2f}s")
    print(f"  RTF     : {compiled_time/DURATION:.2f}x")
    print(f"  Speedup : {baseline/compiled_time:.2f}x faster")
    print(f"  Output  : {text2}\n")
except Exception as e:
    print(f"  Skipped: {e}\n")

# ── Test 3: fp16 on MPS ───────────────────────────────────
print("Test 3 — fp16 on MPS (Apple GPU)...")
try:
    mps_model = WhisperForConditionalGeneration.from_pretrained(
        "ygotrijiya/whisper-small-gujarati-medscript",
        torch_dtype=torch.float16
    ).to("mps")
    mps_model.eval()
    mps_features = input_features.to("mps").half()

    # warmup
    with torch.no_grad():
        mps_model.generate(mps_features, language="gujarati", task="transcribe")

    times = []
    for i in range(3):
        start = time.time()
        with torch.no_grad():
            ids = mps_model.generate(mps_features, language="gujarati", task="transcribe")
        times.append(time.time() - start)
    mps_time = sum(times) / len(times)
    text3 = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"  Average : {mps_time:.2f}s")
    print(f"  RTF     : {mps_time/DURATION:.2f}x")
    print(f"  Speedup : {baseline/mps_time:.2f}x faster")
    print(f"  Output  : {text3}\n")
except Exception as e:
    print(f"  Skipped: {e}\n")

# ── Summary ───────────────────────────────────────────────
print("="*50)
print("SUMMARY")
print("="*50)
print(f"Audio duration  : {DURATION}s")
print(f"fp32 CPU        : {baseline:.2f}s  (RTF: {baseline/DURATION:.2f}x)")
try:
    print(f"torch.compile   : {compiled_time:.2f}s  (RTF: {compiled_time/DURATION:.2f}x)")
    print(f"fp16 MPS        : {mps_time:.2f}s  (RTF: {mps_time/DURATION:.2f}x)")
except:
    pass
print("\nRTF < 1.0 = faster than real time ✅")
print("RTF > 1.0 = slower than real time ⚠️")