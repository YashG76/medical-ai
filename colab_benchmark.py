"""
Colab T4 GPU benchmark for the fine-tuned Gujarati Whisper model.

This measures the GPU *ceiling* (FP32 vs FP16) on a cloud T4. Note:
  - This is a CLOUD GPU. It's fine for benchmarking/training, but NOT a valid production
    runtime for MedScript (patient audio would leave the device). The on-device equivalent
    of "use the GPU" is Apple MPS on your Mac.
  - INT8 dynamic quantization is CPU-only and does NOT apply on GPU. The GPU speed lever is
    FP16 (.half()), so that's what we compare here.

HOW TO RUN ON COLAB:
  1. Runtime -> Change runtime type -> T4 GPU
  2. In a cell:
        !pip install -q transformers accelerate
        !wget -q https://raw.githubusercontent.com/<you>/medical-ai/main/colab_benchmark.py
        # (or just paste this file's contents into a cell)
        !python colab_benchmark.py
     To benchmark a real clip, upload a 16kHz wav and pass its path:
        !python colab_benchmark.py my_clip.wav
"""

import sys
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "ygotrijiya/whisper-small-gujarati-finetuned"
SAMPLE_RATE = 16000


def get_audio(path=None):
    if path:
        import scipy.io.wavfile as wav

        sr, data = wav.read(path)
        if data.dtype != np.float32:
            m = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
            data = data.astype(np.float32) / m
        if data.ndim > 1:
            data = data.mean(axis=1)
        print(f"audio: {len(data) / sr:.1f}s at {sr}Hz")
        return data
    # Synthetic 5s clip if no file given.
    t = np.linspace(0, 5, 5 * SAMPLE_RATE, endpoint=False)
    return (0.1 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 440 * t))).astype(np.float32)


def bench(model, processor, feat, label, runs=5):
    # Warm-up (first CUDA call compiles kernels).
    with torch.no_grad():
        ids = model.generate(feat, language="gujarati", task="transcribe")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(feat, language="gujarati", task="transcribe")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    mean = sum(times) / len(times)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"  {label}: {mean:.3f}s mean over {runs} runs  | {text[:80]}")
    return mean


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found. On Colab: Runtime -> Change runtime type -> T4 GPU.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    audio = get_audio(path)
    feat = processor.feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.cuda()

    print("\n--- FP32 on GPU ---")
    fp32 = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).cuda().eval()
    t_fp32 = bench(fp32, processor, feat)

    print("\n--- FP16 on GPU ---")
    fp16 = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).cuda().eval()
    t_fp16 = bench(fp16, processor, feat.half())

    print(f"\nFP16 speedup vs FP32: {t_fp32 / t_fp16:.2f}x")
    print("Reminder: this is the cloud-GPU ceiling. The local privacy-safe equivalent is MPS.")


if __name__ == "__main__":
    main()
