"""
Week 2 tail — CPU Quantization (benchmark + demo)

Loads the fine-tuned Gujarati Whisper model, applies INT8 dynamic quantization to all
nn.Linear layers, prints a size comparison, and benchmarks FP32 vs INT8 inference latency
on CPU.

Usage:
    python quantize_whisper.py              # quantize + benchmark
    python quantize_whisper.py --save       # also save the INT8 state dict to disk
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "ygotrijiya/whisper-small-gujarati-finetuned"
SAVE_DIR = "models/whisper-small-gujarati"
QUANTIZED_PATH = os.path.join(SAVE_DIR, "quantized_int8.pt")

# Keep everything on CPU — mirrors the privacy / deployment target.
DEVICE = "cpu"

# On Apple Silicon the quantized kernels live in the qnnpack backend; it must be selected
# explicitly or quantize_dynamic fails with "Didn't find engine ... NoQEngine".
torch.backends.quantized.engine = "qnnpack"


def load_fp32():
    """Load the FP32 processor + model from the HuggingFace Hub cache."""
    print("Loading FP32 model (this is the ~967 MB original)...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(DEVICE).eval()
    return processor, model


def quantize(model):
    """Apply INT8 dynamic quantization to every nn.Linear layer."""
    print("Quantizing nn.Linear layers to INT8...")
    q_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return q_model.eval()


def model_size_mb(model):
    """Rough in-memory size of the model's parameters + buffers, in MB."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    # Dynamically-quantized Linear weights live in packed buffers that the loops above
    # don't fully see, so also fall back to the serialized state-dict size.
    return total_bytes / (1024 ** 2)


def state_dict_size_mb(model):
    """Size of the serialized state dict — the honest on-disk/in-RAM footprint."""
    import io

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def make_test_audio(seconds=5, sample_rate=16000):
    """A synthetic 5-second 16kHz signal so the benchmark needs no microphone."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    # A couple of tones so it isn't pure silence (silence can short-circuit decoding).
    audio = 0.1 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 440 * t))
    return audio.astype(np.float32)


def transcribe_once(model, processor, input_features):
    with torch.no_grad():
        ids = model.generate(input_features, language="gujarati", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def benchmark(fp32_model, q_model, processor, runs=3):
    audio = make_test_audio()
    input_features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    def time_model(model, label):
        # Warm-up (first generate is always slower).
        transcribe_once(model, processor, input_features)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            transcribe_once(model, processor, input_features)
            times.append(time.perf_counter() - start)
        mean = sum(times) / len(times)
        print(f"  {label}: {mean:.2f}s mean over {runs} runs")
        return mean

    print(f"\nBenchmarking on a synthetic 5s clip ({runs} runs each)...")
    fp32_time = time_model(fp32_model, "FP32")
    int8_time = time_model(q_model, "INT8")
    if int8_time > 0:
        print(f"\n  Speedup: {fp32_time / int8_time:.2f}x faster")


def save_quantized(q_model, processor):
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(q_model.state_dict(), QUANTIZED_PATH)
    processor.save_pretrained(SAVE_DIR)
    print(f"\nSaved INT8 state dict -> {QUANTIZED_PATH}")
    # NOTE: to reload, you must recreate the quantized structure BEFORE load_state_dict:
    #   model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    #   model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    #   model.load_state_dict(torch.load(QUANTIZED_PATH))
    # In practice quantize-on-load (see transcribe_fast.py) is simpler — quantization is ~1s.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="save INT8 state dict to disk")
    args = parser.parse_args()

    processor, fp32_model = load_fp32()
    q_model = quantize(fp32_model)

    fp32_mb = state_dict_size_mb(fp32_model)
    int8_mb = state_dict_size_mb(q_model)
    print(f"\nSize (serialized state dict):")
    print(f"  FP32: {fp32_mb:.0f} MB")
    print(f"  INT8: {int8_mb:.0f} MB  ({fp32_mb / int8_mb:.1f}x smaller)")

    benchmark(fp32_model, q_model, processor)

    if args.save:
        save_quantized(q_model, processor)


if __name__ == "__main__":
    main()
