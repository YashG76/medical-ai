"""
Fast Gujarati transcription using the INT8-quantized fine-tuned Whisper model.

Uses the "quantize-on-load" pattern: load the FP32 model from the HuggingFace cache once,
quantize it to INT8 in memory (~1s), then transcribe on CPU. This is the reusable inference
entry point going forward (replaces transcribe.py).

Usage:
    python transcribe_fast.py                # transcribes output.wav
    python transcribe_fast.py path/to.wav    # transcribes the given file
"""

import sys
import time

import numpy as np
import scipy.io.wavfile as wav
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "ygotrijiya/whisper-small-gujarati-finetuned"
SAMPLE_RATE = 16000

# Device selection:
#   MPS  = Apple GPU (on-device, ~2x faster than CPU, privacy-safe) ← best on Mac
#   CPU  = fallback
# We never use CUDA here — cloud GPU is not a valid runtime for patient audio.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cpu"   # CUDA means we're on a cloud machine — stay on CPU for privacy
else:
    DEVICE = "cpu"

print(f"Loading fine-tuned model onto {DEVICE.upper()}...")
PROCESSOR = WhisperProcessor.from_pretrained(MODEL_ID)
MODEL = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE).eval()
print("Model ready.")


def load_wav(path):
    """Read a WAV file and return a mono float32 array at 16kHz."""
    sr, data = wav.read(path)
    if data.dtype != np.float32:
        # Normalize integer PCM to [-1, 1].
        max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
        data = data.astype(np.float32) / max_val
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo -> mono
    if sr != SAMPLE_RATE:
        print(f"Warning: file is {sr} Hz, expected {SAMPLE_RATE} Hz. Resample for best results.")
    return data


def transcribe(wav_path):
    """Transcribe a WAV file. Returns (text, elapsed_seconds)."""
    audio = load_wav(wav_path)
    input_features = PROCESSOR.feature_extractor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        ids = MODEL.generate(input_features, language="gujarati", task="transcribe")
    if DEVICE == "mps":
        torch.mps.synchronize()   # wait for GPU to finish before stopping the clock
    elapsed = time.perf_counter() - start

    text = PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0]
    return text, elapsed


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output.wav"
    print(f"\nTranscribing {path} ...")
    text, elapsed = transcribe(path)
    print(f"\nTranscript: {text}")
    print(f"Inference time: {elapsed:.2f}s")
