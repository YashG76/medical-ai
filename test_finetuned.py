import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np

# Load your model
processor = WhisperProcessor.from_pretrained("ygotrijiya/whisper-small-gujarati-medscript")
model = WhisperForConditionalGeneration.from_pretrained("ygotrijiya/whisper-small-gujarati-medscript")
model.eval()

# This time — speak a MIXED sentence when recording
# Something like a doctor would actually say:
# "દર્દીને fever છે અને BP 140 over 90 છે"
# "Patient ને chest pain છે"
# "મારે antibiotic prescribe કરવી છે"

SAMPLE_RATE = 16000
DURATION = 7

print("="*50)
print("CODE-SWITCHING TEST")
print("="*50)
print("\nSpeak a MIXED Gujarati+English sentence like a doctor would:")
print("Example: દર્દીને fever છે અને BP high છે")
print("Example: patient ને chest pain છે")
print("Example: મારે antibiotic આપવી છે")
print()

input("Press Enter to start recording...")
print("🎙️ Speak now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.float32
)
sd.wait()
print("Done!\n")

audio_input = audio.flatten()

# Test 1 — forced Gujarati (current behaviour)
input_features = processor.feature_extractor(
    audio_input,
    sampling_rate=SAMPLE_RATE,
    return_tensors="pt"
).input_features

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        language="gujarati",
        task="transcribe",
    )
result_gu = processor.tokenizer.batch_decode(
    predicted_ids, skip_special_tokens=True
)[0]

# Test 2 — no forced language (model decides)
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        task="transcribe",
    )
result_auto = processor.tokenizer.batch_decode(
    predicted_ids, skip_special_tokens=True
)[0]

# Test 3 — transcribe + translate to English
with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        task="translate",   # Whisper translates to English
    )
result_translate = processor.tokenizer.batch_decode(
    predicted_ids, skip_special_tokens=True
)[0]

print(f"Forced Gujarati : {result_gu}")
print(f"Auto language   : {result_auto}")
print(f"Translated      : {result_translate}")