import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import noisereduce as nr

SAMPLE_RATE = 16000

# ── Step 1: Record audio ──────────────────────────────────
print("🎤 Press Enter to START recording (make some background noise!)")
input()

recording = []
def callback(indata, frames, time, status):
    recording.append(indata.copy())

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
    print("🔴 Recording... Press Enter to STOP")
    input()

# Combine chunks
audio = np.concatenate(recording, axis=0).flatten()

# Save original (noisy) version
wav.write("noisy.wav", SAMPLE_RATE, audio)
print("💾 Saved noisy.wav")

# ── Step 2: Apply noise cancellation ─────────────────────
print("🧹 Applying noise cancellation...")

# noisereduce automatically finds the noise profile
clean_audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)

# Save clean version
wav.write("clean.wav", SAMPLE_RATE, clean_audio.astype(np.float32))
print("✅ Saved clean.wav")

# ── Step 3: Transcribe both and compare ──────────────────
import whisper
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("\n⏳ Loading Whisper...")
model = whisper.load_model("base")

print("\n📝 Transcribing NOISY version...")
noisy_result = model.transcribe("noisy.wav")
print(f"Noisy:  {noisy_result['text']}")

print("\n📝 Transcribing CLEAN version...")
clean_result = model.transcribe("clean.wav")
print(f"Clean:  {clean_result['text']}")

print("\n✅ Done! Compare the two transcripts above.")