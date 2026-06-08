import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import threading

# Settings
SAMPLE_RATE = 16000  # 16kHz — what Whisper expects
CHANNELS = 1         # mono audio

# This list will hold all recorded audio chunks
recorded_chunks = []
is_recording = False

def record_audio():
    """Runs in background — keeps recording until stopped."""
    global is_recording

    def callback(indata, frames, time, status):
        if is_recording:
            recorded_chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        while is_recording:
            sd.sleep(100)  # check every 100ms

# --- Main flow ---
print("🎤 Press Enter to START recording...")
input()

# Start recording in background thread
is_recording = True
recorded_chunks = []
thread = threading.Thread(target=record_audio)
thread.start()

print("🔴 Recording... Press Enter to STOP.")
input()

# Stop recording
is_recording = False
thread.join()

# Combine all chunks into one audio array
audio_data = np.concatenate(recorded_chunks, axis=0)

# Save as .wav file
output_path = "output.wav"
wav.write(output_path, SAMPLE_RATE, audio_data)

print(f"✅ Saved to {output_path} — {len(audio_data)/SAMPLE_RATE:.1f} seconds recorded")