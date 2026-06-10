import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline

# Replace with your HuggingFace username
MODEL = "ygotrijiya/whisper-small-gujarati-finetuned"

print("Loading your fine-tuned model...")
pipe = pipeline("automatic-speech-recognition", model=MODEL)
print("Model loaded!")

def record_audio(duration=5, sample_rate=16000):
    print(f"\nRecording for {duration} seconds...")
    print("Speak now! 🎤")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')
    sd.wait()
    print("Recording done!")
    return audio.flatten()

# Record and transcribe
audio = record_audio(duration=5)

print("\nTranscribing...")
result = pipe(audio, generate_kwargs={"language": "gujarati", "task": "transcribe"})
print(f"\nYou said: {result['text']}")