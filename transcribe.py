import whisper

# Load the base model (downloads ~145MB first time — normal)
print("⏳ Loading Whisper model...")
model = whisper.load_model("small")
print("✅ Model loaded!")

# Transcribe your recording
print("🎤 Transcribing...")
result = model.transcribe("output.wav", language="gu")

# Print the result
print("\n📝 Transcript:")
print(result["text"])

# Also print which language Whisper detected
print(f"\n🌍 Detected language: {result['language']}")