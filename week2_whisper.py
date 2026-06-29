from datasets import load_dataset
from transformers import WhisperProcessor

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# Load dataset
dataset = load_dataset("google/fleurs", "gu_in")

# Load Whisper processor
# Processor = Feature Extractor (audio→numbers) + Tokenizer (text→tokens)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Gujarati",
    task="transcribe"
)

def preprocess(batch):
    # Audio → mel spectrogram (what Whisper actually sees)
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Text → token IDs (numbers Whisper predicts)
    batch["labels"] = processor.tokenizer(
        batch["transcription"]
    ).input_ids

    return batch

print("Preprocessing train set...")
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

print("\nDone! Preprocessed example:")
print("input_features shape:", len(dataset["train"][0]["input_features"]), "x", len(dataset["train"][0]["input_features"][0]))
print("labels (first 10 tokens):", dataset["train"][0]["labels"][:10])
print("labels decoded back:", processor.tokenizer.decode(dataset["train"][0]["labels"]))

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input audio features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels (text tokens), replace padding token with -100
        # -100 tells PyTorch to ignore those positions in loss calculation
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the decoder start token from labels if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Initialize it
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Quick test - collate 2 examples and check shapes
sample_batch = data_collator([dataset["train"][0], dataset["train"][1]])
print("Batch input shape:", sample_batch["input_features"].shape)
print("Batch labels shape:", sample_batch["labels"].shape)
print("\nData collator working ✅")