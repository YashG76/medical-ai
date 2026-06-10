from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import torch
device = "cpu"
print(f"Using device: {device}")
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print("Loading cleaned dataset...")
ds = load_from_disk('data/gujarati_clean')

print("Loading Whisper processor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="gujarati", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

print("Preparing dataset (this takes a few minutes)...")
ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=1)

# Filter out samples where labels are too long for Whisper
print("Filtering long samples...")
ds = ds.filter(lambda x: len(x["labels"]) <= 448)
print(f"After filtering - Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("Loading Whisper-small model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = model.to(device)
model.generation_config.language = "gujarati"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

training_args = Seq2SeqTrainingArguments(
    output_dir="models/whisper-small-gujarati",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=200,
    gradient_checkpointing=False,
    fp16=False,
    bf16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

import evaluate
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("Starting fine-tuning...")
print("This will take 30-60 minutes on Mac. Go make chai ☕")
trainer.train()

print("Saving model...")
model.save_pretrained("models/whisper-small-gujarati")
processor.save_pretrained("models/whisper-small-gujarati")
print("Done! Model saved to models/whisper-small-gujarati")