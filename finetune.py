# Install dependencies
!pip install -q transformers datasets evaluate jiwer accelerate soundfile librosa

from datasets import load_dataset, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import evaluate
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

print("GPU available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU!")

# Load dataset
print("\nLoading Gujarati dataset...")
ds = load_dataset('google/fleurs', 'gu_in')

# Load processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="gujarati", task="transcribe")

# Clean text
def clean_text(batch):
    text = batch['transcription']
    text = re.sub(r'[,।\.!?;:]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    batch['transcription'] = text
    return batch

print("Cleaning dataset...")
ds = ds.map(clean_text)

# Prepare features
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Extracting features...")
ds = ds.map(prepare_dataset, remove_columns=ds.column_names["train"], num_proc=2)

# Filter long samples
ds = ds.filter(lambda x: len(x["labels"]) <= 448)
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

# Data collator
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

# Load model
print("Loading Whisper-small...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.generation_config.language = "gujarati"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Metrics
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-small-gujarati",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("\nStarting training...")
trainer.train()

print("\nSaving model...")
model.save_pretrained("whisper-small-gujarati")
processor.save_pretrained("whisper-small-gujarati")
print("Done!")