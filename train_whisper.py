import torch
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# ── 1. Load dataset ──────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("google/fleurs", "gu_in")

# ── 2. Load processor + model ────────────────────────────────
print("Loading Whisper-small...")
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    language="Gujarati",
    task="transcribe",
    clean_up_tokenization_spaces=False
)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Tell model we're transcribing Gujarati, not translating
model.generation_config.language = "Gujarati"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# ── 3. Preprocess ────────────────────────────────────────────
# Replace your preprocess function with this:
def preprocess(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

print("Preprocessing...")
dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# Filter out examples where transcript is too long for Whisper-small
max_label_length = 448
dataset = dataset.filter(lambda x: len(x["labels"]) <= max_label_length)
print(f"After filtering — train: {len(dataset['train'])}, val: {len(dataset['validation'])}")

# ── 4. Data collator ─────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── 5. WER metric ────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    print(f"\nSample prediction : {pred_str[0]}")
    print(f"Actual transcript : {label_str[0]}")
    return {"wer": wer}

# ── 6. Training arguments ────────────────────────────────────
# CPU-friendly settings — will be slow but will work
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-gujarati-finetuned",
    per_device_train_batch_size=8,       # M1 Pro can handle bigger batch
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=200,
    gradient_checkpointing=False,        # must be False for MPS
    fp16=False,
    bf16=True,                           # MPS supports bf16 — faster than fp32
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_pin_memory=False,         # MPS doesn't support pin_memory
)

# ── 7. Trainer ───────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# ── 8. Train ─────────────────────────────────────────────────
print("\nStarting training — 200 steps on CPU, ~20-40 min...")
print("You'll see loss every 25 steps, WER at step 100 and 200\n")
trainer.train()

# ── 9. Save ──────────────────────────────────────────────────
print("\nSaving model...")
trainer.save_model("./whisper-gujarati-finetuned")
processor.save_pretrained("./whisper-gujarati-finetuned")
print("Saved to ./whisper-gujarati-finetuned ✅")