# Week 3 — SOAP Note Generator (your biggest week)

**Goal:** turn a doctor-patient transcript into a structured **SOAP note** using a small
language model you fine-tune yourself with LoRA — running locally on CPU.

> This is two phases and you must not skip Phase 1. **Phase 1 = learn the SLM/LoRA stack.
> Phase 2 = build the generator.** Budget ~2–3 days for Phase 1 before you touch the build.

---

## ⚠️ Dev A dependency callout (read this first)

Everything in **Dev A's Week 2** (SLM concepts, LoRA, PEFT, instruction/JSON data format,
TinyLlama inference) is a **hard prerequisite** for this week. Your SOAP generator *is* a
LoRA fine-tune of a small LM — that's literally the thing Dev A studied. It's folded into
Phase 1 below so you don't have to chase their notes.

You also **reuse Dev A's 50 doctor-patient dialogue examples** as seed training data.

You do **not** need Dev A's RAG work (their Week 3) — SOAP generation is
transcript → structured text, with no retrieval step.

---

# Phase 1 — Absorb the SLM stack (~2–3 days)

## What is an SLM, and how is it different from GPT-4?

A **Small Language Model** (1–3B parameters) is the same transformer architecture as a big
LLM, just far smaller — so it runs on a laptop CPU, can be fine-tuned cheaply, and keeps
everything on-device (the whole point of MedScript: privacy). GPT-4 is hundreds of billions
of params, cloud-only, general-purpose. For a **narrow, repeatable task** like
"transcript → SOAP note," a fine-tuned SLM is the right tool — it's specialized, free, and
private.

Candidates for this project:
- **TinyLlama-1.1B-Chat** — tiny, fast on CPU, chat-tuned. Best starting point on a Mac.
- **Phi-2 (2.7B)** — smarter, noticeably slower on CPU, bigger memory footprint.
- **DistilBERT** — *not* a generator (it's encoder-only / classification). Don't use it for
  SOAP text generation.

## Training vs fine-tuning vs inference

- **Training (from scratch):** build a model's knowledge from raw text. Millions of $, not us.
- **Fine-tuning:** take a pretrained model and nudge it toward *your* task with a small
  labeled dataset. This is what you do.
- **Inference:** running the finished model to get outputs. This is what the UI calls.

## LoRA — fine-tuning without a big GPU

Full fine-tuning updates *all* the model's weights — needs lots of GPU memory. **LoRA
(Low-Rank Adaptation)** freezes the original weights and trains tiny "adapter" matrices
injected into the attention layers. You end up training **<1% of the parameters**, so it
fits on modest hardware and the adapter file is just a few MB. You can fine-tune
TinyLlama with LoRA on Colab's free GPU (like you did for Whisper) or even slowly on the Mac.

`PEFT` (Parameter-Efficient Fine-Tuning) is the HuggingFace library that implements LoRA.

> 🚨 **This must be IMPLEMENTED, not just understood.** Week 3's SLM fine-tuning is required
> to use PEFT + LoRA properly — a real, runnable training run that produces a saved adapter,
> not a conceptual snippet. The conceptual lines below are the *idea*; the full runnable
> walkthrough lives in **Phase 2 → "PEFT + LoRA implementation"**. Do not stop at the concept.

```python
from peft import LoraConfig, get_peft_model
config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(base_model, config)   # only adapters are now trainable
```

## The data format you fine-tune on (instruction / JSON)

Generators are fine-tuned on **instruction-style** examples — typically JSONL, one example
per line:

```json
{"instruction": "Convert this doctor-patient transcript into a SOAP note.",
 "input": "Doctor: BP ketlu che? Patient: blood pressure 150/95 che. ...",
 "output": "Subjective: Patient reports ...\nObjective: BP 150/95 ...\nAssessment: ...\nPlan: ..."}
```

At training time these get rendered into the model's chat template. The model learns the
mapping `transcript → SOAP`.

### Phase 1 tasks
- [ ] Read a plain-language LoRA explainer + skim the PEFT quickstart
- [ ] `pip install peft` (and `accelerate` if not already there)
- [ ] Run **TinyLlama inference once** locally — feed it a prompt, see it generate text
- [ ] Write one hand-made `transcript → SOAP` JSONL example to internalize the format

---

# Phase 2 — Build the SOAP Note Generator

## Concept — what a SOAP note is

A clinical documentation standard. Four sections:

| Section | Contains | Example |
|---------|----------|---------|
| **S**ubjective | What the patient *reports* (symptoms, history, in their words) | "Patient reports headache and dizziness for 3 days." |
| **O**bjective | Measurable facts (vitals, exam, labs) | "BP 150/95, fasting sugar 280 mg/dL." |
| **A**ssessment | The clinician's diagnosis / interpretation | "Uncontrolled hypertension with type-2 diabetes." |
| **P**lan | Treatment, meds, follow-up | "Start Metformin 500mg BD; recheck BP in 2 weeks." |

Worked example end-to-end:

> **Transcript:** "Doctor: BP ketlu che? Patient: blood pressure 150/95 che. Doctor:
> diabetes che? Patient: ha, sugar level 280 che. Doctor: Metformin 500mg tablet lo."
>
> **SOAP:**
> - **S:** Patient reports high blood pressure and elevated blood sugar.
> - **O:** BP 150/95; sugar level 280.
> - **A:** Hypertension with uncontrolled diabetes.
> - **P:** Metformin 500mg prescribed.

## Architecture note (important)

You now have **two separate models**, deliberately decoupled:

```
Gujarati audio ──▶ [Whisper fine-tune]  ──▶ transcript (Gujarati+English text)
                                              │
                                              ▼
                                      (transcribe / translate step)
                                              │
                                              ▼
                          [SOAP SLM (TinyLlama+LoRA)] ──▶ English SOAP note
```

Keep them independent — different models, different training data. The "glue" is just text
passing from one to the other. This is what makes the Week 4 pipeline clean.

## Build tasks
- [ ] **Collect data:** 20–30 English doctor-patient transcripts from public datasets + reuse
      Dev A's 50 dialogues. Aim for variety (different conditions).
- [ ] **Label:** write the SOAP `output` for each transcript → build a `transcript→SOAP` JSONL
      (this is the slow, valuable part; even 50–80 good pairs go a long way with LoRA)
- [ ] **Fine-tune:** LoRA-tune TinyLlama (or Phi-2) on the JSONL via PEFT, on Colab GPU or Mac
- [ ] **Prompt template:** design a prompt that reliably forces all 4 sections, in order
- [ ] **Summary generator:** a second prompt that extracts key symptoms + diagnosis in 3–5 lines
- [ ] **Test:** feed a mock Gujarati-English transcript (translated to English) → verify SOAP
- [ ] **Evaluate:** confirm all 4 sections appear and are populated in *every* test case
- [ ] **Refine:** look at bad outputs, add corrective examples to the dataset, re-tune

## PEFT + LoRA implementation (the required, runnable path)

This is the part that must actually run. Copy-runnable end-to-end: load → `LoraConfig` →
`get_peft_model` → train → save adapter → reload for inference. Best run on a Colab free GPU
(like your Whisper fine-tune); it will also run slowly on the Mac CPU for tiny experiments.

```bash
pip install transformers peft trl datasets accelerate bitsandbytes
```

### 1. Load base model + tokenizer (optional 4-bit to save memory)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # or "microsoft/phi-2"

# 4-bit quant is OPTIONAL — it cuts GPU memory for QLoRA. On CPU-only, drop quantization_config.
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token        # TinyLlama has no pad token by default

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,   # remove this line for plain CPU LoRA (no bitsandbytes)
    device_map="auto",
)
```

### 2. Define the LoRA adapter — every field explained

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)   # only needed if using 4-bit above

lora = LoraConfig(
    r=16,                       # rank of the adapter matrices — capacity. 8–32 typical. Higher = more to learn, more memory.
    lora_alpha=32,              # scaling for the adapter (effective LR multiplier). Common: alpha = 2 × r.
    lora_dropout=0.05,          # regularization on the adapter
    bias="none",                # don't train bias terms
    task_type="CAUSAL_LM",      # text generation (NOT classification)
    target_modules=[            # WHICH layers get adapters — DIFFERS BY MODEL (see gotcha below)
        "q_proj", "k_proj", "v_proj", "o_proj",   # TinyLlama attention projections
    ],
)

model = get_peft_model(model, lora)
model.print_trainable_parameters()
# Expect something like: trainable params: 4.5M || all params: 1.1B || trainable%: 0.4
# ^ THIS printout is your proof LoRA is actually active and only ~<1% trains.
```

### 3. Format the data into the chat template

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="soap_pairs.jsonl", split="train")

def format_example(ex):
    messages = [
        {"role": "system", "content": "Convert the doctor-patient transcript into a SOAP note."},
        {"role": "user", "content": ex["input"]},
        {"role": "assistant", "content": ex["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

ds = ds.map(format_example)
```

### 4. Train with TRL's SFTTrainer (CPU/Colab-friendly settings)

```python
from trl import SFTTrainer, SFTConfig

args = SFTConfig(
    output_dir="models/soap-tinyllama-lora",
    per_device_train_batch_size=1,      # small — fits modest memory
    gradient_accumulation_steps=8,      # effective batch of 8 without the memory cost
    num_train_epochs=3,                 # small dataset → a few epochs is enough
    learning_rate=2e-4,                 # LoRA tolerates higher LR than full fine-tune
    logging_steps=10,
    save_strategy="epoch",
    dataset_text_field="text",
    max_seq_length=1024,
)

trainer = SFTTrainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

### 5. Save the adapter (a few MB, not the whole model)

```python
model.save_pretrained("models/soap-tinyllama-lora")      # saves ONLY the LoRA adapter
tokenizer.save_pretrained("models/soap-tinyllama-lora")
```

### 6. Inference — load base + adapter

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto")
model = PeftModel.from_pretrained(base, "models/soap-tinyllama-lora")
model.eval()

# Optional: bake the adapter into the base for a standalone model (no PEFT dependency at serve time)
# merged = model.merge_and_unload()
# merged.save_pretrained("models/soap-tinyllama-merged")
```

---

## Model choice on a Mac CPU

| | TinyLlama-1.1B | Phi-2 (2.7B) |
|--|----------------|--------------|
| CPU speed | Fast | Slow (~3× heavier) |
| Quality | Good enough for templated SOAP | Better reasoning |
| Memory | ~1–2 GB | ~3–5 GB |
| Recommendation | **Start here** | Try only if quality is short |

**Why LoRA over full fine-tune:** you don't have GPU memory for full fine-tuning of even a
1B model comfortably, the adapter is tiny and swappable, and you avoid "catastrophic
forgetting" of the base model's general language ability.

---

## Libraries / models
- `transformers` + `peft` + `trl` + `accelerate` — LoRA fine-tuning + inference
- `bitsandbytes` — optional 4-bit (QLoRA) to cut GPU memory; **omit on CPU-only**
- `datasets` — load your JSONL
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fallback: `microsoft/phi-2`)
- Optional later: `sentencepiece` is required by some tokenizers — install if prompted

---

## Gotchas
- **`target_modules` differ by model family — the #1 silent LoRA failure.** Wrong names → the
  adapter attaches to nothing and "training" changes nothing. Correct sets:
  - **TinyLlama / Llama-family:** `["q_proj", "k_proj", "v_proj", "o_proj"]` (add
    `"gate_proj", "up_proj", "down_proj"` to also adapt the MLP)
  - **Phi-2:** `["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]` (older checkpoints use
    fused `"Wqkv"` + `"out_proj"`)
  - When unsure, `print([n for n, _ in model.named_modules()])` and pick the real Linear names.
- **Always check `print_trainable_parameters()`** — if trainable% is ~0 or ~100, LoRA is
  misconfigured (wrong target_modules, or you forgot `get_peft_model`).
- **DistilBERT can't generate SOAP text** — it's encoder-only. Use a causal LM (TinyLlama/Phi-2).
- **Garbage in, garbage out:** the SOAP model is only as good as your labeled pairs. Spend
  your time on clean, consistent `output` formatting — the model copies your style exactly.
- **Pin the output format in the prompt** ("Respond with exactly four sections labeled
  Subjective/Objective/Assessment/Plan") and parse by those labels; add a fallback if parsing
  fails.
- **Keep SOAP in English** even from a code-switched transcript — flan-T5/TinyLlama are
  English-dominant and clinicians read English notes. Translate the Gujarati bits in the
  glue step, not inside the SLM.

---

## Done-when
- [ ] **PEFT + LoRA implemented properly:** `get_peft_model` used, and
      `print_trainable_parameters()` confirms only ~<1% of params train (LoRA is active)
- [ ] LoRA fine-tune runs end-to-end and produces a **saved adapter** (few MB) in
      `models/soap-tinyllama-lora/`
- [ ] Adapter **reloads** via `PeftModel.from_pretrained(base, adapter_dir)` for inference
- [ ] Given a transcript, the model emits all 4 SOAP sections reliably
- [ ] The 3–5 line summary works
- [ ] A mock Gujarati-English transcript yields a correct English SOAP note
- [ ] You've done at least one refine loop based on observed bad outputs
