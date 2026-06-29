# Week 2 (tail) — Closing the Whisper Fine-Tune

**Goal:** finish the last two open items on the Gujarati Whisper model so it's
production-ready for the SOAP pipeline: **(1) handle code-switching** (mixed Gujarati +
English), and **(2) optimize inference on CPU**.

**Dev A dependency:** none. This is pure Dev B audio work.

---

## Where you are

| Done | Open |
|------|------|
| FLEURS `gu_in` downloaded + cleaned → `data/gujarati_clean` | Code-switching (English terms come out as garbled Gujarati script) |
| Whisper-small fine-tuned (Colab T4, 500 steps) | CPU inference optimization (still FP32, slow) |
| WER measured: **48.4%** (vs 323.9% baseline) | |
| Model pushed: `ygotrijiya/whisper-small-gujarati-finetuned` | |
| Tested with your own voice (`test_my_voice.py`) | |

---

## Concept 1 — Why code-switching breaks

Real doctor speech in Gujarat sounds like:

> "દર્દીને **blood pressure** વધારે છે, **diabetes** પણ છે, **Metformin tablet** લખી આપો."

When you call Whisper with `language="gu"`, you force every output token through the
**Gujarati language-model head**. That head has essentially zero probability mass for
Latin-script tokens, so instead of writing `blood pressure` it picks the closest-sounding
Gujarati spelling — e.g. `બ્લડપ્રેશર` or worse, a garbled `બ્લર્ટ્રેસર`. The acoustic model
*heard* English fine; the language model *refused to write it in English*.

There are three levers, cheapest first:

### Lever A — `prompt_ids` priming (no retraining, do this first)
Whisper accepts a text prompt that's prepended to the decoder as "previous context." If that
context contains English medical terms, the decoder is primed to emit them in Latin script.

In **transformers 5.x** the API is *not* a raw string. You build token ids:

```python
MEDICAL_PROMPT = ("blood pressure diabetes tablet medicine injection fever "
                  "cholesterol sugar level ECG BP Metformin")
prompt_ids = processor.tokenizer.get_prompt_ids(MEDICAL_PROMPT, return_tensors="pt")
output = model.generate(input_features, prompt_ids=prompt_ids,
                        language="gujarati", task="transcribe")
```

### Lever B — post-processing term map (zero cost, always on)
Keep a dictionary of known bad transliterations → correct English, and run a regex pass
after decoding. Build it empirically: record yourself saying medical terms, see what comes
out, add the wrong ones to the map. Catches whatever slips past Lever A.

### Lever C — mixed-script fine-tune data (best, but slower; really a Week 3 thing)
The synthetic medical dialogues you'll build for the SOAP work naturally contain
code-switched sentences. Adding them to the fine-tune corpus teaches the model structurally
to emit English-in-English. Don't block Week 2 on this — Levers A+B get you a working demo
today.

### Tasks — code-switching
- [ ] Add a bilingual `MEDICAL_PROMPT` and pass `prompt_ids` in your inference path
- [ ] Build a `MEDICAL_TERM_MAP` (start ~15–20 entries) + a post-processing function
- [ ] Record 3–4 clips mixing English terms into Gujarati; confirm terms stay in English
- [ ] Note remaining failures to feed into Week 3's mixed-script dataset

---

## Concept 2 — Why quantize, and what it does

Your fine-tuned model is ~967 MB of **FP32** (32-bit float) weights. On a Mac CPU, every
matrix multiply moves and computes those 4-byte numbers — slow, and memory-heavy.

**Dynamic INT8 quantization** converts the weights of the `nn.Linear` layers (Whisper-small
has 193 of them — all the attention projections and feed-forward layers) from 32-bit floats
to **8-bit integers**, computed on the fly at inference. Result:

- **Size:** ~967 MB → ~250 MB in memory
- **Speed:** ~2–3× faster matmuls on CPU
- **Accuracy:** usually a tiny WER bump (often <1 absolute %), acceptable for this use case
- **What's untouched:** the encoder's conv layers and the decoder embeddings — only Linear
  ops are quantized, so the audio feature pipeline is unaffected

The simplest path needs **no new dependencies**:

```python
import torch, torch.nn as nn
q_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

> `torch.quantization.quantize_dynamic` prints a deprecation notice in torch 2.12 but works
> fine. The modern successor is `torchao` (separate install) — not worth it for this week.

### Tasks — CPU optimization
- [x] Quantize the fine-tuned model with `quantize_dynamic({nn.Linear}, qint8)` → `quantize_whisper.py`
- [x] Benchmark FP32 vs INT8 (3 runs each) → see finding below
- [x] Confirm transcript quality preserved (identical output FP32 vs INT8)
- [ ] Decide the *speed* path on Apple Silicon (quantization alone didn't deliver it — see below)

### ⚠️ Measured result on this Mac (Apple Silicon) — 2026-06

Ran `quantize_whisper.py` + a real-audio benchmark on `output.wav` (3.8s, 45 tokens):

| Metric | FP32 | INT8 dynamic | Verdict |
|--------|------|--------------|---------|
| Serialized size | 922 MB | 394 MB | ✅ **2.3× smaller** |
| Inference (real clip) | 1.66 s | 2.07 s | ❌ **~20% slower** |
| Transcript | `મારુ માથુ દુખે છે` | `મારુ માથુ દુખે છે` | ✅ identical |

**Why INT8 is slower here:** Apple Silicon runs FP32 matmuls on the AMX/Accelerate units,
which are extremely fast. The INT8 `qnnpack` kernels don't use AMX, so dynamic quantization
*loses* the speed race on this chip. On x86 servers INT8 is usually 2–3× faster (the original
assumption) — it's hardware-dependent. **Takeaway: dynamic quantization is a memory win on
Mac, not a speed win.**

**For actual speed on a Mac CPU, use one of:**
- **`faster-whisper` (CTranslate2)** — the standard fast-CPU Whisper runtime; its INT8 is
  genuinely faster on ARM. Requires converting the fine-tuned model to CT2 format.
- **Apple GPU via `device="mps"`** — still fully on-device (privacy preserved; "no cloud" is
  the real constraint, not "no GPU"). Usually the fastest option with the least code change.
- **Keep FP32 + the memory win** — if 1.6s for ~4s of audio is acceptable for the demo.

---

## Libraries / models

- `torch` (already installed) — quantization, no extra deps
- `transformers` 5.x — `WhisperForConditionalGeneration`, `WhisperProcessor`, `get_prompt_ids`
- Model: `ygotrijiya/whisper-small-gujarati-finetuned` (pulled from HF Hub cache)

---

## Gotchas (this stack specifically)

- **prompt_ids is a tensor, not a string** in transformers 5.x. Passing a string silently
  does the wrong thing. Use `tokenizer.get_prompt_ids(...)`.
- **Quantized models save as a torch state dict (`.pt`), not safetensors.** To reload: build
  the FP32 architecture from the Hub, then `model.load_state_dict(torch.load(path))`. You
  can't `from_pretrained` a dynamically-quantized model directly.
- **Keep inference on CPU** even though your Mac has MPS — the deployment target is CPU and
  quantization gains are a CPU story.
- **Don't over-prompt.** A huge `MEDICAL_PROMPT` eats the decoder context window and can hurt
  general transcription. Keep it to the terms that actually break.

---

## Done-when

- [ ] English medical terms survive in Latin script on your test recordings
- [ ] Quantized model is ~250 MB and measurably faster than FP32 (number written down)
- [ ] WER on the test slice is within ~1–2% of the FP32 model
- [ ] You have a documented load-quantized-model snippet ready for Week 4's UI
