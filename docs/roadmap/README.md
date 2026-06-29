# MedScript — Dev B Roadmap

These are **your** (Dev B) week-by-week study + build guides for MedScript: a fully-local
Gujarati doctor-patient ASR + SOAP-note tool. They cover only your track, plus the Dev A
pieces that are genuine prerequisites for your work.

> **MedScript in one line:** record a Gujarati/English doctor-patient conversation →
> transcribe it locally with a fine-tuned Whisper → generate a structured SOAP note with a
> local small language model → show it in a desktop app. No cloud, no patient audio ever
> leaves the machine.

## The files

| Week | File | Focus | State |
|------|------|-------|-------|
| 2 (tail) | [week2-finish.md](week2-finish.md) | Close the Whisper fine-tune: code-switching + CPU speedup | ~85% done |
| 3 | [week3-soap.md](week3-soap.md) | **SOAP Note Generator** — learn SLM/LoRA, then build it | next |
| 4 | [week4-ui.md](week4-ui.md) | Desktop UI (Tkinter/PyQt) + integration with the pipeline | upcoming |
| 5 | [week5-presentation.md](week5-presentation.md) | Product & business presentation (your half) | upcoming |

## What's already true (don't redo it)

- **Whisper fine-tuned for Gujarati:** `ygotrijiya/whisper-small-gujarati-finetuned` on the
  HuggingFace Hub. WER **48.4%** (vs **323.9%** baseline that hallucinated Hindi).
- **Dataset:** Google FLEURS `gu_in`, cleaned, saved locally at `data/gujarati_clean`
  (~3,145 train / 432 val / 1,000 test) in HuggingFace Arrow format.
- **Working scripts:** `recorder.py`, `transcribe.py`, `test_my_voice.py`, `finetune.py`,
  `prepare_data.py`, `noice_test.py`.
- **Environment:** Python 3.14, `.venv`, Apple Silicon Mac (CPU-only for privacy — MPS is
  available but we keep inference on CPU to mirror the deployment target).

## The Dev A → Dev B dependency map (what you actually need from the other track)

You are Dev B. You do **not** need to do Dev A's whole roadmap. You only need the pieces
that your own deliverables depend on:

| Dev A item | Dev A week | You need it? | Why / where |
|------------|-----------|--------------|-------------|
| ML / NLP / transformer theory, tokenization, embeddings | 1 | **Light skim** | Background for reasoning about Whisper + the SLM. ~1–2 hrs reading. |
| **SLM concepts, LoRA, PEFT, instruction/JSON data format, TinyLlama inference** | 2 | **YES — fully** | Your SOAP generator *is* a LoRA fine-tune of a small LM. Folded into [week3-soap.md](week3-soap.md) Phase 1. |
| The 50 doctor-patient dialogue examples Dev A prepared | 2 | **YES — reuse** | Seed data for your SOAP fine-tune. |
| RAG / ChromaDB / FAISS / sentence-embeddings build | 3 | **NO — skip the build** | SOAP = transcript → structured text. No retrieval needed. Only need *conceptual awareness* for the Week 5 Q&A. |
| End-to-end pipeline class | 4 | **Interface only** | Dev A builds it; your UI plugs in. You agree the function contract, not the internals. See [week4-ui.md](week4-ui.md). |

**One-sentence takeaway:** the only Dev A work you must truly absorb is the **SLM/LoRA stack
from their Week 2** — everything else is skim, reuse, or coordinate.

## How to read a weekly file

Each file is structured the same way:

1. **Dependency callout** — which Dev A items (if any) are folded in this week and why.
2. **Concepts** — the "why," in plain language, with worked examples.
3. **Tasks** — a checkbox list of the "what," each with a short rationale.
4. **Libraries / models** — concrete, CPU-friendly recommendations.
5. **Gotchas** — the traps specific to this project's stack.
6. **Done-when** — the checklist that tells you the week is closed.

These are reference/study docs. They describe *what to build and why*, not finished code.
