# Week 2 Results — Gujarati Whisper Fine-tuning

## Model
- Base: openai/whisper-small
- Fine-tuned: ygotrijiya/whisper-small-gujarati-medscript
- Dataset: google/fleurs gu_in (2807 train, 393 val)
- Training: 500 steps, T4 GPU, ~77 minutes

## Results
| Step | Loss  | WER    |
|------|-------|--------|
| 100  | 0.455 | 82.3%  |
| 200  | 0.185 | 70.1%  |
| 300  | 0.152 | 64.6%  |
| 400  | 0.093 | 62.2%  |
| 500  | 0.087 | 61.4%  |

## Speed Benchmarks (10s audio, M1 Pro)
| Mode         | Time  | RTF   | Speedup |
|--------------|-------|-------|---------|
| fp32 CPU     | 9.17s | 0.92x | baseline|
| torch.compile| 7.87s | 0.79x | 1.16x   |
| fp16 MPS     | 3.00s | 0.30x | 3.05x   |

## Code-switching
Handles Gujarati+English mix correctly.
English medical terms phonetically written in Gujarati script.
fever→ફિવર, BP→બીપી, ECG→એસીજી, normal→નોર્મલ

## Key learnings
- Whisper is an encoder-decoder transformer
- Fine-tuning adjusts weights using LoRA-style targeted updates
- WER = word error rate, lower is better
- fp16 = 3x faster but slightly less accurate than fp32
- Code-switching works out of the box with Whisper