# Week 4 — Desktop UI + Integration

**Goal:** a clean Python desktop app a doctor can actually use — press START, speak, press
STOP, read the transcript / summary / SOAP note, export it. Plus: wire your UI into the
end-to-end pipeline.

---

## ⚠️ Dev A dependency callout

Dev A's Week 4 is the **end-to-end pipeline class** (audio → Whisper → RAG → SLM → SOAP).
**They build it; your UI consumes it.** You only need to agree the **interface contract** —
the function(s) your UI calls and what they return. You do *not* need to build the pipeline
internals, and **RAG stays out of the SOAP path** (it's optional, for a future Q&A feature).

Suggested contract to agree with Dev A:

```python
class MedScriptPipeline:
    def start_recording(self) -> None: ...
    def stop_recording(self) -> str: ...            # returns full transcript
    def on_transcript_chunk(self, callback) -> None # streams partial text to the UI
    def generate_outputs(self, transcript: str) -> dict:
        # -> {"transcript": str, "summary": str, "soap": {"S":..,"O":..,"A":..,"P":..}}
        ...
```

If Dev A's pipeline isn't ready, **stub these methods** so you can build and test the UI
independently, then swap the stub for the real pipeline later.

---

## Concept — Tkinter vs PyQt5

| | Tkinter | PyQt5 |
|--|---------|-------|
| Ships with Python | Yes (zero install) | No (`pip install pyqt5`) |
| Looks | Basic, dated | Polished, native-ish |
| Learning curve | Gentle | Steeper |
| Good for | A working demo fast | A product that looks professional |

**Recommendation:** if the priority is a demo that *works*, start with **Tkinter**. If the
presentation grades on polish (it does — that's your Week 5), **PyQt5** pays off. Pick one
and commit; don't build twice.

## Concept — threading (the trap that freezes every audio UI)

Recording and transcription are **long-running**. If you run them on the UI's main thread,
the window freezes — buttons stop responding, the timer stalls. You must run
recording/transcription on a **background thread** and push results back to the UI thread:

- **Tkinter:** background thread + `root.after(...)` (or a `queue.Queue`) to update widgets.
- **PyQt5:** `QThread` + signals/slots to deliver chunks to the main thread.

Never touch widgets directly from the worker thread.

---

## Build tasks
- [ ] Main window with a large **START (green)** and **STOP (red)** button
- [ ] **Live transcript viewer** — scrollable text box that appends chunks as the doctor speaks
- [ ] **Status bar** — `Recording... 00:04:32 elapsed` while active; idle/processing states too
- [ ] **Output panel with 3 tabs** — `Transcript | Summary | SOAP Note`
- [ ] **Doctor name + Patient name** input fields at the top
- [ ] **Export button** — save output as `.txt`, plus copy-to-clipboard
- [ ] Run recording/transcription on a **background thread**; update widgets safely
- [ ] Wire to the pipeline contract (or a stub); fill the 3 tabs from `generate_outputs(...)`
- [ ] **Polish:** readable fonts, clean spacing, verify it fits a 1080p laptop screen

## Awareness (Dev A owns these, but your UI should not break on them)
- **Long sessions:** Dev A chunks audio into ~30s segments for 2–60 min recordings. Your
  transcript box should keep appending gracefully over a long session.
- **Errors:** silence detection, mic failure, out-of-memory. Show a friendly status message
  instead of crashing.

---

## Libraries
- **Tkinter** (stdlib) *or* **PyQt5** (`pip install pyqt5`)
- `threading` / `queue` (stdlib) for the worker thread
- Reuse your own `recorder.py` for mic capture and the quantized Whisper from Week 2
- `pyperclip` (optional) for copy-to-clipboard, or use the toolkit's native clipboard

---

## Gotchas
- **UI freeze = you're on the main thread.** This is the #1 bug. Thread the audio work.
- **Reload the quantized model once at startup,** not per request — loading is slow.
- **Decouple UI from models:** the UI calls the pipeline contract; it shouldn't import torch
  directly. Makes the stub swap trivial and keeps the UI testable.
- **Timer accuracy:** drive the elapsed clock off real timestamps, not a counter incremented
  in a loop (loops drift).

---

## Done-when
- [ ] START/STOP works; the window never freezes during a recording
- [ ] Transcript streams live; all 3 tabs populate after STOP
- [ ] Name fields + export-to-txt + copy-to-clipboard all work
- [ ] Runs end-to-end against the real pipeline (or a clearly-marked stub)
- [ ] Looks clean on a 1080p screen — ready to demo
