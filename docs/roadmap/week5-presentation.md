# Week 5 — Product & Business Presentation (your half)

**Goal:** present MedScript convincingly to non-technical stakeholders (think CEO). Dev A
takes the technical/architecture half; **you take the product, business, and privacy story**,
and you drive the live demo.

---

## ⚠️ Dev A dependency callout

You only need to **borrow two small things** from Dev A's technical half:
1. **One paragraph of RAG awareness** — enough to answer "why didn't the SOAP note use RAG?"
   (Answer: SOAP is structured extraction from a single transcript, not a knowledge-base
   lookup; RAG is reserved for a future medical-Q&A feature.)
2. **The benchmark numbers** Dev A measures (accuracy %, seconds per minute of audio, model
   size in MB) — cite these in your slides.

Everything else this week is yours.

---

## The story arc (what makes this land with a CEO)

Lead with the problem, not the tech: *doctors spend hours writing notes; MedScript listens to
the consultation and writes the SOAP note for them — in Gujarati + English, entirely on the
doctor's own laptop, with no patient audio ever leaving the device.* Privacy + local-first is
your headline differentiator. Hammer it.

---

## Tasks
- [ ] **Polish the UI** — buttons, font sizes, tab labels; make it look professional (this *is*
      the product in the demo)
- [ ] **Business use-case slide** — who else benefits: clinics, telemedicine, legal
      depositions, field interviews in regional languages, insurance claims
- [ ] **"What we learned"** — 5 key AI takeaways in plain, jargon-free English
- [ ] **Future roadmap** — mobile app, cloud (opt-in) version, more Indian languages, EHR
      integration, speaker diarization improvements
- [ ] **Data-privacy slide** (your strongest card) — how audio is captured, processed, and
      discarded locally; nothing uploaded; on-device models
- [ ] **Demo script** — write out exactly who says what, in what order, with the click path
- [ ] **Rehearse the full demo 3×** before demo day; fix anything that breaks
- [ ] Coordinate with Dev A on slide order and handoffs

## Q&A prep (have crisp answers ready)
- **"Can this run offline?"** → Yes, fully. That's the core design — models run on the laptop.
- **"How accurate is it?"** → Cite WER (Whisper) + SOAP-section completeness; be honest about
  current limits and the path to improve (more fine-tune data).
- **"How do you improve it?"** → More labeled Gujarati medical audio + more transcript→SOAP
  pairs; bigger SLM when hardware allows.
- **"Is patient data safe?"** → Nothing leaves the device; point at the privacy slide.

---

## Demo-day risk management
- [ ] **Record a 2-min backup demo video** in case the live mic/laptop misbehaves (this is
      Dev A's listed task, but make sure it exists — it saves *your* live demo)
- [ ] Pre-load all models before the demo starts (cold-loading on stage = dead air)
- [ ] Have a known-good sample recording ready if the room is too noisy for live capture
- [ ] Test the exact demo laptop + mic in the actual room if you can

---

## Done-when
- [ ] UI looks presentable; demo runs start-to-finish without a hitch
- [ ] All your slides drafted: business use-case, learnings, roadmap, privacy
- [ ] Demo script written and rehearsed 3×
- [ ] Backup video recorded; models pre-load cleanly
- [ ] You can answer the four Q&A questions without hesitating
