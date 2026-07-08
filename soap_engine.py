import requests

MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SOAP_PROMPT = """You are a medical scribe. Convert this doctor-patient dialogue into a SOAP note.

Rules:
- Output ONLY the SOAP note, nothing else
- Use exactly this structure: S:, O:, A:, P:
- S (Subjective): symptoms and history the PATIENT reports, including denied symptoms
- O (Objective): ONLY measured findings — vitals, exam results, test results
- A (Assessment): brief clinical impression based ONLY on stated findings, worded as preliminary
- P (Plan): tests ordered, medications, follow-up as stated
- Do NOT invent any findings, test results, or diagnoses
{context_block}
Dialogue:
{dialogue}

SOAP note:"""

def _call(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.1}
    })
    return r.json()["response"].strip()

def generate_soap(dialogue, context=None):
    """context: optional list of strings from Dev A's RAG retrieval."""
    if context:
        context_block = (
            "\nRelevant medical reference (use ONLY to clarify terminology, "
            "do not add facts from it to the note):\n"
            + "\n".join(f"- {c}" for c in context) + "\n"
        )
    else:
        context_block = ""
    return _call(SOAP_PROMPT.format(context_block=context_block, dialogue=dialogue))

def generate_summary(dialogue):
    prompt = f"""From this doctor-patient dialogue, write a summary.

Rules:
- Output exactly 5 plain sentences, one per line
- No labels, no numbering, no "Line 1:" prefixes
- Cover: main complaint & duration / key symptoms incl. denied / findings & vitals / impression / plan & follow-up
- Only use information stated in the dialogue

Dialogue:
{dialogue}

Summary:"""
    return _call(prompt)

if __name__ == "__main__":
    test = """Doctor: Yes tell me what is problem.
Patient: Sir from 3 days I am having fever and full body pain. Head is also paining too much.
Doctor: Vomiting? Loose motion?
Patient: No vomiting. But not feeling like eating anything.
Doctor: Ok. Temperature is 101. BP 118/76. Throat little red.
Doctor: Take paracetamol 500 three times a day for 5 days. Drink more water. If fever not going in 3 days come back, we will do blood test."""

    # Test without RAG context
    print(generate_soap(test))
    print("\n--- summary ---")
    print(generate_summary(test))

    # Test WITH mock RAG context (simulating Dev A's output)
    mock_context = [
        "Paracetamol: analgesic/antipyretic. Adult dose 500-1000mg every 6-8 hours, max 4g/day.",
        "Viral fever: self-limiting febrile illness, typically 3-7 days, symptomatic treatment."
    ]
    print("\n--- with RAG context ---")
    print(generate_soap(test, context=mock_context))