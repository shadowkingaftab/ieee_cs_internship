"""
classifier.py
-------------
The NLP brain of the system.
Loads a pretrained zero-shot classification model and uses it
to detect the INTENT of any text prompt, along with a confidence score.

Zero-shot means: we never trained this model on our data.
We just give it our labels and it figures out the best match.
Model used: typeform/distilbert-base-uncased-mnli (~268MB, fast)
"""

from transformers import pipeline
import time
import sys

# ── Load model once at startup ──────────────────────────────────────────────
# This line downloads the model the FIRST time (takes ~1-2 min).
# After that it's cached locally and loads in ~3 seconds.
print("Loading classifier model... (first run downloads ~268MB)", file=sys.stderr)

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

print("Model loaded successfully!", file=sys.stderr)

# ── Define your intent labels ────────────────────────────────────────────────
# These are the 5 categories your system understands.
# The model will score the prompt against ALL of these and pick the best match.
INTENT_LABELS = [
    "question",           # User is asking something  → e.g. "What is edge AI?"
    "instruction",        # User wants something done  → e.g. "Translate this text"
    "multi-step task",    # Multiple actions needed    → e.g. "Summarize and send"
    "analysis",           # Needs deep reasoning       → e.g. "Analyze this dataset"
    "creative request"    # Open-ended generation      → e.g. "Write a poem about..."
]


def classify_prompt(prompt: str) -> dict:
    """
    Takes a raw text prompt.
    Returns a dictionary with:
      - prompt        : the original text
      - intent        : the top detected label
      - confidence    : how sure the model is (0.0 to 1.0)
      - all_scores    : scores for every label (so you can see full picture)
      - latency_ms    : how long classification took in milliseconds
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Start the timer — we measure how long classification takes
    start_time = time.time()

    # ── The actual classification ────────────────────────────────────────────
    # The model reads the prompt and scores it against each label.
    # It returns labels sorted from most likely to least likely.
    result = classifier(prompt.strip(), INTENT_LABELS)

    # Stop the timer
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # ── Package everything into a clean dictionary ───────────────────────────
    return {
        "prompt":     prompt.strip(),
        "intent":     result["labels"][0],                   # top label
        "confidence": round(result["scores"][0], 4),         # top score
        "all_scores": {                                       # all label scores
            label: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        },
        "latency_ms": latency_ms
    }


# ── Quick self-test (run this file directly to verify it works) ──────────────
if __name__ == "__main__":
    test_prompts = [
        "What is the difference between edge AI and cloud AI?",
        "Translate this paragraph to Kannada",
        "Analyze the sensor data and generate a weekly report",
        "Summarize this PDF and then email it to my team",
        "Write a poem about distributed computing"
    ]

    print("\n=== CLASSIFIER SELF-TEST ===\n")
    for p in test_prompts:
        out = classify_prompt(p)
        print(f"Prompt     : {out['prompt']}")
        print(f"Intent     : {out['intent']}")
        print(f"Confidence : {out['confidence'] * 100:.1f}%")
        print(f"Latency    : {out['latency_ms']} ms")
        print("-" * 50)
