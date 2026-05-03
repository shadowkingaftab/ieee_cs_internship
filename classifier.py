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
INTENT_LABELS = [
    "summarize", "translate", "analyze_data", "generate_code",
    "multi_step", "ambiguous", "urgent", "simple_query"
]

def classify_prompt(prompt: str) -> dict:
    """
    Takes a raw text prompt.
    Returns a dictionary with:
      - prompt        : the original text
      - top_intent    : the top detected label
      - top_confidence: how sure the model is (0.0 to 1.0)
      - intent_matrix : normalized probabilities for all labels
      - is_multi_step : boolean if multi_step confidence is high
      - is_ambiguous  : boolean if ambiguous confidence is high
      - latency_ms    : how long classification took in milliseconds
      - intent        : (backwards compatibility) same as top_intent
      - confidence    : (backwards compatibility) same as top_confidence
      - all_scores    : (backwards compatibility) same as intent_matrix
    """

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Start the timer — we measure how long classification takes
    start_time = time.time()

    # ── The actual classification ────────────────────────────────────────────
    # The model reads the prompt and scores it against each label.
    result = classifier(prompt.strip(), INTENT_LABELS)
    
    intent_probs = {label: score for label, score in zip(result["labels"], result["scores"])}

    # Add domain-specific logic (e.g., detect multi-step tasks)
    text_lower = prompt.lower()
    if " and " in text_lower or " then " in text_lower:
        intent_probs["multi_step"] *= 1.5  # Boost multi-step confidence
        
    if "urgent" in text_lower or "asap" in text_lower:
        intent_probs["urgent"] *= 1.5

    # Normalize probabilities
    total = sum(intent_probs.values())
    intent_probs = {k: round(v/total, 4) for k, v in intent_probs.items()}

    # Stop the timer
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Get top intent after post-processing
    top_intent, top_confidence = max(intent_probs.items(), key=lambda x: x[1])

    # ── Package everything into a clean dictionary ───────────────────────────
    return {
        "prompt":         prompt.strip(),
        "top_intent":     top_intent,
        "top_confidence": top_confidence,
        "intent_matrix":  intent_probs,
        "is_multi_step":  intent_probs.get("multi_step", 0) > 0.3,
        "is_ambiguous":   intent_probs.get("ambiguous", 0) > 0.2,
        "latency_ms":     latency_ms,
        
        # Backwards compatibility for existing code (like mcp_server.py)
        "intent":         top_intent,
        "confidence":     top_confidence,
        "all_scores":     intent_probs
    }


# ── Quick self-test (run this file directly to verify it works) ──────────────
if __name__ == "__main__":
    test_prompts = [
        "What is the difference between edge AI and cloud AI?",
        "Translate this paragraph to Kannada",
        "Analyze the sensor data and generate a weekly report",
        "Summarize this PDF and then email it to my team ASAP",
        "Write a poem about distributed computing"
    ]

    print("\n=== CLASSIFIER SELF-TEST ===\n")
    for p in test_prompts:
        out = classify_prompt(p)
        print(f"Prompt        : {out['prompt']}")
        print(f"Top Intent    : {out['top_intent']}")
        print(f"Confidence    : {out['top_confidence'] * 100:.1f}%")
        print(f"Is Multi-Step?: {out['is_multi_step']}")
        print(f"Is Urgent?    : {out['intent_matrix'].get('urgent', 0) > 0.3}")
        print(f"Latency       : {out['latency_ms']} ms")
        print("-" * 50)
