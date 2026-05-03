# Edge AI Intent Classifier
### IEEE CS Internship — Phase 1

A text classification system that detects the **intent** of any user prompt
and routes it to the appropriate compute environment:
- 🟢 **ODA** — On-Device AI (fast, private, offline-capable)
- 🟡 **Hybrid** — Edge + Cloud split
- 🔴 **Cloud LLM** — Full cloud model

---

## Project Structure

```
ieee_classifier/
├── classifier.py    # NLP model — intent detection + confidence score
├── router.py        # Routing logic — ODA / Hybrid / Cloud LLM
├── logger.py        # CSV logging — saves every result to logs.csv
├── evaluator.py     # Metrics — confidence, latency, distribution stats
├── app.py           # Gradio UI — real-time testing dashboard
├── requirements.txt # Python dependencies
├── logs.csv         # Auto-created when you run the app
```

---

## Setup

```bash
# 1. Create environment
conda create -n ieee_env python=3.10
conda activate ieee_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open http://localhost:7860 in your browser.

---

## Model Used
`typeform/distilbert-base-uncased-mnli` — zero-shot NLI classifier (~268MB)
Downloads automatically on first run and is cached locally.

---

## How Routing Works

| Condition | Route |
|---|---|
| Confidence < 55% | Cloud LLM (uncertain, need stronger model) |
| Multi-step task | Hybrid (sub-tasks need both resources) |
| Confidence ≥ 75% + simple intent | ODA (safe for edge) |
| Confidence ≥ 75% + complex intent | Cloud LLM |
| Everything else | Hybrid |
