"""
train_classifier.py
-------------------
Fine-tunes the intent classifier on your domain-specific dataset to achieve
>95% confidence consistently on Edge AI prompts.

Strategy:
  - Base model: typeform/distilbert-base-uncased-mnli (same as production)
  - Method: SetFit (few-shot) OR full fine-tune with HuggingFace Trainer
  - Dataset: training/domain_dataset.json (add your own examples!)
  - Output: models/fine_tuned_classifier/  (drop-in replacement)

Usage:
  python training/train_classifier.py
  python training/train_classifier.py --epochs 5 --lr 2e-5 --eval

Requirements:
  pip install transformers datasets scikit-learn torch accelerate
"""

import json
import argparse
import os
import sys
import time

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fine-tune intent classifier")
parser.add_argument("--epochs",    type=int,   default=4,     help="Training epochs")
parser.add_argument("--lr",        type=float, default=2e-5,  help="Learning rate")
parser.add_argument("--batch",     type=int,   default=8,     help="Batch size")
parser.add_argument("--eval",      action="store_true",       help="Run evaluation after training")
parser.add_argument("--data",      type=str,   default=None,  help="Path to dataset JSON")
parser.add_argument("--output",    type=str,   default=None,  help="Output model directory")
parser.add_argument("--test-only", action="store_true",       help="Only run eval on existing model")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = args.data   or os.path.join(ROOT_DIR, "training", "domain_dataset.json")
OUTPUT_DIR  = args.output or os.path.join(ROOT_DIR, "models", "fine_tuned_classifier")
BASE_MODEL  = "typeform/distilbert-base-uncased-mnli"

INTENT_LABELS = [
    "summarize", "translate", "analyze_data", "generate_code",
    "multi_step", "ambiguous", "urgent", "simple_query"
]
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(INTENT_LABELS)}


def load_dataset():
    """Load and split domain_dataset.json into train/eval sets."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate
    valid = [d for d in data if d.get("label") in LABEL2ID]
    invalid = len(data) - len(valid)
    if invalid:
        print(f"[warn] Skipped {invalid} samples with unknown labels.")

    # Stratified split: 80% train, 20% eval
    from collections import defaultdict
    import random
    random.seed(42)

    by_label = defaultdict(list)
    for d in valid:
        by_label[d["label"]].append(d)

    train_data, eval_data = [], []
    for label, samples in by_label.items():
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.8))
        train_data.extend(samples[:split])
        eval_data.extend(samples[split:])

    random.shuffle(train_data)
    print(f"Dataset: {len(train_data)} train | {len(eval_data)} eval samples")
    print(f"Labels : {sorted(by_label.keys())}")
    return train_data, eval_data


def prepare_hf_dataset(train_data, eval_data):
    """Convert list of dicts to HuggingFace Dataset objects."""
    from datasets import Dataset

    def to_hf(data):
        return Dataset.from_dict({
            "text":  [d["text"]  for d in data],
            "label": [LABEL2ID[d["label"]] for d in data],
        })

    return to_hf(train_data), to_hf(eval_data)


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred):
    """Accuracy + per-class confidence (for >95% target tracking)."""
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np

    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=-1)
    top_confs = probs[np.arange(len(preds)), preds]

    acc = accuracy_score(labels, preds)
    avg_conf = float(top_confs.mean())
    high_conf = float((top_confs >= 0.95).mean())

    print(f"\n  Accuracy      : {acc*100:.1f}%")
    print(f"  Avg Confidence: {avg_conf*100:.1f}%")
    print(f"  >95% Confident: {high_conf*100:.1f}% of samples")
    print("\n" + classification_report(
        labels, preds, target_names=INTENT_LABELS, zero_division=0
    ))
    return {"accuracy": acc, "avg_confidence": avg_conf, "pct_above_95": high_conf}


def softmax(x, axis=-1):
    import numpy as np
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def train():
    print("=" * 60)
    print("  Edge AI Classifier — Custom Fine-Tuning")
    print("=" * 60)
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Data file  : {DATA_PATH}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Epochs     : {args.epochs}  LR: {args.lr}  Batch: {args.batch}")
    print("=" * 60)

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
            EarlyStoppingCallback,
        )
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install transformers torch accelerate datasets scikit-learn")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    train_data, eval_data = load_dataset()
    train_ds, eval_ds = prepare_hf_dataset(train_data, eval_data)

    # ── Tokenizer + Model ──────────────────────────────────────────────────────
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # NLI head → classification head
    )

    # ── Tokenize ───────────────────────────────────────────────────────────────
    train_ds = train_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    eval_ds  = eval_ds.map(lambda b: tokenize(b, tokenizer), batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training args ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch,
        per_device_eval_batch_size  = args.batch,
        learning_rate               = args.lr,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = 10,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        tokenizer       = tokenizer,
        data_collator   = collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    t0 = time.time()
    trainer.train()
    elapsed = round(time.time() - t0, 1)
    print(f"\nTraining complete in {elapsed}s")

    # ── Save ───────────────────────────────────────────────────────────────────
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

    # ── Save label map ─────────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "intent_labels.json"), "w") as f:
        json.dump(INTENT_LABELS, f, indent=2)

    # ── Final eval ─────────────────────────────────────────────────────────────
    if args.eval:
        print("\nRunning final evaluation...")
        results = trainer.evaluate()
        acc  = results.get("eval_accuracy", 0)
        conf = results.get("eval_avg_confidence", 0)
        pct  = results.get("eval_pct_above_95", 0)
        print(f"\n{'='*40}")
        print(f"  FINAL RESULTS")
        print(f"  Accuracy        : {acc*100:.1f}%")
        print(f"  Avg Confidence  : {conf*100:.1f}%")
        print(f"  Samples >95%    : {pct*100:.1f}%")
        if pct >= 0.90:
            print("  TARGET MET: >95% confidence on 90%+ samples!")
        else:
            print("  TIP: Add more domain examples to domain_dataset.json")
            print("  TIP: Try --epochs 8 --lr 1e-5 for deeper training")
        print(f"{'='*40}\n")

    print("\nTo use the fine-tuned model in production:")
    print(f"  Set CLASSIFIER_MODEL_PATH={OUTPUT_DIR} in your .env")
    print("  Then update classifier.py to load from that path.\n")


def test_existing():
    """Evaluate already-trained model without re-training."""
    if not os.path.exists(OUTPUT_DIR):
        print(f"No fine-tuned model found at {OUTPUT_DIR}")
        print("Run without --test-only to train first.")
        sys.exit(1)

    from transformers import pipeline
    clf = pipeline("text-classification", model=OUTPUT_DIR, top_k=None)
    _, eval_data = load_dataset()

    correct, total, high_conf = 0, 0, 0
    for sample in eval_data:
        results = clf(sample["text"])[0]
        top = max(results, key=lambda x: x["score"])
        pred_label = top["label"].lower()
        conf = top["score"]
        if pred_label == sample["label"]:
            correct += 1
        if conf >= 0.95:
            high_conf += 1
        total += 1

    print(f"Accuracy : {correct/total*100:.1f}%  ({correct}/{total})")
    print(f">95% conf: {high_conf/total*100:.1f}%  ({high_conf}/{total})")


if __name__ == "__main__":
    if args.test_only:
        test_existing()
    else:
        train()
