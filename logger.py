"""
logger.py
---------
Saves every classification result to a CSV file (logs.csv).

Why log everything?
  1. You build a real dataset of prompts over time
  2. You can evaluate how your classifier performs
  3. You can spot patterns (e.g. most prompts are "instruction" type)
  4. For your IEEE submission — you have real data to show

CSV format (one row per prompt):
  timestamp, prompt, intent, confidence, route, latency_ms
"""

import pandas as pd
import os
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
LOG_FILE = "logs.csv"

# All column names — must stay consistent across every log entry
COLUMNS = [
    "timestamp",
    "prompt",
    "intent",
    "confidence",
    "route",
    "latency_ms"
]


def log_result(classification: dict, route: str) -> dict:
    """
    Saves one classification result as a new row in logs.csv.

    Parameters:
      classification : dict returned by classifier.classify_prompt()
      route          : string returned by router.get_route()

    Returns:
      The row that was saved (as a dict)
    """

    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt":      classification["prompt"],
        "intent":      classification["intent"],
        "confidence":  classification["confidence"],
        "route":       route,
        "latency_ms":  classification["latency_ms"]
    }

    df_new = pd.DataFrame([row])

    # Append mode ("a") adds to existing file.
    # header=not file_exists means we only write the column headers ONCE.
    file_exists = os.path.exists(LOG_FILE)
    df_new.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)

    return row


def load_logs() -> pd.DataFrame:
    """
    Loads and returns all logged data as a pandas DataFrame.
    Returns an empty DataFrame if no logs exist yet.
    """
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(LOG_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # Handle cases where the file exists but is empty or unreadable
        return pd.DataFrame(columns=COLUMNS)

    # Make sure all expected columns exist (safety check)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[COLUMNS]


def get_stats() -> dict:
    """
    Returns summary statistics computed from all logs.
    Used by the evaluator and the metrics panel in the UI.
    """
    df = load_logs()

    if df.empty:
        return {
            "total": 0,
            "avg_confidence": 0,
            "avg_latency_ms": 0,
            "intent_counts": {},
            "route_counts": {}
        }

    return {
        "total":           len(df),
        "avg_confidence":  round(df["confidence"].mean(), 4),
        "avg_latency_ms":  round(df["latency_ms"].mean(), 2),
        "intent_counts":   df["intent"].value_counts().to_dict(),
        "route_counts":    df["route"].value_counts().to_dict(),
        "high_confidence": int((df["confidence"] >= 0.75).sum()),
        "mid_confidence":  int(((df["confidence"] >= 0.55) & (df["confidence"] < 0.75)).sum()),
        "low_confidence":  int((df["confidence"] < 0.55).sum()),
    }


def clear_logs():
    """Deletes the log file. Use with caution — only for testing."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Deleted {LOG_FILE}")
    else:
        print("No log file to delete.")


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate logging a result
    fake_classification = {
        "prompt":     "What is edge computing?",
        "intent":     "question",
        "confidence": 0.9123,
        "all_scores": {"question": 0.9123, "instruction": 0.05},
        "latency_ms": 38.4
    }

    saved = log_result(fake_classification, "ODA")
    print("Saved row:", saved)

    df = load_logs()
    print(f"\nTotal rows in {LOG_FILE}: {len(df)}")
    print(df.tail(3).to_string(index=False))

    stats = get_stats()
    print("\nStats:", stats)
