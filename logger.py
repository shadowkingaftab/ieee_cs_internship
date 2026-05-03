"""
logger.py
---------
Saves every classification result to a CSV file (logs.csv).
Includes full intent matrices for analysis.
"""

import pandas as pd
import os
import json
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
LOG_FILE = "logs.csv"

# All column names — must stay consistent across every log entry
COLUMNS = [
    "timestamp",
    "prompt",
    "intent",
    "confidence",
    "intent_matrix",
    "route",
    "latency_ms"
]

def log_result(classification: dict, route: str) -> dict:
    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt":      classification.get("prompt", ""),
        "intent":      classification.get("top_intent", classification.get("intent", "")),
        "confidence":  classification.get("top_confidence", classification.get("confidence", 0)),
        "intent_matrix": json.dumps(classification.get("intent_matrix", classification.get("all_scores", {}))),
        "route":       route,
        "latency_ms":  classification.get("latency_ms", 0)
    }

    df_new = pd.DataFrame([row])
    file_exists = os.path.exists(LOG_FILE)
    df_new.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)
    return row

def load_logs() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(LOG_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[COLUMNS]

def get_stats() -> dict:
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
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Deleted {LOG_FILE}")
    else:
        print("No log file to delete.")

if __name__ == "__main__":
    fake_classification = {
        "prompt":     "What is edge computing?",
        "top_intent": "simple_query",
        "top_confidence": 0.9123,
        "intent_matrix": {"simple_query": 0.9123, "instruction": 0.05},
        "latency_ms": 38.4
    }
    saved = log_result(fake_classification, "ODA")
    print("Saved row:", saved)
