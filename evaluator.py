"""
evaluator.py
------------
Reads logs.csv and computes evaluation metrics.

What metrics do we track?
  - Total prompts processed
  - Average confidence score
  - Average latency
  - Intent distribution (which intents appear most)
  - Route distribution (how often each route is chosen)
  - Confidence breakdown (high/medium/low)

For advanced evaluation (accuracy, F1 score):
  You would need to manually add a "true_label" column to logs.csv
  (your ground truth) and compare it against the "intent" column.
  That's Phase 2 — for now these stats are enough.
"""

from logger import load_logs, get_stats


def evaluate(verbose: bool = True) -> dict:
    """
    Runs full evaluation on logs.csv.

    Parameters:
      verbose : if True, prints everything to console

    Returns:
      stats dict (same as get_stats())
    """
    df = load_logs()
    stats = get_stats()

    if stats["total"] == 0:
        if verbose:
            print("\nNo logs found. Run the app and submit some prompts first!")
        return stats

    if verbose:
        print("\n" + "=" * 50)
        print("       EVALUATION METRICS")
        print("=" * 50)

        print(f"\n  Total prompts logged : {stats['total']}")
        print(f"  Avg confidence score : {stats['avg_confidence'] * 100:.1f}%")
        print(f"  Avg latency          : {stats['avg_latency_ms']} ms")

        print("\n  --- Intent Distribution ---")
        for intent, count in sorted(stats["intent_counts"].items(), key=lambda x: -x[1]):
            pct = round(count / stats["total"] * 100, 1)
            bar = "█" * int(pct / 5)
            print(f"  {intent:<20} {bar:<20} {count:>3} ({pct}%)")

        print("\n  --- Route Distribution ---")
        route_emojis = {"ODA": "🟢", "Hybrid": "🟡", "Cloud LLM": "🔴"}
        for route, count in sorted(stats["route_counts"].items(), key=lambda x: -x[1]):
            pct = round(count / stats["total"] * 100, 1)
            emoji = route_emojis.get(route, "⚪")
            print(f"  {emoji} {route:<15} {count:>3} ({pct}%)")

        print("\n  --- Confidence Breakdown ---")
        print(f"  High  (>=75%) : {stats['high_confidence']} prompts")
        print(f"  Medium(55-74%): {stats['mid_confidence']} prompts")
        print(f"  Low   (<55%)  : {stats['low_confidence']} prompts")

        print("\n  --- Last 5 Logged Prompts ---")
        last5 = df.tail(5)[["timestamp", "intent", "confidence", "route", "latency_ms"]]
        print(last5.to_string(index=False))

        print("\n" + "=" * 50)

    return stats


def get_metrics_text() -> str:
    """
    Returns metrics as a formatted string — used by the Gradio UI.
    """
    stats = get_stats()

    if stats["total"] == 0:
        return "No logs yet. Submit some prompts in the app first!"

    lines = [
        f"Total prompts:       {stats['total']}",
        f"Avg confidence:      {stats['avg_confidence'] * 100:.1f}%",
        f"Avg latency:         {stats['avg_latency_ms']} ms",
        "",
        "Intent breakdown:",
    ]
    for k, v in sorted(stats["intent_counts"].items(), key=lambda x: -x[1]):
        pct = round(v / stats["total"] * 100, 1)
        lines.append(f"  {k:<20} {v} ({pct}%)")

    lines.append("")
    lines.append("Route breakdown:")
    route_emojis = {"ODA": "🟢", "Hybrid": "🟡", "Cloud LLM": "🔴"}
    for k, v in sorted(stats["route_counts"].items(), key=lambda x: -x[1]):
        pct = round(v / stats["total"] * 100, 1)
        emoji = route_emojis.get(k, "⚪")
        lines.append(f"  {emoji} {k:<15} {v} ({pct}%)")

    lines.append("")
    lines.append("Confidence breakdown:")
    lines.append(f"  High  (>=75%) : {stats['high_confidence']}")
    lines.append(f"  Medium(55-74%): {stats['mid_confidence']}")
    lines.append(f"  Low   (<55%)  : {stats['low_confidence']}")

    return "\n".join(lines)


# ── Run directly to print metrics ────────────────────────────────────────────
if __name__ == "__main__":
    evaluate(verbose=True)
