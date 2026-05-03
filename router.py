"""
router.py
---------
The decision-maker of the system.

Takes the INTENT and CONFIDENCE from classifier.py
and decides WHERE the task should be processed:

  ODA        = On-Device AI  → fast, private, works offline
  Hybrid     = Split between edge device + cloud
  Cloud LLM  = Full cloud model needed

The routing logic uses BOTH intent AND confidence:
  - High confidence + simple task  → trust edge model (ODA)
  - High confidence + complex task → needs cloud power
  - Low confidence                 → don't trust edge, send to cloud
  - Medium confidence              → hedge with hybrid
  - Multi-step                     → always hybrid (needs both)
"""


def get_route(intent: str, confidence: float) -> str:
    """
    Core routing function.

    Parameters:
      intent     : string label from classifier.py
      confidence : float between 0.0 and 1.0

    Returns:
      "ODA", "Hybrid", or "Cloud LLM"
    """

    # ── Rule 1: Low confidence = don't trust the edge model ─────────────────
    # If the model itself isn't sure what the intent is,
    # it's safer to send to cloud which has more reasoning power.
    if confidence < 0.55:
        return "Cloud LLM"

    # ── Rule 2: Multi-step always needs both ────────────────────────────────
    # Multi-step tasks have sub-tasks that might need different resources.
    # Some sub-tasks go on-device, some go to cloud → Hybrid.
    if intent == "multi-step task":
        return "Hybrid"

    # ── Rule 3: High confidence + simple task = safe for edge ───────────────
    # A clear question or simple instruction? The edge model can handle it.
    if confidence >= 0.75 and intent in ["question", "instruction"]:
        return "ODA"

    # ── Rule 4: High confidence + complex task = send to cloud ──────────────
    # Even if we're sure it's an analysis/creative task,
    # those need the full power of a large cloud model.
    if confidence >= 0.75 and intent in ["analysis", "creative request"]:
        return "Cloud LLM"

    # ── Rule 5: Medium confidence = hedge with hybrid ───────────────────────
    # 0.55–0.74 confidence: not sure enough to fully trust edge,
    # not complex enough to force cloud. Split the work.
    return "Hybrid"


def get_route_color(route: str) -> str:
    """Returns an emoji color indicator for the route."""
    colors = {
        "ODA":       "🟢",
        "Hybrid":    "🟡",
        "Cloud LLM": "🔴"
    }
    return colors.get(route, "⚪")


def get_route_explanation(intent: str, confidence: float, route: str) -> str:
    """
    Returns a human-readable explanation of WHY this route was chosen.
    Useful for the UI and for understanding your system's decisions.
    """
    pct = round(confidence * 100, 1)

    if confidence < 0.55:
        return (
            f"Confidence is low ({pct}%) — the model is uncertain about intent. "
            f"Routing to Cloud LLM for safer, more accurate processing."
        )
    if intent == "multi-step task":
        return (
            f"Multi-step tasks contain sub-tasks that need different resources. "
            f"Hybrid routing splits lightweight steps to edge, complex steps to cloud."
        )
    if route == "ODA":
        return (
            f"High confidence ({pct}%) on a simple '{intent}'. "
            f"On-device AI can handle this fast, privately, without internet."
        )
    if route == "Cloud LLM":
        return (
            f"'{intent.capitalize()}' requires deep reasoning or generation. "
            f"Sending to Cloud LLM for full model capability."
        )
    return (
        f"Medium confidence ({pct}%) — not certain enough for pure edge, "
        f"not complex enough to force cloud. Using Hybrid approach."
    )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("question",        0.92),
        ("analysis",        0.88),
        ("multi-step task", 0.79),
        ("instruction",     0.61),
        ("creative request",0.45),
    ]

    print("\n=== ROUTER SELF-TEST ===\n")
    for intent, conf in test_cases:
        route = get_route(intent, conf)
        explanation = get_route_explanation(intent, conf, route)
        color = get_route_color(route)
        print(f"Intent: {intent:<20} Conf: {conf}  →  {color} {route}")
        print(f"  Why: {explanation}\n")
