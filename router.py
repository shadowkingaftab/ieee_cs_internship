"""
router.py
---------
The decision-maker of the system.
Takes the intent matrix and text from classifier.py
and decides WHERE the task should be processed.
"""

def route_task(intent_matrix: dict, text: str) -> str:
    """
    Core routing function using the full intent matrix and context.
    """
    if not intent_matrix:
        return "Cloud LLM"

    top_intent, top_confidence = max(intent_matrix.items(), key=lambda x: x[1])

    # ── Rule 1: Urgent tasks bypass ODA ──────────────────────────────────────
    if "urgent" in text.lower() or intent_matrix.get("urgent", 0) > 0.5:
        return "Cloud LLM"

    # ── Rule 2: Multi-step always needs both ────────────────────────────────
    if intent_matrix.get("multi_step", 0) > 0.3:
        return "Hybrid"

    # ── Rule 3: Low confidence = don't trust the edge model ─────────────────
    if top_confidence < 0.55:
        return "Cloud LLM"

    # ── Rule 4: High confidence + simple task = safe for edge ───────────────
    if top_confidence >= 0.75:
        if top_intent in ["summarize", "translate", "simple_query"]:
            return "ODA"
        else:
            return "Cloud LLM"

    # ── Rule 5: Medium confidence = hedge with hybrid ───────────────────────
    return "Hybrid"

def get_route(intent: str, confidence: float) -> str:
    """Legacy fallback for older tools"""
    if confidence < 0.55:
        return "Cloud LLM"
    if intent in ["multi-step task", "multi_step"]:
        return "Hybrid"
    if confidence >= 0.75 and intent in ["question", "instruction", "summarize", "translate"]:
        return "ODA"
    if confidence >= 0.75 and intent in ["analysis", "creative request", "analyze_data", "generate_code"]:
        return "Cloud LLM"
    return "Hybrid"

def get_route_color(route: str) -> str:
    colors = {
        "ODA":       "🟢",
        "Hybrid":    "🟡",
        "Cloud LLM": "🔴"
    }
    return colors.get(route, "⚪")

def get_route_explanation(intent_matrix: dict, route: str) -> str:
    """Returns an explanation based on the intent matrix."""
    if not intent_matrix:
        return "No intent matrix provided."
        
    top_intent, top_confidence = max(intent_matrix.items(), key=lambda x: x[1])
    pct = round(top_confidence * 100, 1)

    if intent_matrix.get("urgent", 0) > 0.5:
        return "Task marked as urgent. Routing directly to Cloud LLM."
    if intent_matrix.get("multi_step", 0) > 0.3:
        return "Multi-step tasks contain sub-tasks that need different resources. Hybrid routing splits lightweight steps to edge, complex steps to cloud."
    if top_confidence < 0.55:
        return f"Confidence is low ({pct}%) — the model is uncertain about intent. Routing to Cloud LLM for safer processing."
    if route == "ODA":
        return f"High confidence ({pct}%) on a simple '{top_intent}'. On-device AI can handle this fast, privately, without internet."
    if route == "Cloud LLM":
        return f"'{top_intent.capitalize()}' requires deep reasoning or generation. Sending to Cloud LLM for full model capability."
    
    return f"Medium confidence ({pct}%) — not certain enough for pure edge, not complex enough to force cloud. Using Hybrid approach."
