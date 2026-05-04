"""
router.py
---------
The decision-maker AND executor of the system.

Routes prompts to:
  ODA       → Qwen2-0.8B (local, fast, private)
  Hybrid    → Qwen (edge) + Grok (cloud) working together
  Cloud LLM → Grok API (powerful, full cloud model)

Returns a dict with the route decision AND the actual model output.
"""

import time


def _try_import_qwen():
    """Lazy import — only loads Qwen if we actually need it."""
    try:
        from qwen_oda import run_qwen
        return run_qwen
    except ImportError:
        return None


def _try_import_grok():
    """Lazy import — only loads Grok if we actually need it."""
    try:
        from grok_cloud import run_grok
        return run_grok
    except ImportError:
        return None


def route_task(intent_matrix: dict, text: str) -> dict:
    """
    Route the task based on intent matrix, then execute using the right model.

    Returns a dict with:
      route         : "ODA" | "Hybrid" | "Cloud LLM"
      output        : Final response (for ODA and Cloud routes)
      edge_output   : Qwen's response (for Hybrid only)
      cloud_output  : Grok's response (for Hybrid only)
      latency_ms    : Total execution time in ms
    """
    if not intent_matrix:
        return {"route": "Cloud LLM", "output": "No intent matrix provided.", "latency_ms": 0}

    start_time = time.time()
    top_intent, top_confidence = max(intent_matrix.items(), key=lambda x: x[1])
    text_lower = text.lower()

    # ── Determine route ──────────────────────────────────────────────────────
    if "urgent" in text_lower or intent_matrix.get("urgent", 0) > 0.5:
        route = "Cloud LLM"
    elif intent_matrix.get("multi_step", 0) > 0.3:
        route = "Hybrid"
    elif top_confidence < 0.55:
        route = "Cloud LLM"
    elif top_confidence >= 0.75 and top_intent in ["summarize", "translate", "simple_query"]:
        route = "ODA"
    elif top_confidence >= 0.75:
        route = "Cloud LLM"
    else:
        route = "Hybrid"

    # ── Execute ──────────────────────────────────────────────────────────────
    run_qwen = _try_import_qwen()
    run_grok = _try_import_grok()

    if route == "ODA":
        if run_qwen:
            output = run_qwen(text)
        else:
            output = "[Qwen2-0.8B not available — install with: pip install transformers torch]"
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {"route": route, "output": output, "latency_ms": latency_ms}

    elif route == "Cloud LLM":
        if run_grok:
            output = run_grok(text)
        else:
            output = "[Grok not available — check grok_cloud.py]"
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {"route": route, "output": output, "latency_ms": latency_ms}

    else:  # Hybrid
        edge_output  = run_qwen(f"Handle the first part of this task: {text}") if run_qwen else "[Qwen unavailable]"
        cloud_output = run_grok(f"Handle the second part of this task: {text}") if run_grok else "[Grok unavailable]"
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "route":        route,
            "edge_output":  edge_output,
            "cloud_output": cloud_output,
            "output":       f"[Edge]\n{edge_output}\n\n[Cloud]\n{cloud_output}",
            "latency_ms":   latency_ms
        }


# ── Legacy functions — kept for MCP server backward compatibility ─────────────
def get_route(intent: str, confidence: float) -> str:
    """Legacy routing for MCP server / older callers."""
    if confidence < 0.55:
        return "Cloud LLM"
    if intent in ["multi-step task", "multi_step"]:
        return "Hybrid"
    if confidence >= 0.75 and intent in ["question", "instruction", "summarize", "translate", "simple_query"]:
        return "ODA"
    if confidence >= 0.75 and intent in ["analysis", "creative request", "analyze_data", "generate_code"]:
        return "Cloud LLM"
    return "Hybrid"


def get_route_color(route: str) -> str:
    return {"ODA": "🟢", "Hybrid": "🟡", "Cloud LLM": "🔴"}.get(route, "⚪")


def get_route_explanation(intent_matrix: dict, route: str) -> str:
    if not intent_matrix:
        return "No intent matrix provided."
    top_intent, top_confidence = max(intent_matrix.items(), key=lambda x: x[1])
    pct = round(top_confidence * 100, 1)
    if intent_matrix.get("urgent", 0) > 0.5:
        return "Task marked as urgent — routing directly to Cloud LLM."
    if intent_matrix.get("multi_step", 0) > 0.3:
        return "Multi-step task — Hybrid splits lightweight steps to edge, complex steps to cloud."
    if top_confidence < 0.55:
        return f"Confidence is low ({pct}%) — Cloud LLM provides safer processing."
    if route == "ODA":
        return f"High confidence ({pct}%) on a simple '{top_intent}' — Qwen2-0.8B handles this locally."
    if route == "Cloud LLM":
        return f"'{top_intent.capitalize()}' needs deep reasoning — sending to Grok."
    return f"Medium confidence ({pct}%) — Hybrid provides the right balance."
