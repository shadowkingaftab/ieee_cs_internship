"""
router.py
---------
The decision-maker AND executor of the system.

Routes prompts to:
  ODA       → Qwen-1.8B GGUF (local, fast, private, ~300MB)
  Hybrid    → Qwen (edge) + Grok (cloud) working in parallel
  Cloud LLM → Grok API (primary) with automatic Qwen fallback

Every execution path now uses Grok-primary / Qwen-fallback logic,
so the system degrades gracefully even without an internet connection.
"""

import time


# ── Lazy imports ──────────────────────────────────────────────────────────────
def _qwen():
    """Lazy-import run_qwen from qwen_oda."""
    try:
        from qwen_oda import run_qwen
        return run_qwen
    except ImportError:
        return None


def _grok():
    """Lazy-import run_grok and run_grok_with_fallback from grok_cloud."""
    try:
        from grok_cloud import run_grok, run_grok_with_fallback
        return run_grok, run_grok_with_fallback
    except ImportError:
        return None, None


# ── Main router ───────────────────────────────────────────────────────────────
def route_task(intent_matrix: dict, text: str) -> dict:
    """
    Route the task based on the intent matrix, then execute with the right model.

    Execution strategy:
      ODA route       → Qwen GGUF (local, offline-capable)
      Cloud LLM route → Grok API (primary) → Qwen (fallback)
      Hybrid route    → Qwen (edge part) + Grok (cloud part) in sequence

    Returns a dict with:
      route        : "ODA" | "Hybrid" | "Cloud LLM"
      output       : Final response (ODA and Cloud routes)
      edge_output  : Qwen's response (Hybrid only)
      cloud_output : Grok's response (Hybrid only)
      latency_ms   : Total execution time in milliseconds
      model_used   : Actual model that produced the response
    """
    if not intent_matrix:
        return {
            "route":      "Cloud LLM",
            "output":     "No intent matrix provided.",
            "model_used": "none",
            "latency_ms": 0,
        }

    start_time = time.time()
    top_intent, top_confidence = max(intent_matrix.items(), key=lambda x: x[1])
    text_lower = text.lower()

    # ── Routing decision ──────────────────────────────────────────────────────
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

    # ── Get module references ─────────────────────────────────────────────────
    run_qwen = _qwen()
    _, run_grok_with_fallback = _grok()

    # ── ODA: Local Qwen only ──────────────────────────────────────────────────
    if route == "ODA":
        if run_qwen:
            output = run_qwen(text)
            model_used = "Qwen2-0.5B-GGUF"
        else:
            output = (
                "⚠️ Qwen not available.\n"
                "Install: pip install llama-cpp-python"
            )
            model_used = "none"

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "route":      route,
            "output":     output,
            "model_used": model_used,
            "latency_ms": latency_ms,
        }

    # ── Cloud LLM: Grok primary, Qwen fallback ────────────────────────────────
    elif route == "Cloud LLM":
        if run_grok_with_fallback:
            output, source = run_grok_with_fallback(text)
            model_used = "Grok" if source == "grok" else (
                "Qwen2-0.5B-GGUF (Grok fallback)" if source == "qwen" else "none"
            )
        elif run_qwen:
            # grok_cloud.py not importable — go straight to Qwen
            output = run_qwen(text)
            model_used = "Qwen2-0.5B-GGUF (Grok unavailable)"
        else:
            output = "❌ No inference engine available. Set GROK_API_KEY or install llama-cpp-python."
            model_used = "none"

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "route":      route,
            "output":     output,
            "model_used": model_used,
            "latency_ms": latency_ms,
        }

    # ── Hybrid: Qwen handles edge, Grok handles cloud ─────────────────────────
    else:  # Hybrid
        edge_prompt  = f"Handle the first part of this task concisely: {text}"
        cloud_prompt = f"Handle the second part of this task in detail: {text}"

        # Edge (Qwen)
        if run_qwen:
            edge_output  = run_qwen(edge_prompt)
            edge_model   = "Qwen2-0.5B-GGUF"
        else:
            edge_output  = "⚠️ Qwen not available (install llama-cpp-python)."
            edge_model   = "none"

        # Cloud (Grok → Qwen fallback)
        if run_grok_with_fallback:
            cloud_output, cloud_source = run_grok_with_fallback(cloud_prompt)
            cloud_model = "Grok" if cloud_source == "grok" else "Qwen2-0.5B-GGUF (fallback)"
        elif run_qwen:
            cloud_output = run_qwen(cloud_prompt)
            cloud_model  = "Qwen2-0.5B-GGUF"
        else:
            cloud_output = "❌ No cloud or local engine available."
            cloud_model  = "none"

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "route":        route,
            "edge_output":  edge_output,
            "cloud_output": cloud_output,
            "output":       f"[Edge — {edge_model}]\n{edge_output}\n\n[Cloud — {cloud_model}]\n{cloud_output}",
            "model_used":   f"{edge_model} + {cloud_model}",
            "latency_ms":   latency_ms,
        }


# ── Legacy helpers (kept for MCP server backward compatibility) ───────────────
def get_route(intent: str, confidence: float) -> str:
    """Legacy routing for MCP server / older callers."""
    if confidence < 0.55:
        return "Cloud LLM"
    if intent in ["multi-step task", "multi_step"]:
        return "Hybrid"
    if confidence >= 0.75 and intent in [
        "question", "instruction", "summarize", "translate", "simple_query"
    ]:
        return "ODA"
    if confidence >= 0.75 and intent in [
        "analysis", "creative request", "analyze_data", "generate_code"
    ]:
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
        return "Task marked as urgent — routing directly to Grok (Cloud LLM)."
    if intent_matrix.get("multi_step", 0) > 0.3:
        return "Multi-step task — Hybrid splits lightweight steps to Qwen (edge), complex steps to Grok (cloud)."
    if top_confidence < 0.55:
        return f"Low confidence ({pct}%) — Grok Cloud LLM provides safer, deeper processing."
    if route == "ODA":
        return f"High confidence ({pct}%) on a simple '{top_intent}' — Qwen-1.8B handles this locally (fast & private)."
    if route == "Cloud LLM":
        return f"'{top_intent.capitalize()}' needs deep reasoning — sending to Grok (with Qwen fallback)."
    return f"Medium confidence ({pct}%) — Hybrid gives the best balance of speed and quality."
