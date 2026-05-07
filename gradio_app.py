"""
app.py
------
The web UI for your Edge AI Intent Classifier.
Built with Gradio — a Python library for ML demo UIs.

What this UI does:
  1. Takes a text prompt from the user
  2. Runs it through classifier.py → router.py → logger.py
  3. Shows: intent, confidence, route, latency, all scores
  4. Shows a live table of all logged prompts
  5. Has a metrics panel that summarizes all history

To run:
  python app.py
  Then open http://localhost:7860 in your browser
"""

import gradio as gr
from classifier import classify_prompt
from router import get_route, get_route_explanation, get_route_color
from logger import log_result, load_logs
from evaluator import get_metrics_text


# ── Core pipeline function ────────────────────────────────────────────────────
def process_prompt(prompt: str):
    """
    This function runs when the user clicks "Classify & Route".
    It chains together: classify → route → log → return results.
    Returns 7 values, one for each output component in the UI.
    """

    # Guard: empty input
    if not prompt or not prompt.strip():
        # Explicit tuple to avoid any unhashable star-expansion issues in some versions
        return "—", "—", "—", "—", "—", "", load_logs()

    try:
        # ── Step 1: Classify ─────────────────────────────────────────────────
        result = classify_prompt(prompt)

        # ── Step 2: Route ────────────────────────────────────────────────────
        route       = get_route(result["intent"], result["confidence"])
        explanation = get_route_explanation(result["intent"], result["confidence"], route)
        color       = get_route_color(route)

        # ── Step 3: Log ──────────────────────────────────────────────────────
        log_result(result, route)

        # ── Step 4: Format for display ───────────────────────────────────────
        intent_display     = result["intent"].upper()
        confidence_display = f"{result['confidence'] * 100:.1f}%"
        route_display      = f"{color} {route}"
        latency_display    = f"{result['latency_ms']} ms"

        # All scores formatted as a readable list
        scores_display = "\n".join([
            f"  {label:<20} {score * 100:.1f}%"
            for label, score in sorted(
                result["all_scores"].items(),
                key=lambda x: -x[1]
            )
        ])

        # Load updated log table
        df = load_logs()

        return (
            intent_display,
            confidence_display,
            route_display,
            latency_display,
            explanation,
            scores_display,
            df
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "—", "—", "—", "—", "", load_logs()


def refresh_metrics():
    """Called when user clicks Refresh Metrics."""
    return get_metrics_text()


def clear_input():
    """Clears the input box."""
    return ""


# ── Example prompts to help the user test quickly ────────────────────────────
EXAMPLES = [
    ["What is the difference between edge AI and cloud AI?"],
    ["Translate this paragraph into Kannada"],
    ["Analyze the weekly sensor data and generate a summary report"],
    ["Summarize this PDF and then email it to my team"],
    ["Write a short story about a robot learning to feel emotions"],
    ["Run the diagnostics and upload the results"],
    ["Design a new circuit board layout for the IoT module"],
]


# ── Build the Gradio UI ───────────────────────────────────────────────────────
with gr.Blocks(
    title="Edge AI Intent Classifier",
    theme="soft"
) as demo:

    # Header
    gr.Markdown("# 🧠 Edge AI Intent Classifier")
    gr.Markdown(
        "Enter a prompt → detect its **intent** → get a **confidence-based routing decision** "
        "(ODA / Hybrid / Cloud LLM) → auto-log to CSV."
    )

    # ── Input + Results Row ───────────────────────────────────────────────────
    with gr.Row():

        # Left column: input
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Enter your prompt",
                placeholder='e.g. "Analyze sensor data and send a report to the team"',
                lines=4
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Classify & Route", variant="primary", scale=3)
                clear_btn  = gr.Button("🗑 Clear", scale=1)

        # Right column: results
        with gr.Column(scale=1):
            intent_out     = gr.Textbox(label="🎯 Detected Intent")
            confidence_out = gr.Textbox(label="📊 Confidence Score")
            route_out      = gr.Textbox(label="📡 Routing Decision")
            latency_out    = gr.Textbox(label="⚡ Latency")

    # Explanation + all scores
    with gr.Row():
        explanation_out = gr.Textbox(
            label="💡 Why this route?",
            lines=3,
            interactive=False
        )
        scores_out = gr.Textbox(
            label="📋 All Intent Scores",
            lines=7,
            interactive=False
        )

    # Example prompts
    gr.Examples(
        examples=EXAMPLES,
        inputs=prompt_input,
        label="Try these example prompts"
    )

    # ── Logs Table ────────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📁 Logged Results (logs.csv)")
    logs_table = gr.Dataframe(
        value=load_logs(),
        label="All logged prompts",
        interactive=False,
        wrap=True
    )

    # ── Metrics Panel ─────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### 📈 Evaluation Metrics")
    metrics_btn = gr.Button("🔄 Refresh Metrics")
    metrics_out = gr.Textbox(
        label="Metrics Summary",
        lines=16,
        interactive=False
    )

    # ── Wire up the buttons ───────────────────────────────────────────────────
    submit_btn.click(
        fn=process_prompt,
        inputs=[prompt_input],
        outputs=[
            intent_out,
            confidence_out,
            route_out,
            latency_out,
            explanation_out,
            scores_out,
            logs_table
        ]
    )

    clear_btn.click(
        fn=clear_input,
        inputs=[],
        outputs=[prompt_input]
    )

    metrics_btn.click(
        fn=refresh_metrics,
        inputs=[],
        outputs=[metrics_out]
    )


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True
    )
