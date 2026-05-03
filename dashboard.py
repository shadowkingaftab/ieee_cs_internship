"""
dashboard.py
------------
Premium multi-tab Streamlit dashboard for the Edge AI Intent Classifier.
Matches logger.py column names EXACTLY: timestamp, prompt, intent,
confidence, intent_matrix, route, latency_ms
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import time
from classifier import classify_prompt
from router import route_task
from logger import log_result

# ── Constants ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE   = os.path.join(script_dir, "logs.csv")
COLUMNS    = ["timestamp", "prompt", "intent", "confidence", "intent_matrix", "route", "latency_ms"]

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge AI Intent Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #6c5ce7;
        margin-bottom: 10px;
    }
    h1 { color: #2d3436; }
    h2 { color: #6c5ce7; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c5ce7 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover { background-color: #5649d1; }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load logs fresh from disk ────────────────────────────────────────
def load_logs_fresh() -> pd.DataFrame:
    """Always reads from disk. Never cached."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(LOG_FILE)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[COLUMNS]
    except Exception:
        return pd.DataFrame(columns=COLUMNS)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Edge AI Classifier")
    st.markdown("---")
    st.markdown("""
**Route Legend:**
- 🟢 **ODA** — On-Device AI (fast, private, offline)
- 🟡 **Hybrid** — Edge + Cloud split
- 🔴 **Cloud LLM** — Full cloud model
""")
    st.markdown("---")
    st.info("**Model:** `typeform/distilbert-base-uncased-mnli`\n\n**Size:** ~268 MB\n\n**Type:** Zero-shot NLI")
    st.markdown("---")
    if st.button("🗑️ Clear All Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Logs cleared!")
        else:
            st.info("No logs to clear.")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🤖 Edge AI Intent Classifier Dashboard")
st.markdown("Classify prompts, inspect intent matrices, and monitor routing decisions in real-time.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Analytics", "📜 Logs"])


# ════════════════════════════════════════════════════════════
# TAB 1 – CLASSIFY
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("Classify a Prompt")
    st.markdown("Enter a natural-language prompt to detect its intent and get a routing decision.")

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        prompt = st.text_area(
            "Your prompt:",
            placeholder="e.g. Summarize my emails from last week and translate them to French.",
            height=80,
            key="prompt_input"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        classify_btn = st.button("⚡ Classify & Route", type="primary")

    if classify_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt before classifying.")
        else:
            with st.spinner("Running intent classification…"):
                result = classify_prompt(prompt)
                route  = route_task(result["intent_matrix"], prompt)
                log_result(result, route)   # Write to logs.csv immediately

            st.markdown("---")
            st.subheader("🎯 Results")

            left, right = st.columns(2)

            # ── Left: Routing Decision ──────────────────────────────────────
            with left:
                st.markdown("### Routing Decision")

                color_map = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"}
                icon_map  = {"ODA": "🟢",       "Hybrid": "🟡",       "Cloud LLM": "🔴"}
                desc_map  = {
                    "ODA":       "On-Device AI · Fast, Private, Offline",
                    "Hybrid":    "Edge + Cloud Split",
                    "Cloud LLM": "Full Cloud Model"
                }
                rc = color_map.get(route, "#888")
                ri = icon_map.get(route, "⚪")
                rd = desc_map.get(route, "")

                st.markdown(f"""
                <div style="background:{rc}; padding:20px; border-radius:12px; color:white; margin-bottom:12px;">
                    <h2 style="margin:0;">{ri} {route}</h2>
                    <p style="margin:4px 0 0 0; opacity:0.9;">{rd}</p>
                </div>
                """, unsafe_allow_html=True)

                # Reasoning
                conf = result["top_confidence"]
                if conf < 0.55:
                    reason = f"Confidence too low ({conf*100:.1f}%) — Cloud LLM provides safer processing."
                elif result.get("is_multi_step"):
                    reason = "Multi-step task detected — Hybrid splits sub-tasks between edge and cloud."
                elif conf >= 0.75:
                    reason = f"High confidence ({conf*100:.1f}%) — safe to route to {route}."
                else:
                    reason = f"Medium confidence ({conf*100:.1f}%) — Hybrid provides the right balance."

                st.info(f"**Why:** {reason}")

                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Top Intent:</strong> {result['top_intent']} &nbsp;({conf*100:.1f}%)</p>
                    <p><strong>Latency:</strong> {result['latency_ms']:.1f} ms</p>
                    <p><strong>Multi-Step?</strong> {'✅ Yes' if result.get('is_multi_step') else '❌ No'}</p>
                    <p><strong>Urgent?</strong> {'✅ Yes' if result['intent_matrix'].get('urgent', 0) > 0.5 else '❌ No'}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Right: Intent Matrix Chart ──────────────────────────────────
            with right:
                st.markdown("### Intent Matrix")
                matrix = result["intent_matrix"]
                df_m = (pd.DataFrame({"Intent": list(matrix.keys()),
                                       "Confidence": list(matrix.values())})
                          .sort_values("Confidence", ascending=True))

                fig = px.bar(
                    df_m, x="Confidence", y="Intent", orientation="h",
                    color="Confidence", color_continuous_scale="Viridis",
                    labels={"Confidence": "Score"}
                )
                fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                                  showlegend=False, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 – ANALYTICS
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Analytics Dashboard")

    col_refresh, _, _ = st.columns([1, 3, 1])
    with col_refresh:
        if st.button("🔄 Refresh Analytics"):
            st.rerun()

    logs = load_logs_fresh()   # Always fresh from disk

    if logs.empty:
        st.warning("⚠️ No data yet — classify some prompts first!")
    else:
        # ── KPI row ─────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Prompts",  len(logs))
        k2.metric("Avg Latency",    f"{logs['latency_ms'].mean():.1f} ms")
        k3.metric("Avg Confidence", f"{logs['confidence'].mean()*100:.1f}%")
        k4.metric("Hybrid Routes",  len(logs[logs["route"] == "Hybrid"]))

        st.markdown("---")

        c1, c2 = st.columns(2)

        # Pie chart – route distribution
        with c1:
            st.subheader("Route Distribution")
            route_counts = logs["route"].value_counts().reset_index()
            route_counts.columns = ["Route", "Count"]
            color_map = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"}
            fig_pie = px.pie(
                route_counts, names="Route", values="Count",
                color="Route", color_discrete_map=color_map
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Histogram – confidence distribution
        with c2:
            st.subheader("Confidence Distribution")
            fig_hist = px.histogram(
                logs, x="confidence", nbins=15,
                color_discrete_sequence=["#6c5ce7"],
                labels={"confidence": "Confidence Score"}
            )
            fig_hist.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")

        # Latency over time
        st.subheader("Latency Over Time")
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])
        fig_line = px.scatter(
            logs, x="timestamp", y="latency_ms",
            color="route", color_discrete_map=color_map,
            labels={"latency_ms": "Latency (ms)", "timestamp": "Time"},
            size_max=8
        )
        fig_line.update_traces(mode="markers+lines")
        st.plotly_chart(fig_line, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 – LOGS
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("📜 Classification Logs")

    col_refresh, _, _ = st.columns([1, 3, 1])
    with col_refresh:
        if st.button("🔄 Refresh Logs"):
            st.rerun()

    logs = load_logs_fresh()   # Always fresh from disk

    if logs.empty:
        st.warning("⚠️ No logs yet — classify some prompts first!")
    else:
        f1, f2, f3 = st.columns(3)

        with f1:
            route_opts = logs["route"].dropna().unique().tolist()
            sel_route = st.multiselect("Filter by Route:", route_opts, default=route_opts)
        with f2:
            intent_opts = logs["intent"].dropna().unique().tolist()
            sel_intent = st.multiselect("Filter by Intent:", intent_opts, default=intent_opts)
        with f3:
            sel_date = st.date_input("Filter by Date:", value=None)

        filtered = logs.copy()
        if sel_route:
            filtered = filtered[filtered["route"].isin(sel_route)]
        if sel_intent:
            filtered = filtered[filtered["intent"].isin(sel_intent)]
        if sel_date:
            filtered["_date"] = pd.to_datetime(filtered["timestamp"]).dt.date
            filtered = filtered[filtered["_date"] == sel_date]
            filtered = filtered.drop(columns=["_date"])

        # Display without the raw intent_matrix blob
        display_cols = ["timestamp", "prompt", "intent", "confidence", "route", "latency_ms"]
        st.dataframe(filtered[display_cols], use_container_width=True, height=400)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download as CSV", csv, "classification_logs.csv", "text/csv")

        with st.expander("🔬 Show raw intent_matrix data"):
            st.dataframe(filtered[["timestamp", "prompt", "intent_matrix"]], use_container_width=True)


# ── Debug Info ────────────────────────────────────────────────────────────────
with st.expander("🔍 Debug Info (expand to check)"):
    st.code(f"Log file path : {LOG_FILE}")
    st.code(f"File exists   : {os.path.isfile(LOG_FILE)}")
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        total = len(lines) - 1  # subtract header
        st.code(f"Total rows    : {total}")
        st.code("Last 5 rows:\n" + "".join(lines[-5:]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7f8c8d;'>"
    "Built with ❤️ using Streamlit · "
    "Model: <code>typeform/distilbert-base-uncased-mnli</code>"
    "</div>",
    unsafe_allow_html=True
)
