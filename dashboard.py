"""
dashboard.py
------------
Premium multi-tab Streamlit dashboard for the Edge AI Intent Classifier.
Now shows actual Qwen2-0.8B (ODA) and Grok (Cloud) model outputs.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from classifier import classify_prompt
from router import route_task, get_route_explanation
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
    .output-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.9em;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: Load logs fresh from disk (no caching) ───────────────────────────
def load_logs_fresh() -> pd.DataFrame:
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
**Route → Model:**
- 🟢 **ODA** → Qwen2-0.8B (local)
- 🟡 **Hybrid** → Qwen + Grok
- 🔴 **Cloud LLM** → Grok API
""")
    st.markdown("---")

    # API Key input in sidebar
    st.markdown("### 🔑 Grok API Key")
    grok_key = st.text_input("Paste your Grok API key:", type="password", key="grok_key_input")
    if grok_key:
        os.environ["GROK_API_KEY"] = grok_key
        st.success("Key loaded for this session.")
    else:
        existing = os.environ.get("GROK_API_KEY", "")
        if existing:
            st.success("Key detected from environment.")
        else:
            st.warning("No key set — Cloud/Hybrid routes will show a warning.")

    st.markdown("---")
    st.info("**Classifier Model:**\n`typeform/distilbert-base-uncased-mnli`\n\n**ODA Model:**\n`Qwen/Qwen2-0.8B`")
    st.markdown("---")
    if st.button("🗑️ Clear All Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Logs cleared!")


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🤖 Edge AI Intent Classifier Dashboard")
st.markdown("Classify prompts → route to **Qwen2-0.8B** (edge) or **Grok** (cloud) → see real outputs.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify & Execute", "📊 Analytics", "📜 Logs"])


# ════════════════════════════════════════════════════════════
# TAB 1 – CLASSIFY & EXECUTE
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("Classify & Execute a Prompt")
    st.markdown("The classifier detects intent, the router picks the model, and the model runs your prompt.")

    prompt = st.text_area(
        "Your prompt:",
        placeholder="e.g. Summarize the benefits of edge AI in IoT systems.",
        height=100,
        key="prompt_input"
    )
    classify_btn = st.button("⚡ Classify, Route & Execute", type="primary")

    if classify_btn:
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Step 1/2 — Classifying intent…"):
                clf_result = classify_prompt(prompt)

            route_label = None
            with st.spinner(f"Step 2/2 — Executing via {route_label or 'best route'}…"):
                route_result = route_task(clf_result["intent_matrix"], prompt)

            # Log it
            clf_result["latency_ms"] = route_result["latency_ms"]
            log_result(clf_result, route_result["route"])

            st.markdown("---")

            # ── Routing badge ───────────────────────────────────────────────
            color_map = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"}
            icon_map  = {"ODA": "🟢",       "Hybrid": "🟡",       "Cloud LLM": "🔴"}
            desc_map  = {
                "ODA":       "Executed by **Qwen2-0.8B** — local, private, no internet needed.",
                "Hybrid":    "Edge portion by **Qwen2-0.8B** · Cloud portion by **Grok**.",
                "Cloud LLM": "Executed by **Grok API** — full cloud power."
            }
            route = route_result["route"]
            st.markdown(f"""
            <div style="background:{color_map[route]}; padding:18px; border-radius:12px; color:white; margin-bottom:16px;">
                <h2 style="margin:0;">{icon_map[route]} {route}</h2>
                <p style="margin:4px 0 0 0; opacity:0.92;">{desc_map[route]}</p>
            </div>
            """, unsafe_allow_html=True)

            reason = get_route_explanation(clf_result["intent_matrix"], route)
            st.info(f"**Why:** {reason}")

            # ── Model output ─────────────────────────────────────────────────
            st.subheader("📤 Model Output")
            if route == "Hybrid":
                col_e, col_c = st.columns(2)
                with col_e:
                    st.markdown("**🟢 Qwen2-0.8B (Edge)**")
                    st.markdown(f"<div class='output-box'>{route_result.get('edge_output','')}</div>",
                                unsafe_allow_html=True)
                with col_c:
                    st.markdown("**🔴 Grok (Cloud)**")
                    st.markdown(f"<div class='output-box'>{route_result.get('cloud_output','')}</div>",
                                unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='output-box'>{route_result.get('output','')}</div>",
                            unsafe_allow_html=True)

            # ── Intent Matrix + Metrics ──────────────────────────────────────
            st.markdown("---")
            left, right = st.columns(2)

            with left:
                st.subheader("📊 Intent Matrix")
                matrix = clf_result["intent_matrix"]
                df_m = (pd.DataFrame({"Intent": list(matrix.keys()),
                                       "Confidence": list(matrix.values())})
                          .sort_values("Confidence", ascending=True))
                fig = px.bar(df_m, x="Confidence", y="Intent", orientation="h",
                             color="Confidence", color_continuous_scale="Viridis",
                             labels={"Confidence": "Score"})
                fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                  showlegend=False, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("⚡ Performance")
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Top Intent:</strong> {clf_result['top_intent']} &nbsp;({clf_result['top_confidence']*100:.1f}%)</p>
                    <p><strong>Total Latency:</strong> {route_result['latency_ms']:.0f} ms</p>
                    <p><strong>Multi-Step?</strong> {'✅ Yes' if clf_result.get('is_multi_step') else '❌ No'}</p>
                    <p><strong>Urgent?</strong> {'✅ Yes' if matrix.get('urgent',0) > 0.5 else '❌ No'}</p>
                    <p><strong>Model Used:</strong> {"Qwen2-0.8B" if route=="ODA" else "Grok" if route=="Cloud LLM" else "Qwen + Grok"}</p>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 – ANALYTICS
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Analytics Dashboard")

    if st.button("🔄 Refresh"):
        st.rerun()

    logs = load_logs_fresh()

    if logs.empty:
        st.warning("⚠️ No data yet — classify some prompts first!")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Prompts",  len(logs))
        k2.metric("Avg Latency",    f"{logs['latency_ms'].mean():.0f} ms")
        k3.metric("Avg Confidence", f"{logs['confidence'].mean()*100:.1f}%")
        k4.metric("Cloud Routes",   len(logs[logs["route"] == "Cloud LLM"]))

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Route Distribution")
            rc = logs["route"].value_counts().reset_index()
            rc.columns = ["Route", "Count"]
            cmap = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"}
            fig_pie = px.pie(rc, names="Route", values="Count",
                             color="Route", color_discrete_map=cmap)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Confidence Distribution")
            fig_hist = px.histogram(logs, x="confidence", nbins=15,
                                    color_discrete_sequence=["#6c5ce7"],
                                    labels={"confidence": "Confidence Score"})
            fig_hist.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Latency Over Time")
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])
        cmap = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"}
        fig_line = px.scatter(logs, x="timestamp", y="latency_ms",
                              color="route", color_discrete_map=cmap,
                              labels={"latency_ms": "Latency (ms)", "timestamp": "Time"})
        fig_line.update_traces(mode="markers+lines")
        st.plotly_chart(fig_line, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 – LOGS
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("📜 Classification Logs")

    if st.button("🔄 Refresh Logs"):
        st.rerun()

    logs = load_logs_fresh()

    if logs.empty:
        st.warning("⚠️ No logs yet — classify some prompts first!")
    else:
        f1, f2 = st.columns(2)
        with f1:
            route_opts = logs["route"].dropna().unique().tolist()
            sel_route = st.multiselect("Filter by Route:", route_opts, default=route_opts)
        with f2:
            intent_opts = logs["intent"].dropna().unique().tolist()
            sel_intent = st.multiselect("Filter by Intent:", intent_opts, default=intent_opts)

        filtered = logs.copy()
        if sel_route:
            filtered = filtered[filtered["route"].isin(sel_route)]
        if sel_intent:
            filtered = filtered[filtered["intent"].isin(sel_intent)]

        display_cols = ["timestamp", "prompt", "intent", "confidence", "route", "latency_ms"]
        st.dataframe(filtered[display_cols], use_container_width=True, height=400)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "classification_logs.csv", "text/csv")

        with st.expander("🔬 Raw intent_matrix data"):
            st.dataframe(filtered[["timestamp", "prompt", "intent_matrix"]], use_container_width=True)


# ── Debug Info ────────────────────────────────────────────────────────────────
with st.expander("🔍 Debug Info"):
    st.code(f"Log file : {LOG_FILE}")
    st.code(f"Exists   : {os.path.isfile(LOG_FILE)}")
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        st.code(f"Rows: {len(lines)-1}\nLast 3 rows:\n" + "".join(lines[-3:]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;'>"
    "🤖 Edge AI Classifier · Qwen2-0.8B (ODA) · Grok (Cloud) · MCP + Corsair Ready"
    "</div>",
    unsafe_allow_html=True
)
