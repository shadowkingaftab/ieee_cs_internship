"""
dashboard.py
------------
Premium multi-tab Streamlit dashboard for the Edge AI Intent Classifier.
Grok API (primary) + Qwen-1.8B GGUF (local fallback, ~300MB)
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
from qwen_oda import is_model_downloaded, get_model_size_mb, download_model, get_hardware_info, get_quant_options
from grok_cloud import is_api_key_set

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

# streamlit_app.py
# ─────────────────────────────────────────────────────────────────────────────
# ── Premium Theme Styling ──────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;700&display=swap" rel="stylesheet">

<style>
    /* Global Reset & Typography */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .stHeader {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(90deg, #a78bfa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Glassmorphism Cards */
    .metric-card, .output-box, div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border 0.2s ease;
        color: #f1f5f9 !important;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    /* Text Legibility */
    p, span, label, .stMarkdown {
        color: #cbd5e1 !important;
        font-weight: 400;
    }
    
    strong { color: #f8fafc !important; font-weight: 600; }

    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6d28d9 0%, #4c1d95 100%);
        color: #ffffff !important;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(109, 40, 217, 0.39);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
        box-shadow: 0 6px 20px rgba(109, 40, 217, 0.5);
        transform: scale(1.02);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        color: #94a3b8 !important;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(109, 40, 217, 0.2) !important;
        border-color: #7c3aed !important;
        color: #e879f9 !important;
    }

    /* Text Areas & Inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
    }

    /* Output Box Special */
    .output-box {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        background: #020617 !important;
        border-left: 4px solid #8b5cf6;
        color: #38bdf8 !important; /* Cyber Blue output text */
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Hide Sidebar Default border */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
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
**Route -> Model:**
- ODA -> Qwen2-0.5B GGUF (local, ~379MB)
- Hybrid -> Qwen (edge) + Grok (cloud)
- Cloud LLM -> Grok API *(primary)* + Qwen *(fallback)*
""")
    st.markdown("---")

    # ── Grok API Key ──────────────────────────────────────────────────────────
    st.markdown("### 🔑 Grok API Key (Primary)")
    grok_key = st.text_input(
        "Paste your xAI Grok API key:",
        type="password",
        key="grok_key_input",
        help="Get a free key at https://console.x.ai"
    )
    if grok_key:
        os.environ["GROK_API_KEY"] = grok_key
        st.success("✅ Key loaded for this session.")
    elif is_api_key_set():
        st.success("✅ Key detected from .env / environment.")
    else:
        st.warning("⚠️ No key — Cloud/Hybrid will fall back to local Qwen.")

    st.markdown("---")

    # ── Qwen GGUF Model Status ────────────────────────────────────────────────
    st.markdown("### 🖥️ Local Qwen Model (Fallback)")
    if is_model_downloaded():
        size_mb = get_model_size_mb()
        st.success(f"✅ Downloaded ({size_mb:.0f} MB)")
    else:
        st.warning("⚠️ Not downloaded yet")
        if st.button("📥 Download Qwen GGUF (~300MB)", key="dl_qwen"):
            progress_bar = st.progress(0, text="Downloading…")

            def _progress(done, total):
                if total > 0:
                    progress_bar.progress(
                        min(done / total, 1.0),
                        text=f"Downloading… {done // (1024*1024)} / {total // (1024*1024)} MB"
                    )

            ok = download_model(progress_callback=_progress)
            if ok:
                st.success("✅ Qwen model ready!")
                st.rerun()
            else:
                st.error("❌ Download failed. Check internet connection.")

    st.markdown("---")
    st.info(
        "**Classifier:** `distilbert-base-uncased-mnli`\n\n"
        "**ODA / Fallback:** `Qwen2-0.5B-Instruct-Q4_K_M.gguf` (~379MB)\n\n"
        "**Cloud (primary):** `grok-3-mini`"
    )
    st.markdown("---")
    if st.button("🗑️ Clear All Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Logs cleared!")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; margin-bottom:40px;'>
        <h1 style='font-size:3.5rem; margin-bottom:0;'>🧠 Edge AI Dashboard</h1>
        <p style='font-size:1.2rem; opacity:0.8;'>Intelligent Intent Classification · Cloud-Primary · Edge-Fallback</p>
    </div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Analytics", "Logs", "ODA Insights"])


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
            glow_map  = {"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#ef4444"}
            icon_map  = {"ODA": "🟢",       "Hybrid": "🟡",       "Cloud LLM": "🔴"}
            desc_map  = {
                "ODA":       "Edge-Only: Local, private, and ultra-fast.",
                "Hybrid":    "Co-Processor: Distributed between Edge and Cloud.",
                "Cloud LLM": "Remote Intelligence: Full Grok-3 power."
            }
            route = route_result["route"]
            st.markdown(f"""
            <div style="
                background: rgba(30, 41, 59, 0.4);
                backdrop-filter: blur(10px);
                border: 2px solid {glow_map[route]};
                padding: 20px;
                border-radius: 16px;
                box-shadow: 0 0 20px {glow_map[route]}33;
                margin-bottom: 24px;
            ">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:24px;">{icon_map[route]}</span>
                    <h2 style="margin:0; background:none; -webkit-text-fill-color:{glow_map[route]};">{route}</h2>
                </div>
                <p style="margin:8px 0 0 0; color:#f1f5f9 !important; font-size:1.1em; font-weight:500;">{desc_map[route]}</p>
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
                             color="Confidence", color_continuous_scale="Purples",
                             template="plotly_dark",
                             labels={"Confidence": "Score"})
                fig.update_layout(
                    height=360, 
                    margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, 
                    xaxis_title="", 
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("⚡ Performance")
                model_used_label = route_result.get("model_used", "Unknown")
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Top Intent:</strong> {clf_result['top_intent']} &nbsp;({clf_result['top_confidence']*100:.1f}%)</p>
                    <p><strong>Total Latency:</strong> {route_result['latency_ms']:.0f} ms</p>
                    <p><strong>Multi-Step?</strong> {'✅ Yes' if clf_result.get('is_multi_step') else '❌ No'}</p>
                    <p><strong>Urgent?</strong> {'✅ Yes' if matrix.get('urgent',0) > 0.5 else '❌ No'}</p>
                    <p><strong>Model Used:</strong> {model_used_label}</p>
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



# ===========================================================
# TAB 4 - ODA INSIGHTS
# ===========================================================
with tab4:
    import json as _json
    st.header("ODA Optimization Insights")
    st.markdown(
        "Identify **Cloud LLM** prompts that could move to **Edge ODA** "
        "to cut costs and latency. Use this view to guide fine-tuning."
    )
    if st.button("Refresh", key="refresh_insights"):
        st.rerun()

    logs_oda = load_logs_fresh()

    # Hardware panel
    with st.expander("Hardware & Quantization Settings", expanded=True):
        hw = get_hardware_info()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("CPU Cores",    str(hw["cpu_physical"]) + " phys")
        h2.metric("RAM",          str(hw["ram_gb"]) + " GB")
        h3.metric("Threads Used", str(hw["n_threads"]))
        h4.metric("Quant Level",  hw["quant_level"].upper())

        st.markdown("##### Quantization Comparison")
        st.markdown(
            "| Level | Size | Speed | Quality | Best For |\n"
            "|-------|------|-------|---------|----------|\n"
            "| **q2_k** | 323 MB | Fastest | Lower | Raspberry Pi, old laptops |\n"
            "| **q4_k_m** | 379 MB | Balanced | Good | Most PCs (default) |\n"
            "| **q5_k_m** | 444 MB | Slower | Better | 8GB+ RAM machines |\n"
            "| **q8_0** | 506 MB | Slowest | Best | Servers, workstations |"
        )
        st.info(
            f"Active: **{hw['quant_level'].upper()}** ({hw['model_size_mb']} MB) | "
            f"Threads: **{hw['n_threads']}** | Batch: **{hw['n_batch']}** | "
            f"GPU layers: **{hw['n_gpu_layers']}**\n\n"
            "To change: set `QWEN_QUANT=q2_k` in your `.env` and restart."
        )

    st.markdown("---")

    if logs_oda.empty:
        st.warning("No logs yet. Classify some prompts in Tab 1 first!")
    else:
        cloud_rows = logs_oda[logs_oda["route"] == "Cloud LLM"].copy()
        st.subheader("Cloud LLM Prompts - ODA Migration Candidates")

        if cloud_rows.empty:
            st.success("All traffic is on Edge ODA or Hybrid - nothing to migrate!")
        else:
            cloud_rows["confidence"] = pd.to_numeric(cloud_rows["confidence"], errors="coerce")
            cands     = cloud_rows[(cloud_rows["confidence"] >= 0.50) & (cloud_rows["confidence"] < 0.75)]
            high_conf = cloud_rows[cloud_rows["confidence"] >= 0.75]

            ca, cb, cc = st.columns(3)
            ca.metric("Total Cloud LLM",       len(cloud_rows))
            cb.metric("ODA Candidates 50-74%", len(cands))
            cc.metric("High-Conf Cloud >=75%", len(high_conf))

            if not cands.empty:
                st.markdown("##### Borderline prompts (fine-tuning pushes these to ODA)")
                disp = cands[["timestamp", "prompt", "intent", "confidence", "latency_ms"]].copy()
                disp["confidence"] = (disp["confidence"] * 100).round(1).astype(str) + "%"
                st.dataframe(disp, use_container_width=True, height=280)

                freq = cands["intent"].value_counts().reset_index()
                freq.columns = ["Intent", "Count"]
                fig_bar = px.bar(
                    freq, x="Intent", y="Count",
                    color="Count", color_continuous_scale="Viridis",
                    title="Intents needing more training examples"
                )
                fig_bar.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

                export = [
                    {"text": r["prompt"], "label": r["intent"]}
                    for _, r in cands.iterrows()
                    if pd.notna(r["prompt"]) and pd.notna(r["intent"])
                ]
                st.download_button(
                    "Download as Training JSON",
                    _json.dumps(export, indent=2),
                    "oda_candidates_training.json",
                    "application/json",
                    key="dl_training"
                )

        st.markdown("---")
        st.subheader("Confidence Distribution by Route")
        logs_oda["confidence"] = pd.to_numeric(logs_oda["confidence"], errors="coerce")
        fig_box = px.box(
            logs_oda.dropna(subset=["confidence", "route"]),
            x="route", y="confidence", color="route",
            color_discrete_map={"ODA": "#2ecc71", "Hybrid": "#f39c12", "Cloud LLM": "#e74c3c"},
            title="Confidence by Route (ideal: ODA > 75%)",
            labels={"confidence": "Confidence Score", "route": "Route"},
        )
        fig_box.add_hline(y=0.75, line_dash="dash", line_color="grey",
                          annotation_text="ODA threshold (75%)")
        fig_box.add_hline(y=0.55, line_dash="dot", line_color="red",
                          annotation_text="Cloud floor (55%)")
        fig_box.update_layout(height=340)
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.subheader("Fine-Tuning Guide - Hit >95% Confidence")
        st.markdown("""
1. **Export candidates** above as JSON
2. **Merge** into `training/domain_dataset.json`
3. **Run fine-tuning:**
```bash
python training/train_classifier.py --epochs 5 --eval
```
4. **Set in `.env`:**
```
CLASSIFIER_MODEL_PATH=models/fine_tuned_classifier
```
5. **Restart** the dashboard - it auto-loads the fine-tuned model
        """)



# ── Debug Info ────────────────────────────────────────────────────────────────
with st.expander("🔍 Debug Info"):
    st.code(f"Log file : {LOG_FILE}")
    st.code(f"Exists   : {os.path.isfile(LOG_FILE)}")
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        st.code(f"Rows: {len(lines)-1}\nLast 3 rows:\n" + "".join(lines[-3:]))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; padding: 20px; opacity: 0.6; font-size: 0.9rem;'>
        <hr style='border-color: rgba(148, 163, 184, 0.1);'>
        🤖 <strong>Edge AI Classifier</strong> &nbsp;|&nbsp; 
        ⚡ <strong>Qwen2-0.5B GGUF</strong> (Local) &nbsp;|&nbsp; 
        ☁️ <strong>Grok API</strong> (Cloud) &nbsp;|&nbsp; 
        🛡️ <strong>MCP + Corsair Ready</strong>
    </div>
""", unsafe_allow_html=True)
