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


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🤖 Edge AI Intent Classifier Dashboard")
st.markdown("Classify prompts → route to **Qwen2-0.8B** (edge) or **Grok** (cloud) → see real outputs.")

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
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;'>"
    "🤖 Edge AI Classifier · Qwen2-0.5B GGUF (ODA/Fallback) · Grok API (Cloud/Primary) · MCP + Corsair Ready"
    "</div>",
    unsafe_allow_html=True
)
