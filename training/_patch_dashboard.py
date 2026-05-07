"""
Patch dashboard.py to:
1. Update tabs line to add tab4 (ODA Insights)
2. Append the ODA Insights tab block before the debug/footer section
3. Update footer text
4. Add psutil to requirements.txt
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASH = os.path.join(ROOT, "dashboard.py")
REQ  = os.path.join(ROOT, "requirements.txt")

with open(DASH, "r", encoding="utf-8") as f:
    src = f.read()

# ── 1. Fix tabs line ──────────────────────────────────────────────────────────
OLD_TABS = 'tab1, tab2, tab3 = st.tabs(["'
if OLD_TABS in src:
    src = re.sub(
        r'tab1, tab2, tab3 = st\.tabs\(\[.*?\]\)',
        'tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Analytics", "Logs", "ODA Insights"])',
        src,
        count=1,
        flags=re.DOTALL,
    )
    print("OK: tabs line updated")
else:
    print("WARN: tabs line not found, skipping")

# ── 2. Replace debug/footer with tab4 + debug + footer ───────────────────────
ODA_BLOCK = '''
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
            "| Level | Size | Speed | Quality | Best For |\\n"
            "|-------|------|-------|---------|----------|\\n"
            "| **q2_k** | 323 MB | Fastest | Lower | Raspberry Pi, old laptops |\\n"
            "| **q4_k_m** | 379 MB | Balanced | Good | Most PCs (default) |\\n"
            "| **q5_k_m** | 444 MB | Slower | Better | 8GB+ RAM machines |\\n"
            "| **q8_0** | 506 MB | Slowest | Best | Servers, workstations |"
        )
        st.info(
            f"Active: **{hw['quant_level'].upper()}** ({hw['model_size_mb']} MB) | "
            f"Threads: **{hw['n_threads']}** | Batch: **{hw['n_batch']}** | "
            f"GPU layers: **{hw['n_gpu_layers']}**\\n\\n"
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

'''

# Find the debug/footer block and insert tab4 before it
FOOTER_MARKER = "# ── Debug Info ──"
if FOOTER_MARKER in src:
    src = src.replace(FOOTER_MARKER, ODA_BLOCK + "\n\n" + FOOTER_MARKER, 1)
    print("OK: ODA Insights tab inserted")
else:
    # fallback: append before last st.markdown footer
    src = src + "\n" + ODA_BLOCK
    print("OK: ODA Insights tab appended (fallback)")

# ── 3. Update footer text ─────────────────────────────────────────────────────
src = src.replace(
    "Qwen-1.8B GGUF (ODA/Fallback)",
    "Qwen2-0.5B GGUF (ODA/Fallback)"
)

with open(DASH, "w", encoding="utf-8") as f:
    f.write(src)
print("dashboard.py patched successfully")

# ── 4. Add psutil to requirements.txt ────────────────────────────────────────
with open(REQ, "r", encoding="utf-8") as f:
    req_content = f.read()
if "psutil" not in req_content:
    with open(REQ, "a", encoding="utf-8") as f:
        f.write("\npsutil>=5.9\n")
    print("psutil added to requirements.txt")
else:
    print("psutil already in requirements.txt")
