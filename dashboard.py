import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime
from classifier import classify_prompt
from router import route_task
from logger import log_result
from logger import load_logs as base_load_logs

# --- Page Config ---
st.set_page_config(
    page_title="Edge AI Intent Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #5649d1;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    .intent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .route-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #6c5ce7;
    }
    h1 {
        color: #2d3436;
    }
    h2 {
        color: #6c5ce7;
    }
    h3 {
        color: #2d3436;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c5ce7;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Function to Load Logs with Caching ---
@st.cache_data(ttl=1)  # Auto-refresh every 1 second
def load_logs():
    return base_load_logs()

# --- Initialize Session State for Logs ---
if "logs_df" not in st.session_state:
    st.session_state.logs_df = load_logs()

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/6c5ce7/ffffff?text=Edge+AI", width=150)
    st.markdown("## Navigation")
    st.markdown("---")
    st.markdown("""
    ### About
    **Edge AI Intent Classifier** detects the intent of user prompts and routes them to the optimal compute environment:
    - 🟢 **ODA** (On-Device AI)
    - 🟡 **Hybrid** (Edge + Cloud)
    - 🔴 **Cloud LLM** (Full Cloud Model)
    """)
    st.markdown("---")
    st.markdown("### Model")
    st.info("**Model:** `typeform/distilbert-base-uncased-mnli`\n\n**Size:** ~268MB\n\n**Type:** Zero-shot NLI Classifier")

# --- Main App ---
st.title("🤖 Edge AI Intent Classifier Dashboard")
st.markdown("Classify user prompts, analyze intent matrices, and monitor routing decisions in real-time.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Classify", "📊 Analytics", "📜 Logs"])

# --- Tab 1: Classify ---
with tab1:
    st.header("Classify a Prompt")
    st.markdown("Enter a prompt to detect its intent matrix and see the routing decision.")

    # Input
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input("Enter your prompt:", placeholder="e.g., Summarize my emails and translate to French", key="prompt_input")
    with col2:
        classify_btn = st.button("Classify & Route", type="primary")

    if classify_btn and prompt:
        with st.spinner("Analyzing intent..."):
            result = classify_prompt(prompt)
            route = route_task(result["intent_matrix"], prompt)
            latency = result["latency_ms"]

            # Log the classification
            log_result(result, route)

            # Update session state logs and force refresh
            st.session_state.logs_df = load_logs()
            st.rerun()

            # --- Results ---
            st.markdown("---")
            st.subheader("🎯 Results")

            # Routing Decision Card
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### Routing Decision")
                if route == "ODA":
                    route_color = "#2ecc71"
                    route_icon = "🟢"
                    route_desc = "On-Device AI (Fast, Private, Offline)"
                elif route == "Hybrid":
                    route_color = "#f39c12"
                    route_icon = "🟡"
                    route_desc = "Edge + Cloud Split"
                else:
                    route_color = "#e74c3c"
                    route_icon = "🔴"
                    route_desc = "Full Cloud Model"

                st.markdown(f"""
                <div style="background: {route_color}; padding: 20px; border-radius: 10px; color: white;">
                    <h2 style="margin: 0;">{route_icon} {route}</h2>
                    <p style="margin: 5px 0 0 0;">{route_desc}</p>
                </div>
                """, unsafe_allow_html=True)

                # Reasoning
                if result["top_confidence"] < 0.55:
                    reasoning = f"Confidence is low ({result['top_confidence']*100:.1f}%) — the model is uncertain about intent. Routing to Cloud LLM for safer processing."
                elif result.get("is_multi_step", False):
                    reasoning = "Multi-step task detected. Routing to Hybrid for edge + cloud split."
                elif result["top_confidence"] >= 0.75:
                    reasoning = f"High confidence ({result['top_confidence']*100:.1f}%) — safe for {route}."
                else:
                    reasoning = f"Default routing to {route}."

                st.info(f"**Reasoning:** {reasoning}")

                # Metrics
                st.markdown(f"""
                <div class="metric-card">
                    <p><strong>Latency:</strong> {latency:.2f} ms</p>
                    <p><strong>Top Intent:</strong> {result['top_intent']} ({result['top_confidence']*100:.1f}%)</p>
                    <p><strong>Is Multi-Step?</strong> {'Yes' if result.get('is_multi_step', False) else 'No'}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### Intent Matrix")
                st.markdown("**Intent Confidence Scores**")

                # Intent Matrix Bar Chart
                df_intent = pd.DataFrame({
                    "Intent": list(result["intent_matrix"].keys()),
                    "Confidence": list(result["intent_matrix"].values())
                }).sort_values("Confidence", ascending=True)

                fig = px.bar(
                    df_intent,
                    x="Confidence",
                    y="Intent",
                    orientation="h",
                    color="Confidence",
                    color_continuous_scale="Viridis",
                    title="",
                    labels={"Confidence": "Confidence Score"}
                )
                fig.update_layout(
                    showlegend=False,
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis_title="",
                    yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Analytics ---
with tab2:
    st.header("📊 Analytics Dashboard")
    st.markdown("Monitor performance metrics, intent distributions, and routing trends.")

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🔄 Refresh Analytics"):
            st.session_state.logs_df = load_logs()
            st.rerun()

    # Load logs
    logs_df = st.session_state.logs_df
    if logs_df.empty:
        st.warning("No logs available yet. Classify some prompts first!")
    else:
        # --- Metrics Row ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Prompts", len(logs_df))
        with col2:
            st.metric("Avg Latency", f"{logs_df['latency_ms'].mean():.2f} ms")
        with col3:
            st.metric("ODA Routes", f"{len(logs_df[logs_df['route'] == 'ODA'])}")
        with col4:
            st.metric("Cloud Routes", f"{len(logs_df[logs_df['route'] == 'Cloud LLM'])}")

        st.markdown("---")

        # --- Charts Row ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Route Distribution")
            route_counts = logs_df["route"].value_counts()
            fig_pie = px.pie(
                values=route_counts.values,
                names=route_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Confidence Distribution")
            # Extract top confidence from logs (assuming it's stored as a string in intent_matrix)
            # For simplicity, let's assume logs_df has a 'confidence' column
            if "confidence" in logs_df.columns:
                fig_hist = px.histogram(
                    logs_df,
                    x="confidence",
                    nbins=20,
                    title="Top Intent Confidence Scores",
                    labels={"confidence": "Confidence"}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Confidence data not available in logs. Update your logger to include it.")

        st.markdown("---")

        # --- Latency Over Time ---
        st.subheader("Latency Over Time")
        logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])
        fig_line = px.line(
            logs_df,
            x="timestamp",
            y="latency_ms",
            color="route",
            title="Classification Latency Trends",
            labels={"latency_ms": "Latency (ms)", "timestamp": "Time"}
        )
        st.plotly_chart(fig_line, use_container_width=True)

# --- Tab 3: Logs ---
with tab3:
    st.header("📜 Classification Logs")
    st.markdown("View and filter all classification logs.")

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🔄 Refresh Logs"):
            st.session_state.logs_df = load_logs()
            st.rerun()

    # Load logs
    logs_df = st.session_state.logs_df
    if logs_df.empty:
        st.warning("No logs available yet. Classify some prompts first!")
    else:
        # --- Filters ---
        col1, col2, col3 = st.columns(3)
        with col1:
            route_filter = st.multiselect(
                "Filter by Route:",
                options=logs_df["route"].unique(),
                default=logs_df["route"].unique()
            )
        with col2:
            intent_filter = st.multiselect(
                "Filter by Top Intent:",
                options=logs_df["intent"].unique() if "intent" in logs_df.columns else [],
                default=logs_df["intent"].unique() if "intent" in logs_df.columns else []
            )
        with col3:
            date_filter = st.date_input("Filter by Date:", value=None)

        # Apply filters
        filtered_df = logs_df.copy()
        if route_filter:
            filtered_df = filtered_df[filtered_df["route"].isin(route_filter)]
        if intent_filter and "intent" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["intent"].isin(intent_filter)]
        if date_filter:
            filtered_df["date"] = pd.to_datetime(filtered_df["timestamp"]).dt.date
            filtered_df = filtered_df[filtered_df["date"] == date_filter]

        # Display logs
        st.dataframe(
            filtered_df.drop(columns=["date"], errors="ignore"),
            use_container_width=True,
            height=400
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Logs as CSV",
            data=csv,
            file_name="classification_logs.csv",
            mime="text/csv"
        )

# --- Debug Info ---
st.markdown("---")
st.markdown("### 🔍 Debug Info")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "logs.csv")
st.code(f"Log file path: {log_file}")
st.code(f"File exists: {os.path.isfile(log_file)}")
if os.path.isfile(log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        # Read only last 10 lines to save space
        lines = f.readlines()
        if len(lines) > 10:
            st.code("... showing last 10 lines ...\n" + "".join(lines[-10:]))
        else:
            st.code("".join(lines))

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p>Built with ❤️ using Streamlit | Model: <code>typeform/distilbert-base-uncased-mnli</code></p>
</div>
""", unsafe_allow_html=True)
