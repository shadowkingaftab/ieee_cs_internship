import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

from classifier import classify_prompt
from router import route_task, get_route_explanation
from logger import log_result, load_logs

st.set_page_config(page_title="Edge AI Intent Classifier", layout="wide")

st.title("🧠 Edge AI Intent Classifier Dashboard")

tab1, tab2, tab3 = st.tabs(["🎯 Classify", "📈 Analytics", "📁 Logs"])

with tab1:
    st.header("Classify a Prompt")
    st.markdown("Enter a prompt to detect its **intent matrix** and see the **routing decision**.")
    
    text = st.text_area("Enter your prompt:", "Summarize my emails from last week and translate them to French.", height=100)
    
    if st.button("Classify & Route", type="primary"):
        with st.spinner("Classifying..."):
            result = classify_prompt(text)
            intent_matrix = result["intent_matrix"]
            
            # Get routing decision
            route = route_task(intent_matrix, text)
            explanation = get_route_explanation(intent_matrix, route)
            
            # Log the result
            log_result(result, route)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Routing Decision")
                if route == "ODA":
                    st.success(f"🟢 **{route}**")
                elif route == "Hybrid":
                    st.warning(f"🟡 **{route}**")
                else:
                    st.error(f"🔴 **{route}**")
                st.info(explanation)
                
                st.write(f"**Latency:** {result['latency_ms']} ms")
                st.write(f"**Top Intent:** {result['top_intent']} ({result['top_confidence']*100:.1f}%)")
                st.write(f"**Is Multi-Step?** {'Yes' if result['is_multi_step'] else 'No'}")
                st.write(f"**Is Urgent?** {'Yes' if intent_matrix.get('urgent', 0) > 0.5 else 'No'}")
                
            with col2:
                st.subheader("Intent Matrix")
                # Visualize intent matrix
                df = pd.DataFrame({
                    "Intent": list(intent_matrix.keys()),
                    "Confidence": list(intent_matrix.values())
                }).sort_values(by="Confidence", ascending=True)
                
                fig = px.bar(df, x="Confidence", y="Intent", orientation='h', 
                             title="Intent Confidence Scores",
                             color="Confidence", color_continuous_scale="Viridis")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analytics Dashboard")
    logs = load_logs()
    
    if not logs.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Route Distribution
            route_counts = logs["route"].value_counts()
            fig_pie = px.pie(values=route_counts.values, names=route_counts.index, 
                             title="Route Distribution", color=route_counts.index,
                             color_discrete_map={"ODA": "green", "Hybrid": "gold", "Cloud LLM": "red"})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # Latency by Route
            fig_box = px.box(logs, x="route", y="latency_ms", title="Latency by Route",
                             color="route", color_discrete_map={"ODA": "green", "Hybrid": "gold", "Cloud LLM": "red"})
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Top Intents
        st.subheader("Intent Distribution")
        intent_counts = logs["intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig_bar = px.bar(intent_counts, x="Intent", y="Count", title="Most Common Top Intents")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("No logs available yet. Go to the Classify tab to process some prompts!")

with tab3:
    st.header("Classification Logs")
    logs = load_logs()
    
    if not logs.empty:
        # Add a filter for routes
        selected_route = st.selectbox("Filter by Route", ["All", "ODA", "Hybrid", "Cloud LLM"])
        
        display_logs = logs if selected_route == "All" else logs[logs["route"] == selected_route]
        
        st.dataframe(display_logs, use_container_width=True, hide_index=True)
        
        # Download button
        csv = display_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='classifier_logs.csv',
            mime='text/csv',
        )
    else:
        st.info("No logs available yet.")
