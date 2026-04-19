import streamlit as st
import time
from engine import process_prompt

st.set_page_config(page_title="SynapseFlow", page_icon="⬡", layout="centered")

# Load CSS
with open("style.css", encoding='utf-8') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject Animated Background SVGs
st.markdown("""
<div class="wave-container">
    <svg class="wave w1" viewBox="0 0 2000 1000" preserveAspectRatio="none">
        <path d="M0,500 C300,300 500,700 1000,500 C1500,300 1700,700 2000,500" />
    </svg>
    <svg class="wave w2" viewBox="0 0 2000 1000" preserveAspectRatio="none">
        <path d="M0,400 C400,700 600,200 1000,500 C1400,800 1600,300 2000,400" />
    </svg>
    <svg class="wave w3" viewBox="0 0 2000 1000" preserveAspectRatio="none">
        <path d="M0,600 C250,400 750,400 1000,600 C1250,800 1750,800 2000,600" />
    </svg>
</div>
""", unsafe_allow_html=True)

# State
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "results" not in st.session_state:
    st.session_state.results = []

def process_submission():
    if st.session_state.input_prompt.strip():
        st.session_state.user_prompt = st.session_state.input_prompt
        st.session_state.has_run = True
        st.session_state.results = []

# --- Brand Header ---
st.markdown("""
<div class='brand-container'>
    <div class='brand-icon'>⬡</div>
    <h1 class='brand-title'>SynapseFlow</h1>
    <div class='brand-subtitle'>INTELLIGENT COMPUTE ORCHESTRATION</div>
</div>
""", unsafe_allow_html=True)

# --- Full Width Search/Input UI ---
st.text_input(
    "Orchestration Prompt", 
    placeholder="Search tasks, commands, or insights...", 
    label_visibility="collapsed",
    key="input_prompt",
    on_change=process_submission
)

if st.session_state.has_run and st.session_state.user_prompt:
    
    if not st.session_state.results:
        # Micro-text analyzing indicator
        status_box = st.empty()
        status_box.markdown("""
        <div class='status-indicator'>
            <div class='pulse-circle'></div> ROUTING COMPUTATIONAL NODES...
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.6)
        
        st.session_state.results = process_prompt(st.session_state.user_prompt)
        status_box.empty()

    # --- Output Cards ---
    st.markdown("<div class='step-list'>", unsafe_allow_html=True)
    
    for i, (step, route) in enumerate(st.session_state.results, 1):
        if "LLM" in route:
            b_class = "badge-llm"
            b_icon = "👁️"
        elif "device" in route:
            b_class = "badge-device"
            b_icon = "🖧"
        else:
            b_class = "badge-hybrid"
            b_icon = "🖧"
            
        display_route = route.upper().replace("CLOUD LLM", "LLM").replace("ON-DEVICE AI", "ON-DEVICE")
        
        card_html = f"""
        <div class='horizontal-card' style='animation-delay: {i * 0.12}s;'>
            <div class='card-num'>{i:02d}</div>
            <div class='card-body'>{step}</div>
            <div class='badge-container'>
                <div class='pill-badge {b_class}'>
                    <span class='badge-icon'>{b_icon}</span> {display_route}
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        time.sleep(0.04)

    st.markdown("</div>", unsafe_allow_html=True)
