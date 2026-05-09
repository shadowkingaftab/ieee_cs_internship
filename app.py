import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# --- Config ---
REPO_ID = "Qwen/Qwen2-0.5B-Instruct-GGUF"
FILENAME = "qwen2-0_5b-instruct-q4_k_m.gguf"

@st.cache_resource
def load_llm():
    """Safely downloads and loads the GGUF model using HuggingFace Hub."""
    # This automatically handles downloading, resuming, and caching the model securely
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    return Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window
        n_threads=4,  # CPU threads
        n_gpu_layers=0  # CPU-only
    )

# --- Lazy-load Qwen (ONLY WHEN NEEDED) ---
if "llm" not in st.session_state:
    with st.spinner("Downloading & Loading Qwen (379MB, first run only)..."):
        try:
            st.session_state.llm = load_llm()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

def get_response(prompt):
    output = st.session_state.llm(
        prompt=prompt,
        max_tokens=256,
        temperature=0.7,
        echo=False
    )
    return output["choices"][0]["text"].strip()

# --- UI ---
st.title("⚡ Qwen Offline Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask me anything (offline)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.spinner("Thinking..."):
        response = get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
