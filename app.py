import streamlit as st
from llama_cpp import Llama
import os
import requests

# --- Config ---
# Qwen2-0.5B is ~379MB and fits perfectly on Streamlit Cloud's 1GB RAM limit.
# The 1.8B model is 1.13GB and would crash the cloud instance with Out Of Memory errors.
QWEN_MODEL_URL = "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf"
QWEN_MODEL_PATH = "qwen2-0_5b-instruct-q4_k_m.gguf"

@st.cache_resource
def download_model():
    """Downloads the GGUF model if it doesn't exist, avoiding GitHub's 100MB file limit."""
    if not os.path.exists(QWEN_MODEL_PATH):
        response = requests.get(QWEN_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(QWEN_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return True

# --- Lazy-load Qwen (ONLY WHEN NEEDED) ---
if "llm" not in st.session_state:
    with st.spinner("Loading Qwen (379MB, downloading on first run)..."):
        download_model()
        st.session_state.llm = Llama(
            model_path=QWEN_MODEL_PATH,
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=0  # CPU-only (Streamlit Cloud has no GPU)
        )

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
