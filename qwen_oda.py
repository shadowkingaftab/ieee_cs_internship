"""
qwen_oda.py
-----------
On-Device AI module using Qwen2-0.5B-Instruct in GGUF format via llama-cpp-python.

Quantization levels (set QWEN_QUANT in .env):
  q2_k   → 323 MB — fastest, lowest quality  (very low-end hardware)
  q4_k_m → 379 MB — balanced   ← DEFAULT
  q8_0   → 506 MB — highest quality CPU inference

Hardware auto-tuning:
  n_threads → auto-detected from CPU core count
  n_batch   → configurable for throughput vs latency tradeoff
  n_gpu_layers → set QWEN_GPU_LAYERS in .env (0 = CPU-only, -1 = full GPU)

Model source: https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF
"""

import os
import sys
import threading
import platform
import psutil
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# Helper to get config from Streamlit Secrets or Environment
def get_config(key, default=None):
    # 1. Try Streamlit Secrets (Cloud)
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except (ImportError, Exception):
        pass
    
    # 2. Try Environment (Local / Docker)
    return os.getenv(key, default)

# ── Quantization config (from .env or defaults) ───────────────────────────────
_QUANT_MAP = {
    "q2_k":   ("qwen2-0_5b-instruct-q2_k.gguf",   323),
    "q4_0":   ("qwen2-0_5b-instruct-q4_0.gguf",   357),
    "q4_k_m": ("qwen2-0_5b-instruct-q4_k_m.gguf", 379),
    "q5_k_m": ("qwen2-0_5b-instruct-q5_k_m.gguf", 444),
    "q8_0":   ("qwen2-0_5b-instruct-q8_0.gguf",   506),
}
QUANT_LEVEL    = get_config("QWEN_QUANT", "q4_k_m").lower()
if QUANT_LEVEL not in _QUANT_MAP:
    QUANT_LEVEL = "q4_k_m"

MODEL_FILENAME, MODEL_SIZE_MB = _QUANT_MAP[QUANT_LEVEL]
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL  = (
    "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/"
    + MODEL_FILENAME
)

# ── Hardware auto-tuning ──────────────────────────────────────────────────────
def _get_optimal_threads() -> int:
    """Use physical CPU cores, capped at 8 to avoid thrashing."""
    try:
        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except Exception:
        cores = os.cpu_count() or 4
    return max(1, min(cores, 8))

def _get_optimal_batch() -> int:
    """
    Set n_batch based on available RAM.
    More RAM → larger batch → faster throughput.
    """
    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram_gb = 4
    if ram_gb >= 16:
        return 512
    elif ram_gb >= 8:
        return 256
    else:
        return 128   # safe for 4GB RAM machines

N_THREADS    = int(get_config("QWEN_THREADS",    str(_get_optimal_threads())))
N_BATCH      = int(get_config("QWEN_BATCH",      str(_get_optimal_batch())))
N_GPU_LAYERS = int(get_config("QWEN_GPU_LAYERS", "0"))   # 0=CPU, -1=full GPU
N_CTX        = int(get_config("QWEN_CTX",        "2048"))

# ── Globals ───────────────────────────────────────────────────────────────────
_llm        = None
_llm_lock   = threading.Lock()
_load_error: str | None = None


# ── Hardware info (for dashboard display) ─────────────────────────────────────
def get_hardware_info() -> dict:
    """Return hardware summary for display in the dashboard sidebar."""
    try:
        ram_gb   = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        cpu_phys = psutil.cpu_count(logical=False) or "?"
        cpu_log  = psutil.cpu_count(logical=True)  or "?"
    except Exception:
        ram_gb, cpu_phys, cpu_log = "?", "?", "?"
    return {
        "os":          platform.system(),
        "cpu_physical": cpu_phys,
        "cpu_logical":  cpu_log,
        "ram_gb":       ram_gb,
        "n_threads":    N_THREADS,
        "n_batch":      N_BATCH,
        "n_gpu_layers": N_GPU_LAYERS,
        "quant_level":  QUANT_LEVEL,
        "model_size_mb": MODEL_SIZE_MB,
    }


def get_quant_options() -> dict:
    """Return the full quantization map for the dashboard selector."""
    return _QUANT_MAP


# ── Auto-download helper ──────────────────────────────────────────────────────
def download_model(progress_callback=None) -> bool:
    """
    Downloads the GGUF model if it's not already on disk.
    Supports progress callback: callable(downloaded_bytes, total_bytes).
    """
    if os.path.exists(MODEL_PATH):
        return True

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading Qwen2 GGUF [{QUANT_LEVEL}] (~{MODEL_SIZE_MB}MB) to {MODEL_PATH}",
          file=sys.stderr)

    try:
        import requests
        resp = requests.get(MODEL_URL, stream=True, timeout=300)
        resp.raise_for_status()

        total      = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2MB chunks for faster download

        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        print("Qwen GGUF model downloaded successfully!", file=sys.stderr)
        return True

    except Exception as e:
        print(f"Failed to download model: {e}", file=sys.stderr)
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


def is_model_downloaded() -> bool:
    return os.path.exists(MODEL_PATH)


def get_model_size_mb() -> float:
    if not os.path.exists(MODEL_PATH):
        return 0.0
    return os.path.getsize(MODEL_PATH) / (1024 * 1024)


# ── Lazy model loader with hardware tuning ────────────────────────────────────
def _load_llm():
    """Thread-safe lazy loader with auto-tuned hardware parameters."""
    global _llm, _load_error

    with _llm_lock:
        if _llm is not None:
            return _llm

        if not is_model_downloaded():
            ok = download_model()
            if not ok:
                _load_error = "Model download failed. Check your internet connection."
                return None

        try:
            from llama_cpp import Llama

            print(
                f"Loading Qwen2-0.5B [{QUANT_LEVEL}] | "
                f"threads={N_THREADS} batch={N_BATCH} gpu_layers={N_GPU_LAYERS} ctx={N_CTX}",
                file=sys.stderr,
            )
            _llm = Llama(
                model_path   = MODEL_PATH,
                n_ctx        = N_CTX,
                n_threads    = N_THREADS,
                n_batch      = N_BATCH,
                n_gpu_layers = N_GPU_LAYERS,
                verbose      = False,
            )
            print("Qwen2-0.5B GGUF loaded successfully!", file=sys.stderr)
            _load_error = None
            return _llm

        except ImportError:
            _load_error = "llama-cpp-python not installed. Run: pip install llama-cpp-python"
            print(_load_error, file=sys.stderr)
            return None
        except Exception as e:
            _load_error = str(e)
            print(f"Failed to load Qwen model: {e}", file=sys.stderr)
            return None


# ── Cached inference ──────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _cached_run(prompt: str, max_tokens: int, temperature: float) -> str:
    """LRU-cached inference — identical prompts skip re-computation."""
    llm = _load_llm()
    if llm is None:
        return f"Qwen unavailable: {_load_error or 'unknown error'}"

    # Qwen2 ChatML prompt format
    formatted = (
        "<|im_start|>system\n"
        "You are a concise, helpful AI assistant. Answer clearly and briefly.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    output = llm(
        prompt      = formatted,
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 0.9,
        repeat_penalty = 1.1,   # reduces repetition on low-quant models
        echo        = False,
        stop        = ["<|im_end|>", "<|im_start|>"],
    )

    text = output["choices"][0]["text"].strip()
    return text if text else "Qwen returned an empty response."


# ── Public API ────────────────────────────────────────────────────────────────
def run_qwen(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Run Qwen2-0.5B-Instruct locally via llama-cpp-python.

    Quantization level is controlled by QWEN_QUANT in .env:
      q2_k (323MB, fastest) | q4_k_m (379MB, default) | q8_0 (506MB, best quality)

    Args:
        prompt        : User prompt text.
        max_new_tokens: Max tokens to generate (default 256).
        temperature   : 0.0 = deterministic, 1.0 = creative (default 0.7).
    Returns:
        Generated text string, or error description.
    """
    if not prompt or not prompt.strip():
        return "Empty prompt provided."
    try:
        return _cached_run(prompt.strip(), max_new_tokens, temperature)
    except Exception as e:
        return f"Qwen inference error: {str(e)}"


def get_load_error() -> str | None:
    return _load_error


def reload_model():
    """Force a model reload (e.g., after changing QWEN_QUANT)."""
    global _llm, _load_error
    with _llm_lock:
        _llm = None
        _load_error = None
    _cached_run.cache_clear()


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hw = get_hardware_info()
    print("=== Qwen2-0.5B GGUF Self-Test ===")
    print(f"Quant   : {hw['quant_level']} (~{hw['model_size_mb']} MB)")
    print(f"Threads : {hw['n_threads']}  Batch: {hw['n_batch']}  GPU layers: {hw['n_gpu_layers']}")
    print(f"RAM     : {hw['ram_gb']} GB  |  CPU cores: {hw['cpu_physical']} physical")
    print(f"Model   : {MODEL_PATH}")
    print(f"Ready   : {is_model_downloaded()}")
    if not is_model_downloaded():
        download_model()
    result = run_qwen("What is edge AI? Answer in one sentence.")
    print(f"Output  : {result}")
