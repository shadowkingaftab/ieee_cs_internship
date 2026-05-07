"""
qwen_oda.py
-----------
On-Device AI module using Qwen2-0.5B-Instruct in GGUF format via llama-cpp-python.

Benefits over transformers:
  ✅ ~379MB model (Q4_K_M quantization) — truly lightweight
  ✅ Runs efficiently on CPU (no GPU required)
  ✅ Lazy-loaded — only downloaded/loaded when actually needed
  ✅ Single .gguf file — no auth token required

Model source (official Qwen repo, public access):
  https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF

First run: downloads the model automatically (~379MB).
Subsequent runs: loads from cache instantly.
"""

import os
import sys
import threading
from functools import lru_cache

# ── Model Config ──────────────────────────────────────────────────────────────
MODEL_FILENAME  = "qwen2-0_5b-instruct-q4_k_m.gguf"
MODEL_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH      = os.path.join(MODEL_DIR, MODEL_FILENAME)
MODEL_URL       = (
    "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/"
    + MODEL_FILENAME
)

# Global model instance (lazy-loaded)
_llm       = None
_llm_lock  = threading.Lock()
_load_error: str | None = None


# ── Auto-download helper ──────────────────────────────────────────────────────
def download_model(progress_callback=None) -> bool:
    """
    Downloads the GGUF model if it's not already on disk.

    Args:
        progress_callback: optional callable(downloaded_bytes, total_bytes)
    Returns:
        True on success, False on failure.
    """
    if os.path.exists(MODEL_PATH):
        return True

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"📥 Downloading Qwen GGUF model (~300MB) to {MODEL_PATH} …", file=sys.stderr)

    try:
        import requests
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        print("✅ Qwen GGUF model downloaded successfully!", file=sys.stderr)
        return True

    except Exception as e:
        print(f"❌ Failed to download model: {e}", file=sys.stderr)
        # Clean up partial download
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False


def is_model_downloaded() -> bool:
    """Returns True if the GGUF file exists on disk."""
    return os.path.exists(MODEL_PATH)


def get_model_size_mb() -> float:
    """Returns model file size in MB, or 0 if not downloaded."""
    if not os.path.exists(MODEL_PATH):
        return 0.0
    return os.path.getsize(MODEL_PATH) / (1024 * 1024)


# ── Lazy model loader ─────────────────────────────────────────────────────────
def _load_llm(n_gpu_layers: int = 0):
    """
    Thread-safe lazy loader for the Llama model.
    Set n_gpu_layers > 0 (e.g., 35) to offload layers to GPU for faster inference.
    """
    global _llm, _load_error

    with _llm_lock:
        if _llm is not None:
            return _llm

        # Ensure model is downloaded first
        if not is_model_downloaded():
            ok = download_model()
            if not ok:
                _load_error = "Model download failed. Check your internet connection."
                return None

        try:
            from llama_cpp import Llama

            print(f"Loading Qwen2-0.5B GGUF model from {MODEL_PATH} ...", file=sys.stderr)
            _llm = Llama(
                model_path    = MODEL_PATH,
                n_ctx         = 2048,        # Context window
                n_threads     = min(4, os.cpu_count() or 4),  # Use up to 4 CPU threads
                n_gpu_layers  = n_gpu_layers, # 0 = CPU only; >0 for GPU offload
                verbose       = False,
            )
            print("✅ Qwen GGUF model loaded successfully!", file=sys.stderr)
            _load_error = None
            return _llm

        except ImportError:
            _load_error = (
                "llama-cpp-python not installed. "
                "Run: pip install llama-cpp-python"
            )
            print(f"❌ {_load_error}", file=sys.stderr)
            return None
        except Exception as e:
            _load_error = str(e)
            print(f"❌ Failed to load Qwen model: {e}", file=sys.stderr)
            return None


# ── Cached inference ──────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def _cached_run(prompt: str, max_tokens: int, temperature: float) -> str:
    """LRU-cached wrapper so identical prompts skip re-inference."""
    llm = _load_llm()
    if llm is None:
        return f"❌ Qwen unavailable: {_load_error or 'unknown error'}"

    # Format prompt in Qwen2 ChatML style
    formatted = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        prompt      = formatted,
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 0.9,
        echo        = False,
        stop        = ["<|im_end|>", "<|im_start|>"],
    )

    text = output["choices"][0]["text"].strip()
    return text if text else "❌ Qwen returned an empty response."


# ── Public API ────────────────────────────────────────────────────────────────
def run_qwen(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Run Qwen-1.8B-Chat locally via llama-cpp-python.

    First call:
      - Downloads model if missing (~300MB)
      - Loads model into memory
    Subsequent calls:
      - Instant (model stays in memory, responses cached for identical prompts)

    Args:
        prompt        : User text prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature   : 0.0 = deterministic, 1.0 = creative.
    Returns:
        Generated text, or an error string prefixed with ❌.
    """
    if not prompt or not prompt.strip():
        return "❌ Empty prompt provided."

    try:
        return _cached_run(prompt.strip(), max_new_tokens, temperature)
    except Exception as e:
        return f"❌ Qwen inference error: {str(e)}"


def get_load_error() -> str | None:
    """Returns the last load error message, or None if healthy."""
    return _load_error


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Qwen GGUF Self-Test ===")
    print(f"Model path : {MODEL_PATH}")
    print(f"Downloaded : {is_model_downloaded()}")
    if not is_model_downloaded():
        print("Downloading model...")
        download_model()
    result = run_qwen("What is edge AI? Answer in one sentence.")
    print(f"Output: {result}")
