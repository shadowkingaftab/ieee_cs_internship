"""
grok_cloud.py
-------------
Cloud LLM module using the Grok API (xAI) — PRIMARY inference engine.

This is now the PRIMARY model. Qwen (local) is the fallback.

Set API key before running:
  Windows PowerShell:  $env:GROK_API_KEY = "your-key-here"
  Linux/Mac:           export GROK_API_KEY="your-key-here"
  Or place it in your .env file as: GROK_API_KEY=your-key-here
"""

import requests
import os
import sys
from dotenv import load_dotenv

load_dotenv()

GROK_API_URL    = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL   = "grok-3-mini"   # grok-3-mini is fast & cheap; use "grok-3" for max quality
DEFAULT_TIMEOUT = 30              # seconds


def is_api_key_set() -> bool:
    """Returns True if a Grok API key is available in the environment."""
    return bool(os.getenv("GROK_API_KEY", "").strip())


def run_grok(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """
    Call the Grok API (PRIMARY inference path).

    Args:
        prompt      : The user's prompt text.
        model       : Grok model ID. Options: "grok-3-mini", "grok-3", "grok-1".
        max_tokens  : Maximum tokens in the response.
        temperature : 0.0 = deterministic, 1.0 = creative.
        timeout     : Request timeout in seconds.

    Returns:
        Generated text on success, or an error string prefixed with ❌/⚠️.
    """
    api_key = os.getenv("GROK_API_KEY", "").strip()

    if not api_key:
        return (
            "⚠️ GROK_API_KEY not set.\n"
            "• Set it in your .env file: GROK_API_KEY=your-key-here\n"
            "• Or in PowerShell: $env:GROK_API_KEY = 'your-key-here'\n"
            "• Or paste it in the Dashboard sidebar."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            GROK_API_URL, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.strip() if content else "❌ Grok returned an empty response."

    except requests.exceptions.Timeout:
        return f"❌ Grok API timed out after {timeout}s. Try again or use local Qwen."

    except requests.exceptions.HTTPError as e:
        code = response.status_code
        if code == 401:
            return "Grok API: Invalid API key (401). Check your GROK_API_KEY."
        if code == 403:
            body = response.json() if response.content else {}
            detail = body.get("error", "")
            return f"Grok API: Access denied (403). {detail} Visit https://console.x.ai to add credits."
        if code == 429:
            return "Grok API: Rate limit exceeded (429). Wait a moment and retry."
        if code == 402:
            return "Grok API: Quota exceeded (402). Check your xAI billing at https://console.x.ai."
        return f"Grok API HTTP error {code}: {e}"

    except requests.exceptions.ConnectionError:
        return "❌ Grok API: No internet connection. Local Qwen will handle this."

    except requests.exceptions.RequestException as e:
        return f"❌ Grok API connection error: {str(e)}"

    except (KeyError, IndexError):
        return "❌ Grok API: Unexpected response format."


def run_grok_with_fallback(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> tuple[str, str]:
    """
    Try Grok first. If it fails (no key, timeout, error), fall back to local Qwen.

    Returns:
        (response_text, source) where source is "grok" or "qwen".
    """
    from qwen_oda import run_qwen, is_model_downloaded

    grok_resp = run_grok(prompt, model=model, max_tokens=max_tokens, temperature=temperature)

    # If Grok returned cleanly (no error prefix), use it
    if not grok_resp.startswith(("Grok API", "No capacity", "timed out", "connection", "Unexpected")):
        return grok_resp, "grok"

    # Grok failed — fall back to local Qwen
    print(f"[router] Grok unavailable: {grok_resp[:80]}", flush=True)

    if not is_model_downloaded():
        return (
            f"Grok is currently unavailable: {grok_resp}\n\n"
            "Local Qwen fallback: model not downloaded yet.\n"
            "Click 'Download Qwen GGUF' in the sidebar to enable offline inference.",
            "none",
        )

    qwen_resp = run_qwen(prompt, max_new_tokens=max_tokens, temperature=temperature)
    return qwen_resp, "qwen"


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_prompt = "What is edge AI? Answer in one sentence."
    resp, source = run_grok_with_fallback(test_prompt)
    print(f"Source : {source}")
    print(f"Output : {resp.encode('ascii', 'ignore').decode('ascii')}")
