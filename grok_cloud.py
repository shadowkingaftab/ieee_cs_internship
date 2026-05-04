"""
grok_cloud.py
-------------
Cloud LLM module using the Grok API (xAI).
Requires a valid GROK_API_KEY environment variable.

Set it before running:
  Windows PowerShell:  $env:GROK_API_KEY = "your-key-here"
  Linux/Mac:           export GROK_API_KEY="your-key-here"
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def run_grok(prompt: str, model: str = "grok-3-mini", max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Call the Grok API for Cloud LLM inference.

    Parameters:
      prompt      : The user's prompt.
      model       : Grok model to use (e.g. "grok-3-mini", "grok-3").
      max_tokens  : Maximum tokens to generate.
      temperature : Creativity factor.

    Returns:
      Generated text response, or an error message string.
    """
    api_key = os.getenv("GROK_API_KEY")

    if not api_key:
        return (
            "⚠️ GROK_API_KEY not set. "
            "Run: $env:GROK_API_KEY = 'your-key-here' in PowerShell, "
            "then restart Streamlit."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "❌ Grok API timed out after 30 seconds."
    except requests.exceptions.HTTPError as e:
        return f"❌ Grok API HTTP error {response.status_code}: {e}"
    except requests.exceptions.RequestException as e:
        return f"❌ Grok API connection error: {str(e)}"
    except (KeyError, IndexError):
        return "❌ Unexpected response format from Grok API."


if __name__ == "__main__":
    test = run_grok("What is edge AI? Answer in one sentence.")
    print(f"Grok output: {test}")
