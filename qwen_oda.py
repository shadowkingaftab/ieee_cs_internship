"""
qwen_oda.py
-----------
On-Device AI module using Qwen2-0.8B.
Runs entirely locally — no internet required.
~1.6GB model, works on CPU (slow) or GPU (fast).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

MODEL_NAME = "Qwen/Qwen2-0.8B"

print("Loading Qwen2-0.8B (On-Device AI)...", file=sys.stderr)
print("First run will download ~1.6GB. Subsequent runs load from cache.", file=sys.stderr)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    device_map="auto"           # Auto-detects GPU if available, falls back to CPU
).eval()

print("Qwen2-0.8B loaded successfully!", file=sys.stderr)


def run_qwen(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Run Qwen2-0.8B locally.

    Parameters:
      prompt         : The user's prompt to respond to.
      max_new_tokens : Maximum number of tokens to generate.
      temperature    : Creativity factor (0.0 = deterministic, 1.0 = creative).

    Returns:
      Generated text response as a string.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Only return the newly generated tokens (not the input)
    new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    test = run_qwen("What is edge AI? Answer in one sentence.")
    print(f"Qwen output: {test}")
