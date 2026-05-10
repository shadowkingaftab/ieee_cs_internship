# Edge AI Intent Classifier & Dashboard: Comprehensive Implementation Report

## 1. Project Overview
The **Edge AI Intent Classifier** is a production-grade, hybrid intelligent routing system. It is designed to intercept user prompts, classify their underlying "intent" and complexity in real-time, and dynamically route the execution to either a fast, private **Local Edge Model** or a powerful **Cloud LLM**.

The system prevents over-reliance on cloud APIs by executing simple or privacy-sensitive tasks locally, while gracefully falling back to the cloud for complex, urgent, or multi-step reasoning tasks. It includes a comprehensive Streamlit dashboard for real-time visualization of routing decisions, performance metrics, and system health.

---

## 2. System Architecture & Workflow
The architecture operates in a multi-stage pipeline:

1. **Input Reception**: A user query is received via the Streamlit UI or the Model Context Protocol (MCP) server.
2. **Intent Classification**: The query is passed to `classifier.py`, which uses a zero-shot transformer model to generate an **Intent Matrix** (a probability distribution over 8 predefined intents).
3. **Decision Engine (Router)**: `router.py` analyzes the Intent Matrix and applies deterministic thresholds to choose the optimal execution path (ODA, Cloud, or Hybrid).
4. **Execution Engine**:
   - **ODA (On-Device AI)**: Handled by `qwen_oda.py` using a local GGUF model.
   - **Cloud LLM**: Handled by `grok_cloud.py` using the Grok API, with automatic local fallback.
   - **Hybrid**: Sequentially executes lightweight steps locally and complex steps in the cloud.
5. **Response & Analytics**: The final response is returned to the user, and the entire transaction (latency, confidence, route) is logged to `logs.csv` for dashboard analytics.

---

## 3. Technology Stack

| Category | Technology / Library | Description & Purpose |
| :--- | :--- | :--- |
| **Backend & Orchestration** | Python 3.10+ | Core logic and system integration. |
| **Intent Classification** | `transformers`, `torch` | Hugging Face pipeline for zero-shot text classification. |
| **Edge AI Engine** | `llama-cpp-python` | CPU-optimized inference for GGUF local models. |
| **Edge AI Model** | Qwen2-0.5B (GGUF) | Lightweight, 500M parameter model (`q4_k_m` quantization). Fits in ~300MB RAM. |
| **Cloud AI Engine** | xAI Grok API (`requests`) | High-reasoning model (`grok-3-mini`) for complex tasks. |
| **Dashboard UI** | `streamlit`, `plotly`, `pandas` | Real-time web UI, metric rendering, and CSV log parsing. |
| **Agent Protocol** | Model Context Protocol (MCP) | Allows external agents to interact with the classifier via `mcp_server.py`. |
| **Environment Control** | `python-dotenv` | Secure API key and config management (`.env`). |

---

## 4. Deep Dive: Component Modules

### 4.1. `classifier.py` (The NLP Brain)
- **Model Used**: `typeform/distilbert-base-uncased-mnli` (~268MB footprint).
- **Strategy**: Zero-shot classification.
- **Labels Tracked**: `summarize`, `translate`, `analyze_data`, `generate_code`, `multi_step`, `ambiguous`, `urgent`, `simple_query`.
- **Post-Processing**: Applies heuristics (e.g., boosting `multi_step` confidence if "and" or "then" is detected in the prompt).
- **Output**: Returns a normalized Intent Matrix, top intent, confidence score, and latency.

### 4.2. `router.py` (The Decision Maker)
Applies a strict ruleset based on the classifier's output:
- **Urgent Task** (`urgent` > 0.5) → Routes to **Cloud LLM** for immediate, reliable processing.
- **Multi-Step Task** (`multi_step` > 0.3) → Routes to **Hybrid** (splits the prompt into Edge and Cloud segments).
- **Low Confidence** (`< 0.55`) → Routes to **Cloud LLM** (Grok provides safer, deeper reasoning).
- **High Confidence & Simple** (`>= 0.75` for translate, summarize, simple_query) → Routes to **ODA** (Local Edge).
- **High Confidence & Complex** (`>= 0.75` for analyze_data, generate_code) → Routes to **Cloud LLM**.

### 4.3. `qwen_oda.py` (Edge Execution)
- Initializes the `Llama` class from `llama_cpp`.
- Operates entirely offline. If initialized correctly, latency is virtually zero (excluding generation time).

### 4.4. `grok_cloud.py` (Cloud Execution)
- Sends payloads to `https://api.x.ai/v1/chat/completions`.
- **Critical Feature**: Implements `run_grok_with_fallback`. If the API key is missing, network fails, or Grok times out, the script transparently reroutes the payload to `qwen_oda.py`.

### 4.5. `streamlit_app.py` (Production Dashboard)
- **Playground**: Interactive interface for testing queries.
- **Analytics View**: Parses `logs.csv` to generate:
  - Total queries processed.
  - Average latency (ms).
  - A Plotly Pie Chart showing the Edge vs. Cloud routing distribution.
  - A confidence distribution gauge.
- **System Health**: Displays the status of the local LLM engine and Grok API connection.

### 4.6. `mcp_server.py` (Agent Discovery Layer)
- Exposes the classifier's capabilities to other AI systems using the Model Context Protocol.
- Defines a tool `classify_intent` that external agents can invoke to get routing recommendations.

---

## 5. Recent Optimizations & Deployment Triumphs

1. **Streamlit Cloud Compilation Fix**: 
   - Addressed Streamlit Cloud `ModuleNotFoundError` and Out-Of-Memory (OOM) crashes by strictly defining dependencies.
   - Optimized `requirements.txt` to use the CPU-only version of PyTorch (`--extra-index-url https://download.pytorch.org/whl/cpu`), drastically reducing container size and build time.
2. **Robust Fallback Engine**: 
   - Engineered the system to degrade gracefully. If Streamlit Cloud blocks the C++ compilation of `llama-cpp-python`, `router.py` automatically detects the missing module and routes everything to Grok without crashing the app.
3. **Hybrid Parallelism**: 
   - Implemented sequential prompting for Hybrid routes, ensuring multi-step tasks get the benefit of both edge privacy/speed and cloud reasoning.

---

## 6. Future Roadmap

- **Domain-Specific Fine-Tuning**: Utilizing the `training/` pipeline to fine-tune the DistilBERT model on specific enterprise data to push baseline confidence above 95%.
- **RLHF (Reinforcement Learning from Human Feedback)**: Adding feedback mechanisms to the Streamlit UI to dynamically adjust routing thresholds over time.
- **Hardware Acceleration**: Integrating CUDA/Metal detection for `llama-cpp-python` to push Edge generation speeds higher on capable hardware.
