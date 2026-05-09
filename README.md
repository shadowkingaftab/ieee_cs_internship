# Edge AI Intent Classifier & Dashboard
### Production-Grade Hybrid Inference Engine

A robust intent classification system that intelligently routes tasks between local **Edge AI (Qwen2-0.5B)** and **Cloud AI (Grok API)**. Features a real-time Streamlit dashboard with performance analytics and a fine-tuning pipeline.

---

## 🚀 Key Features
- **Hybrid Execution**: Cloud-primary (Grok-3-mini) with seamless local fallback (Qwen2-0.5B GGUF).
- **Edge-Optimized**: Runs on CPU using `llama-cpp-python` with auto-hardware tuning.
- **Quantization Support**: Choose between `q2_k`, `q4_k_m`, and `q8_0` via config.
- **Production Analytics**: Track latency, confidence, and routing distributions.
- **Fine-Tuning Ready**: Built-in pipeline to improve classification accuracy >95%.
- **Agent Compatible**: Includes **MCP** (Model Context Protocol) and **Corsair** security layers.

---

## 🛠 Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and add your keys:
```bash
GROK_API_KEY=your-key-here
QWEN_QUANT=q4_k_m
```

### 3. Run the Dashboard (Streamlit)
```bash
streamlit run streamlit_app.py
```
*Note: dashboard.py was renamed to streamlit_app.py for Cloud deployment.*

### 4. Run the Legacy Gradio UI
```bash
python gradio_app.py
```

### 5. Run the MCP Server (For Agents)
```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

---

## 🐳 Docker Deployment
```bash
docker-compose up --build
```
Access the dashboard at `http://localhost:8501`.

---

## 🎯 Fine-Tuning the Classifier
To hit >95% confidence on your domain:
1. Collect samples in `training/domain_dataset.json`.
2. Run the trainer:
   ```bash
   python training/train_classifier.py --epochs 5 --eval
   ```
3. Set `CLASSIFIER_MODEL_PATH=models/fine_tuned_classifier` in `.env`.

---

## 📁 Project Structure
```
├── streamlit_app.py      # Main Streamlit Dashboard (Production)
├── qwen_oda.py           # Local AI Module (Qwen2-0.5B GGUF)
├── grok_cloud.py         # Cloud AI Module (Grok API)
├── classifier.py         # Intent Classifier (DistilBERT)
├── router.py             # Routing Logic & Fallback Orchestration
├── training/             # Fine-tuning scripts & datasets
├── .streamlit/           # Cloud deployment config
├── Dockerfile            # Containerization
└── mcp_server.py         # Agent Discovery Layer
```
