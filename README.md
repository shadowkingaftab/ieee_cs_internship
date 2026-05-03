# Edge AI Intent Classifier
### IEEE CS Internship — Phase 1 & 2 (Agent Integration)

A text classification system that detects the **intent** of any user prompt and routes it to the appropriate compute environment. This project is now a **first-class citizen in the agent ecosystem**, supporting both **MCP** (Model Context Protocol) and **Corsair** (Security/Permission layer).

---

## 🚀 Hybrid Architecture
Your module now supports three distinct layers of interaction:
1. **Web Dashboard**: Local Gradio UI for manual testing.
2. **MCP Server**: Allows AI agents (like Claude or ChatGPT) to discover and call your classifier as a tool.
3. **Corsair Plugin**: Adds a security/permission layer that intercepts agent calls and requests human approval for risky routes (like Cloud LLM).

---

## Project Structure

```
ieee_cs_internship/
├── classifier.py              # NLP model — intent detection + confidence score
├── router.py                  # Routing logic — ODA / Hybrid / Cloud LLM
├── logger.py                  # CSV logging — saves every result to logs.csv
├── evaluator.py               # Metrics — confidence, latency, distribution stats
├── app.py                     # Gradio UI — real-time testing dashboard
├── mcp_server.py              # NEW: MCP Server — Agent discovery layer
├── corsair-plugin-edge-ai/    # NEW: Corsair Plugin — Permission/Security layer
│   └── index.ts               # Plugin logic & approval rules
├── requirements.txt           # Python dependencies
├── logs.csv                   # Auto-created when you run the app
└── .gitignore                 # Excludes cache and node_modules
```

---

## 🛠 Setup & Usage

### 1. Python Environment
```bash
# Install dependencies
pip install -r requirements.txt
pip install mcp  # For the MCP server
```

### 2. Run the Manual Dashboard
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) to test manually.

### 3. Run the MCP Server (For Agents)
To test the agent-ready tool using the MCP Inspector:
```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

---

## 🔒 Corsair Permission Layer
The included Corsair plugin (`corsair-plugin-edge-ai`) implements the following security rules:
- **Classify**: Set to `open` (safe to run automatically).
- **Route**: Set to `cautious`.
- **Approval Rule**: If the routing logic recommends **Cloud LLM**, Corsair will pause and request human approval before allowing the agent to proceed.

---

## 🧠 Model & Routing
- **Model**: `typeform/distilbert-base-uncased-mnli` (Zero-shot NLI)
- **Routing Logic**:
    - Confidence < 55% → **Cloud LLM**
    - Multi-step task → **Hybrid**
    - High Confidence (≥ 75%) + Simple → **ODA**
    - High Confidence (≥ 75%) + Complex → **Cloud LLM**
    - Everything else → **Hybrid**
