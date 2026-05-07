# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Edge AI Intent Classifier Dashboard
# Multi-stage build: keeps final image lean (~600MB vs ~3GB naive)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed for llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all Python deps into /install (isolated from system)
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt


# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="IEEE CS Internship — Edge AI Classifier"
LABEL description="Streamlit dashboard for Edge AI Intent Classifier (Qwen2-0.5B + Grok)"

# Runtime deps only (no gcc/cmake needed at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY classifier.py router.py logger.py evaluator.py ./
COPY qwen_oda.py grok_cloud.py mcp_server.py ./
COPY dashboard.py app.py ./
COPY training/ ./training/
COPY .env.example .env.example

# Create directories
RUN mkdir -p /app/models /app/logs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Environment defaults (override with docker-compose or -e flags)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV QWEN_QUANT=q4_k_m
ENV QWEN_THREADS=4
ENV QWEN_GPU_LAYERS=0

# Run the dashboard
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
