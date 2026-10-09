# MIVA Studio Docker Image
# Supports GPU with NVIDIA CUDA

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p vector_store outputs logs models data \
    logs/traces logs/enrollments \
    outputs/generated outputs/ablation_samples \
    data/eval_dataset

# Install MIVA package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:${PORT:-8080}/health || exit 1

# Expose Prometheus metrics port
EXPOSE 8080

# Default command
CMD ["sh", "-c", "exec uvicorn miva.api.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
