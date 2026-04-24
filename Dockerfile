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
RUN mkdir -p {vector_store,outputs,logs,models,data} && \
    mkdir -p logs/{traces,enrollments} && \
    mkdir -p outputs/{generated,ablation_samples} && \
    mkdir -p data/eval_dataset

# Install MIVA package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from miva.pipeline import MIVAPipeline; p = MIVAPipeline(); exit(0 if p.health_check() else 1)"

# Expose Prometheus metrics port
EXPOSE 8000

# Default command
CMD ["miva", "health-check"]
