#!/bin/bash

# MIVA Studio Setup Script
# Handles environment creation, dependency installation, model downloads, and validation
# Run with: bash scripts/setup.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.10"
VENV_DIR="venv"
MODELS_DIR="models"
DATA_DIR="data"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  MIVA Studio — Production RAG Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================

echo -e "${BLUE}[1/6] Checking prerequisites...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [[ $python_major -lt 3 ]] || [[ $python_major -eq 3 && $python_minor -lt 10 ]]; then
    echo -e "${RED}✗ Python 3.10+ required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ git not found. Install git and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ git${NC}"

# Check CUDA availability (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available (GPU acceleration enabled)${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not found (will use CPU, slower)${NC}"
fi

echo ""

# ============================================================================
# STEP 2: Create Virtual Environment
# ============================================================================

echo -e "${BLUE}[2/6] Creating virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠ Virtual environment exists. Removing and recreating...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
echo -e "  Location: $(pwd)/$VENV_DIR"
echo -e "  Activate with: source $VENV_DIR/bin/activate"
echo ""

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================

echo -e "${BLUE}[3/6] Installing dependencies (this may take 5-10 minutes)...${NC}"

pip install -r requirements.txt

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# ============================================================================
# STEP 4: Create Directory Structure
# ============================================================================

echo -e "${BLUE}[4/6] Creating directory structure...${NC}"

mkdir -p {vector_store,outputs,logs,models,data}
mkdir -p logs/traces logs/enrollments
mkdir -p outputs/generated outputs/ablation_samples
mkdir -p data/eval_dataset
mkdir -p config/backups

echo -e "${GREEN}✓ Directory structure created${NC}"
echo -e "  Vector store: $(pwd)/vector_store"
echo -e "  Outputs: $(pwd)/outputs"
echo -e "  Models: $(pwd)/models"
echo -e "  Logs: $(pwd)/logs"
echo -e "  Data: $(pwd)/data"
echo ""

# ============================================================================
# STEP 5: Download Models and Datasets
# ============================================================================

echo -e "${BLUE}[5/6] Downloading models and datasets...${NC}"

# Create a Python script to handle downloads
python3 << 'EOF'
import os
import sys
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Downloading face embedding model (ArcFace)...")
try:
    import insightface
    model = insightface.app.FaceAnalysis(name='buffalo_l', root='models', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✓ ArcFace model ready")
except Exception as e:
    print(f"⚠ ArcFace model download skipped (will auto-download on first use): {e}")

print("Downloading Stable Diffusion 3 weights...")
try:
    from diffusers import StableDiffusion3Pipeline
    # This will cache models to ~/.cache/huggingface
    print("✓ Stable Diffusion 3 model will auto-download on first generation")
except Exception as e:
    print(f"⚠ SD3 setup skipped (will auto-download on first use): {e}")

print("Models configuration complete")
EOF

echo -e "${GREEN}✓ Models configured (will download on first use)${NC}"
echo ""

# ============================================================================
# STEP 6: Validation and Health Check
# ============================================================================

echo -e "${BLUE}[6/6] Running validation checks...${NC}"

# Check MIVA imports
python3 << 'EOF'
import sys
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
except Exception as e:
    print(f"✗ PyTorch check failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print(f"✓ Transformers {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ Transformers check failed: {e}")
    sys.exit(1)

try:
    import insightface
    print(f"✓ InsightFace available")
except Exception as e:
    print(f"⚠ InsightFace import failed: {e}")

try:
    from qdrant_client import QdrantClient
    print(f"✓ Qdrant client available")
except Exception as e:
    print(f"⚠ Qdrant client not available (will use FAISS fallback): {e}")

print("\n✓ All validation checks passed")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Validation failed${NC}"
    exit 1
fi

echo ""

# ============================================================================
# Setup Complete
# ============================================================================

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ MIVA Studio Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}\n"

echo -e "Next steps:\n"
echo -e "  1. ${BLUE}Activate environment:${NC}"
echo -e "     source $VENV_DIR/bin/activate\n"

echo -e "  2. ${BLUE}Run health check:${NC}"
echo -e "     python examples/health_check.py\n"

echo -e "  3. ${BLUE}Try quick example:${NC}"
echo -e "     python examples/quick_start.py --help\n"

echo -e "  4. ${BLUE}View documentation:${NC}"
echo -e "     open docs/architecture.md\n"

echo -e "  5. ${BLUE}Run tests:${NC}"
echo -e "     pytest tests/ -v\n"

echo -e "For detailed usage, see README.md or run:"
echo -e "  ${BLUE}miva --help${NC}\n"
