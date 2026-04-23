#!/usr/bin/env bash
# scripts/setup.sh
# One-command environment setup for MIVA Studio.
# Run: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -euo pipefail

PYTHON_MIN="3.10"
VENV_DIR=".venv"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         MIVA Studio Setup                ║"
echo "║  Identity-Critical Visual Generation RAG ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check Python version ──────────────────────────────────────────────────
echo -e "${YELLOW}[1/7] Checking Python version...${NC}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ python3 not found. Install Python 3.10+ first.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "      Found Python $PYTHON_VERSION"

python3 -c "
import sys
major, minor = sys.version_info.major, sys.version_info.minor
if (major, minor) < (3, 10):
    print('Python 3.10+ required.')
    sys.exit(1)
" || { echo -e "${RED}✗ Python $PYTHON_MIN+ required. Found $PYTHON_VERSION${NC}"; exit 1; }
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# ── 2. Create virtual environment ────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Creating virtual environment at $VENV_DIR...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo "      $VENV_DIR already exists — skipping creation"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 3. Upgrade pip ───────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# ── 4. Install package and dependencies ──────────────────────────────────────
echo -e "${YELLOW}[4/7] Installing MIVA Studio (dev mode)...${NC}"
pip install -e ".[dev]" --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# ── 5. Set up .env ───────────────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] Setting up environment variables...${NC}"
if [ -f ".env" ]; then
    echo "      .env already exists — skipping"
else
    cp .env.example .env
    echo -e "${GREEN}✓ .env created from template${NC}"
    echo -e "${YELLOW}      ⚠ Edit .env before running the full pipeline${NC}"
fi

# ── 6. Create output directories ─────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Creating output directories...${NC}"
mkdir -p eval_outputs traces logs
echo -e "${GREEN}✓ Directories ready${NC}"

# ── 7. Run test suite ────────────────────────────────────────────────────────
echo -e "${YELLOW}[7/7] Running unit tests to verify installation...${NC}"
if pytest tests/unit/ -q --tb=short 2>&1; then
    echo -e "${GREEN}✓ All unit tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed. Check output above.${NC}"
    echo -e "  This may be expected if optional model dependencies are not installed."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Setup complete!                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "  Activate the environment:"
echo -e "    ${YELLOW}source $VENV_DIR/bin/activate${NC}"
echo ""
echo "  Key commands:"
echo -e "    ${YELLOW}make test${NC}        — Run unit tests"
echo -e "    ${YELLOW}make evaluate${NC}    — Run full evaluation"
echo -e "    ${YELLOW}make serve${NC}       — Start API server"
echo -e "    ${YELLOW}make enroll SUBJECT=id IMAGES=./path/${NC}  — Enroll subject"
echo ""
echo "  Edit configs/local.yaml to set your vector store endpoint and model paths."
echo ""
