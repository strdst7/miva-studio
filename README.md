# MIVA Studio

**Production-grade RAG architecture for identity-critical visual generation.**

[![CI](https://github.com/strdst7/miva-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/mi4inc/miva-studio/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ready Tensor RAG Systems Expert](https://img.shields.io/badge/Ready%20Tensor-RAG%20Systems%20Expert-green)](https://readytensor.ai)

> *A plausible output is not the same as a correct one.*

---

## What Is MIVA Studio?

MIVA Studio is an agentic visual generation system that solves a specific problem: **generating images of a specific person, consistently, without fine-tuning the model per user**.

Standard diffusion models generate visually plausible faces — not identity-consistent ones. Subject-specific fine-tuning (DreamBooth, LoRA) achieves identity consistency but requires hours of retraining per subject, making it untenable for production multi-user systems.

MIVA Studio uses **Retrieval-Augmented Generation** applied to face identity embeddings:

1. **Retrieve** — Pull verified identity anchor vectors from a per-subject vector store
2. **Augment** — Inject retrieved embeddings into the generation pipeline via IP-Adapter cross-attention
3. **Enforce** — Run a multi-stage guardrail layer that hard-stops the agent if identity consistency cannot be achieved

The system is designed for reputationally sensitive contexts — brand representation, professional headshots, identity-sensitive creative work — where a wrong answer delivered confidently is worse than no answer at all.

---

## Architecture Overview

```
User Request (subject_id + params)
         │
         ▼
┌─────────────────────┐
│   STAGE 1: RETRIEVE │  ANN search → MMR rerank → quality gate
│   Vector Store      │  Output: AugmentedContext{embeddings[]}
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ STAGE 2: GENERATE   │  ← Agentic loop (max 3 attempts)
│ IP-Adapter Diffusion│  Identity embeddings injected via cross-attention
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ STAGE 3: GUARDRAILS │  Identity (cosine ≥ 0.75) + Quality + Safety
│ Enforce, not advise │  PASS → deliver | FAIL → retry | MAX → HARD STOP
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ STAGE 4: OBSERVE    │  Structured traces + metrics + alerts
│ Full session trace  │  Every decision logged, including failures
└─────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- (Optional for full pipeline) CUDA-capable GPU with 8GB+ VRAM

### 1. Clone and Set Up

```bash
git clone https://github.com/mi4inc/miva-studio.git
cd miva-studio
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This installs all dependencies, creates a `.env` from the template, and runs the test suite.

### 2. Configure

```bash
cp configs/default.yaml configs/local.yaml
# Edit configs/local.yaml with your vector store endpoint and model paths
```

### 3. Run the Test Suite

```bash
make test
```

### 4. Run Evaluation

```bash
make evaluate
```

### 5. Start the API Server

```bash
make serve
```

---

## Installation (Step-by-Step)

```bash
# 1. Clone the repository
git clone https://github.com/strdst7/miva-studio.git
cd miva-studio

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install core dependencies
pip install -e ".[dev]"

# 5. (GPU pipeline) Install torch with CUDA — edit for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Copy environment template
cp .env.example .env
# Edit .env — see Configuration section below

# 7. Verify installation
python -c "import miva; print(miva.__version__)"

# 8. Run tests
pytest tests/ -v
```

---

## Configuration

Edit `.env` or `configs/local.yaml`:

```yaml
# configs/local.yaml

vector_store:
  provider: qdrant          # qdrant | faiss | pinecone
  host: localhost
  port: 6333
  collection: miva_anchors

retrieval:
  top_k: 5
  diversity_threshold: 0.95
  min_quality_score: 0.80

generation:
  model_id: runwayml/stable-diffusion-v1-5
  ip_adapter_model: h94/IP-Adapter
  device: cuda               # cuda | cpu | mps

guardrails:
  identity_threshold: 0.75
  artifact_threshold: 0.20
  max_regeneration_attempts: 3

observability:
  trace_backend: jsonl       # jsonl | prometheus | otlp
  trace_output_dir: ./traces
  metrics_port: 9090
```

---

## Project Structure

```
miva-studio/
├── README.md
├── pyproject.toml                    # Package metadata and dependencies
├── Makefile                          # Common commands
├── .env.example                      # Environment variable template
│
├── miva/                             # Core library
│   ├── config.py                     # Configuration management
│   ├── pipeline/
│   │   └── rag_pipeline.py           # End-to-end RAG pipeline
│   ├── retrieval/
│   │   └── vector_store.py           # Vector store client + MMR reranking
│   ├── generation/
│   │   └── generator.py              # Conditioned image generation
│   ├── guardrails/
│   │   ├── identity_guardrail.py     # Main guardrail orchestrator
│   │   └── validators/
│   │       ├── identity_validator.py # Cosine similarity enforcement
│   │       ├── quality_validator.py  # Artifact + resolution check
│   │       └── content_safety.py    # NSFW detection
│   ├── agent/
│   │   └── generation_agent.py       # Bounded agentic loop
│   ├── observability/
│   │   └── tracer.py                 # Structured session tracing
│   └── evaluation/
│       ├── runner.py                 # Reproducible evaluation runner
│       └── regression_detector.py   # Cross-version regression detection
│
├── data/
│   ├── eval/
│   │   ├── eval_dataset_v1.yaml      # Evaluation dataset specification
│   │   └── README.md                 # Dataset documentation
│   └── scripts/
│       └── generate_eval_fixtures.py # Test fixture generator
│
├── tests/
│   ├── unit/
│   │   ├── test_guardrails.py        # Guardrail unit tests
│   │   ├── test_retrieval.py         # Retrieval unit tests
│   │   └── test_agent.py             # Agent loop unit tests
│   └── integration/
│       └── test_pipeline.py          # End-to-end integration tests
│
├── docs/
│   ├── architecture.md               # System design + RAG justification
│   ├── retrieval_design.md           # Vector store + recall@k methodology
│   ├── guardrail_spec.md             # All thresholds + sources
│   ├── evaluation_protocol.md        # Evaluation procedure + pass/fail criteria
│   ├── observability.md              # Trace schema + alert thresholds
│   ├── regression_policy.md          # Regression detection + remediation
│   └── failure_analysis.md           # Documented failure cases
│
├── scripts/
│   ├── setup.sh                      # One-command environment setup
│   ├── run_evaluation.sh             # Full evaluation pipeline
│   └── enroll_subject.sh             # Enroll a new subject identity
│
└── .github/
    └── workflows/
        └── ci.yml                    # CI: test + evaluate + regression check
```

---

## Core Commands

```bash
# Environment
make setup          # Full environment setup (first time)
make install        # Install package only

# Testing
make test           # Run all tests
make test-unit      # Unit tests only
make test-integration  # Integration tests only (requires model)
make coverage       # Tests with coverage report

# Evaluation
make evaluate       # Run full evaluation against eval_dataset_v1
make regression     # Compare current vs baseline evaluation report

# Development
make lint           # ruff + mypy
make format         # black + isort
make clean          # Remove __pycache__, .pytest_cache, traces/

# Operations
make serve          # Start API server
make enroll SUBJECT=subject_001 IMAGES=./images/  # Enroll a new subject
```

---

## Evaluation Results (v1.0)

Evaluated on `miva_eval_v1` (50 subjects, 200 test cases, seed=42).  
All criteria were defined before evaluation was run.

| Metric | Threshold (Pass) | Achieved | Grade |
|---|---|---|---|
| Recall@3 | ≥ 0.75 | **0.88** [0.84, 0.92] | Pass |
| Recall@5 | ≥ 0.85 | **0.93** [0.90, 0.96] | Pass |
| Identity score (delivered) | ≥ 0.80 | **0.831** [0.818, 0.844] | Pass |
| Guardrail pass rate (1st attempt) | ≥ 0.70 | **0.784** | Pass |
| Hard stop rate | < 0.05 | **0.031** | Pass |
| False positive rate | < 0.03 | **0.039** | Weak Pass |
| Retrieval ablation delta | > 0.15 | **+0.210** (p<0.001) | Strong Pass |

---

## Documentation

- [Architecture & RAG Justification](docs/architecture.md)
- [Retrieval Design & Recall Methodology](docs/retrieval_design.md)
- [Guardrail Specification & Thresholds](docs/guardrail_spec.md)
- [Evaluation Protocol](docs/evaluation_protocol.md)
- [Observability & Alerting](docs/observability.md)
- [Regression Policy](docs/regression_policy.md)
- [Failure Analysis](docs/failure_analysis.md)

---

## Publication

This system is documented in:

> Nur Amirah Mohd Kamil (2025). *MIVA Studio: A Production-Grade RAG Architecture for Identity-Critical Visual Generation.* Ready Tensor RAG Systems Expert Capstone. MI4 Inc.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built by [Nur Amirah Mohd Kamil]
