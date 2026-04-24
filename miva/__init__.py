"""
MIVA Studio — Production-Grade RAG for Identity-Critical Visual Generation

Version: 1.0.0
Author: Nur Amirah Mohd Kamil
Affiliation: MI4 Inc.

A production-grade agentic visual generation system that retrieves identity embeddings
from a vector store and injects them into diffusion-based image generation via
cross-attention conditioning. The system enforces identity fidelity through multi-stage
guardrails with hard-stopping behavior, observability traces, and regression detection.

Key Components:
    - miva.retrieval: Vector store and embedding retrieval with ANN search + reranking
    - miva.generation: Diffusion-based image generation with IP-Adapter conditioning
    - miva.guardrails: Multi-stage enforcement guardrails (identity, quality, safety)
    - miva.agent: Agentic loop with semantic stopping conditions (max 3 attempts)
    - miva.observability: Structured session traces and real-time metrics
    - miva.evaluation: Independent retrieval eval, ablation tests, regression detection
    - miva.cli: Command-line interface for enrollment, generation, evaluation

Usage:
    >>> from miva import MIVAPipeline
    >>> pipeline = MIVAPipeline(config_path="config/miva_default.yaml")
    >>> result = pipeline.generate(subject_id="alice", prompt="portrait")
    >>> print(result.output_path)

For CLI usage:
    $ miva generate --subject_id alice --prompt "portrait"
    $ miva enroll --subject_id alice --reference_images ./data/alice/

For detailed documentation:
    - README.md: Quick start and overview
    - docs/architecture.md: System design and data flows
    - docs/evaluation_protocol.md: Evaluation methodology and pass/fail criteria
    - docs/guardrail_spec.md: Guardrail thresholds and justification
"""

__title__ = "MIVA Studio"
__version__ = "1.0.0"
__author__ = "Nur Amirah Mohd Kamil"
__author_email__ = "amirah@mi4-ai.com"
__url__ = "https://github.com/mi4-ai/miva-studio"
__license__ = "MIT"
__copyright__ = "Copyright 2025 MI4 Inc."

# Version info
VERSION_INFO = (1, 0, 0)

# Import main components for convenience
try:
    from miva.pipeline import MIVAPipeline
    from miva.config import MIVAConfig
    from miva.observability import SessionTracer
except ImportError as e:
    # Allow partial imports during development
    import warnings
    warnings.warn(f"Could not import main components: {e}")

__all__ = [
    "MIVAPipeline",
    "MIVAConfig",
    "SessionTracer",
    "__version__",
    "__author__",
]
