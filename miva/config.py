"""
miva/config.py
Configuration management — loads from YAML, overridable by environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Sub-configs ───────────────────────────────────────────────────────────────

class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection: str = "miva_anchors"
    embedding_dim: int = 512


class RetrievalConfig(BaseModel):
    top_k: int = 5
    candidate_multiplier: int = 3
    diversity_threshold: float = 0.95
    min_anchor_count: int = 3


class GenerationConfig(BaseModel):
    model_id: str = "runwayml/stable-diffusion-v1-5"
    ip_adapter_model: str = "h94/IP-Adapter"
    device: str = "cpu"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    ip_adapter_scale: float = 0.8
    model_cache_dir: str = "./model_cache"
    output_resolution: list[int] = Field(default_factory=lambda: [512, 512])


class FaceEncoderConfig(BaseModel):
    backend: str = "arcface"
    model_path: str = "./model_cache/arcface_r100.onnx"
    detection_threshold: float = 0.5
    embedding_dim: int = 512


class GuardrailConfig(BaseModel):
    identity_threshold: float = 0.75
    artifact_threshold: float = 0.20
    max_regeneration_attempts: int = 3
    enable_content_safety: bool = True


class ObservabilityConfig(BaseModel):
    trace_backend: str = "jsonl"
    trace_output_dir: str = "./traces"
    metrics_port: int = 9090
    log_level: str = "INFO"


class EvaluationConfig(BaseModel):
    seed: int = 42
    output_dir: str = "./eval_outputs"
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


# ── Root config ───────────────────────────────────────────────────────────────

class MIVAConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    face_encoder: FaceEncoderConfig = Field(default_factory=FaceEncoderConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: str | Path | None = None) -> MIVAConfig:
    """
    Load configuration from YAML file with environment variable overrides.

    Resolution order (highest to lowest priority):
    1. Environment variables (MIVA_* prefix)
    2. configs/local.yaml (if exists)
    3. configs/default.yaml
    4. Built-in defaults
    """
    config_data: dict = {}

    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    local_path = Path(__file__).parent.parent / "configs" / "local.yaml"

    for path in [default_path, local_path, config_path]:
        if path and Path(path).exists():
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
                _deep_merge(config_data, loaded)

    _apply_env_overrides(config_data)
    return MIVAConfig(**config_data)


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(config: dict) -> None:
    """Apply MIVA_* environment variables to config dict."""
    mappings = {
        "MIVA_VECTOR_STORE_PROVIDER": ("vector_store", "provider"),
        "MIVA_QDRANT_HOST":           ("vector_store", "host"),
        "MIVA_QDRANT_PORT":           ("vector_store", "port"),
        "MIVA_QDRANT_COLLECTION":     ("vector_store", "collection"),
        "MIVA_DEVICE":                ("generation", "device"),
        "MIVA_MODEL_ID":              ("generation", "model_id"),
        "MIVA_IDENTITY_THRESHOLD":    ("guardrails", "identity_threshold"),
        "MIVA_ARTIFACT_THRESHOLD":    ("guardrails", "artifact_threshold"),
        "MIVA_MAX_REGENERATION_ATTEMPTS": ("guardrails", "max_regeneration_attempts"),
        "MIVA_TRACE_BACKEND":         ("observability", "trace_backend"),
        "MIVA_TRACE_OUTPUT_DIR":      ("observability", "trace_output_dir"),
        "MIVA_LOG_LEVEL":             ("observability", "log_level"),
        "MIVA_EVAL_SEED":             ("evaluation", "seed"),
        "MIVA_EVAL_OUTPUT_DIR":       ("evaluation", "output_dir"),
    }

    numeric_keys = {
        "MIVA_QDRANT_PORT", "MIVA_MAX_REGENERATION_ATTEMPTS", "MIVA_EVAL_SEED"
    }
    float_keys = {
        "MIVA_IDENTITY_THRESHOLD", "MIVA_ARTIFACT_THRESHOLD"
    }

    for env_key, (section, field) in mappings.items():
        val = os.environ.get(env_key)
        if val is not None:
            if env_key in numeric_keys:
                val = int(val)
            elif env_key in float_keys:
                val = float(val)
            config.setdefault(section, {})[field] = val
