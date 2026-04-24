"""
Configuration management for MIVA Studio.

Loads and validates YAML configuration with environment variable overrides.
All thresholds and parameters are defined here for transparency and reproducibility.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, validator


@dataclass
class RetrievalConfig:
    """Retrieval stage configuration."""
    vector_store_type: str = "qdrant"  # qdrant, faiss, pinecone
    vector_store_path: str = "./vector_store"
    embedding_model: str = "arcface_r50"
    embedding_dim: int = 512
    top_k: int = 5
    diversity_threshold: float = 0.95  # Reranking: avoid near-duplicates
    recall_threshold: float = 0.70  # Minimum acceptable recall@k


@dataclass
class GenerationConfig:
    """Generation stage configuration."""
    model: str = "stable-diffusion-3"
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    ip_adapter_scale: float = 1.0
    seed: Optional[int] = None


@dataclass
class GuardrailConfig:
    """Guardrail enforcement configuration."""
    
    @dataclass
    class IdentityConfig:
        threshold: float = 0.75  # FaceNet-derived threshold
        source: str = "FaceNet (Schroff et al., 2015); same-identity pairs ≥ 0.75"
    
    @dataclass
    class QualityConfig:
        artifact_score_threshold: float = 0.20
    
    @dataclass
    class SafetyConfig:
        nsfw_threshold: float = 0.5
    
    @dataclass
    class AgentConfig:
        max_regeneration_attempts: int = 3
        # CRITICAL: max_regeneration_attempts drives hard-stop behavior
        # Each attempt fails on the same subject_id → systematic problem
        # HARD_STOP should_retry must always be False (invariant)
    
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class ObservabilityConfig:
    """Observability and monitoring configuration."""
    trace_output_dir: str = "./logs/traces"
    metrics_enabled: bool = True
    prometheus_port: int = 8000
    log_level: str = "INFO"
    
    # Alert thresholds (see docs/observability.md)
    alerts: Dict[str, Any] = field(default_factory=lambda: {
        "identity_score_p50": {"threshold": 0.78, "direction": "below", "severity": "CRITICAL"},
        "guardrail_pass_rate": {"threshold": 0.70, "direction": "below", "severity": "WARNING"},
        "hard_stop_rate": {"threshold": 0.05, "direction": "above", "severity": "CRITICAL"},
        "retrieval_p95_latency_ms": {"threshold": 300, "direction": "above", "severity": "WARNING"},
    })


@dataclass
class EvaluationConfig:
    """Evaluation and regression detection configuration."""
    enabled: bool = True
    dataset_path: str = "./data/eval_dataset"
    regression_detection: bool = True
    
    # Regression thresholds (see Section 6 of publication)
    regression_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "identity_score_mean": -0.03,      # 3% absolute drop
        "retrieval_recall_at_3": -0.05,    # 5% absolute drop
        "guardrail_pass_rate": -0.05,
        "hard_stop_rate": 0.02,            # 2% absolute increase
        "false_positive_rate": 0.01,       # 1% absolute increase → CRITICAL
    })


@dataclass
class MIVAConfig:
    """Main MIVA Studio configuration."""
    
    system_version: str = "1.0.0"
    pipeline_version: str = "v1.0"
    
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "MIVAConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Build nested configs
        config = cls()
        
        if 'retrieval' in config_dict:
            config.retrieval = RetrievalConfig(**config_dict['retrieval'])
        if 'generation' in config_dict:
            config.generation = GenerationConfig(**config_dict['generation'])
        if 'guardrails' in config_dict:
            config.guardrails = GuardrailConfig(**config_dict['guardrails'])
        if 'observability' in config_dict:
            config.observability = ObservabilityConfig(**config_dict['observability'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        return config
    
    @classmethod
    def from_env(cls) -> "MIVAConfig":
        """Load configuration from environment variables (overrides YAML)."""
        config_path = os.getenv("MIVA_CONFIG", "config/miva_default.yaml")
        config = cls.from_yaml(config_path)
        
        # Environment variable overrides
        if os.getenv("MIVA_IDENTITY_THRESHOLD"):
            config.guardrails.identity.threshold = float(os.getenv("MIVA_IDENTITY_THRESHOLD"))
        if os.getenv("MIVA_MAX_ATTEMPTS"):
            config.guardrails.agent.max_regeneration_attempts = int(os.getenv("MIVA_MAX_ATTEMPTS"))
        if os.getenv("MIVA_LOG_LEVEL"):
            config.observability.log_level = os.getenv("MIVA_LOG_LEVEL")
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration for production safety."""
        errors = []
        
        # Identity threshold must be in [0.65, 0.85] range (reasonable bounds)
        if not (0.65 <= self.guardrails.identity.threshold <= 0.85):
            errors.append(f"Identity threshold {self.guardrails.identity.threshold} outside safe range [0.65, 0.85]")
        
        # Max attempts must be reasonable (2-5 range)
        if not (2 <= self.guardrails.agent.max_regeneration_attempts <= 5):
            errors.append(f"Max attempts {self.guardrails.agent.max_regeneration_attempts} outside reasonable range [2, 5]")
        
        # Quality threshold must be low (artifacts are acceptable up to this point)
        if self.guardrails.quality.artifact_score_threshold > 0.3:
            errors.append(f"Artifact threshold {self.guardrails.quality.artifact_score_threshold} too high (>0.3)")
        
        if errors:
            for error in errors:
                print(f"CONFIG ERROR: {error}")
            return False
        
        return True
    
    def summary(self) -> str:
        """Return human-readable configuration summary."""
        summary = f"""
MIVA Studio Configuration Summary
==================================
System Version:     {self.system_version}
Pipeline Version:   {self.pipeline_version}

Retrieval:
  Vector Store:     {self.retrieval.vector_store_type} ({self.retrieval.vector_store_path})
  Embedding Model:  {self.retrieval.embedding_model} ({self.retrieval.embedding_dim}D)
  Top-k:            {self.retrieval.top_k}

Generation:
  Model:            {self.generation.model}
  Steps:            {self.generation.num_inference_steps}
  Guidance Scale:   {self.generation.guidance_scale}

Guardrails:
  Identity Threshold: {self.guardrails.identity.threshold} (from {self.guardrails.identity.source})
  Quality Threshold:  {self.guardrails.quality.artifact_score_threshold}
  Max Attempts:       {self.guardrails.agent.max_regeneration_attempts} (HARD_STOP at limit)

Observability:
  Traces:           {self.observability.trace_output_dir}
  Metrics:          {'Enabled' if self.observability.metrics_enabled else 'Disabled'}
  Log Level:        {self.observability.log_level}

Evaluation:
  Dataset:          {self.evaluation.dataset_path}
  Regression Check: {'Enabled' if self.evaluation.regression_detection else 'Disabled'}
"""
        return summary


def get_default_config() -> MIVAConfig:
    """Get default configuration."""
    return MIVAConfig()


def get_config_from_file(path: str) -> MIVAConfig:
    """Load configuration from file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return MIVAConfig.from_yaml(path)


def get_config() -> MIVAConfig:
    """Get configuration (from env or defaults)."""
    return MIVAConfig.from_env()
