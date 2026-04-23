# configs/default.yaml
# Default configuration for MIVA Studio.
# Override with configs/local.yaml or environment variables.

vector_store:
  provider: qdrant
  host: localhost
  port: 6333
  collection: miva_anchors
  embedding_dim: 512

retrieval:
  top_k: 5
  candidate_multiplier: 3        # fetch top_k * multiplier before reranking
  diversity_threshold: 0.95      # MMR: max intra-anchor cosine sim
  min_anchor_count: 3            # warn if fewer than this many anchors found

generation:
  model_id: runwayml/stable-diffusion-v1-5
  ip_adapter_model: h94/IP-Adapter
  device: cpu
  num_inference_steps: 30
  guidance_scale: 7.5
  ip_adapter_scale: 0.8
  model_cache_dir: ./model_cache
  output_resolution: [512, 512]

face_encoder:
  backend: arcface
  model_path: ./model_cache/arcface_r100.onnx
  detection_threshold: 0.5
  embedding_dim: 512

guardrails:
  identity_threshold: 0.75       # Cosine sim — see docs/guardrail_spec.md
  artifact_threshold: 0.20       # Max artifact score
  max_regeneration_attempts: 3
  enable_content_safety: true

observability:
  trace_backend: jsonl
  trace_output_dir: ./traces
  metrics_port: 9090
  log_level: INFO

evaluation:
  seed: 42
  output_dir: ./eval_outputs
  bootstrap_samples: 1000
  confidence_level: 0.95

api:
  host: 0.0.0.0
  port: 8000
  workers: 1
