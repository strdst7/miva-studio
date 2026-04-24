# MIVA Studio Architecture

## Overview

MIVA Studio is a four-stage production pipeline for identity-critical visual generation:

```
Stage 1: RETRIEVAL          Stage 2: GENERATION         Stage 3: GUARDRAILS        Stage 4: OBSERVABILITY
(independent)               (agentic loop, max 3)       (enforcement, not logging) (traces + metrics)

Query Vector Store  ──→   Diffusion + IP-Adapter  ──→   Identity Check      ──→   Emit Trace
  • ANN search            • Cross-attention              • Quality Check             • Session trace
  • Reranking             • Conditioning                 • Safety Check              • Metrics
  • Quality gate          • Max 3 attempts               • HARD_STOP logic           • Regression detection
```

---

## Stage 1: Retrieval

**Purpose:** Fetch identity embeddings from vector store, independent from generation.

**Key Property:** Retrieval quality is evaluated separately from generation quality (Section 4.2 of publication).

### Retrieval Pipeline

```python
def retrieve_and_rerank(subject_id: str, top_k: int = 5) -> list[AnchorEmbedding]:
    # Stage 1a: ANN search
    candidates = vector_store.query(subject_id, top_k=top_k * 3)
    
    if not candidates:
        return []  # Cold start — handled by caller
    
    # Stage 1b: Rerank by pose diversity (max-marginal relevance)
    selected = []
    for candidate in sorted(candidates, key=lambda c: c.quality_score, reverse=True):
        if not selected:
            selected.append(candidate)
        else:
            max_sim_to_selected = max(
                cosine_similarity(candidate.embedding, s.embedding)
                for s in selected
            )
            if max_sim_to_selected < 0.95:  # Diversity threshold
                selected.append(candidate)
        if len(selected) == top_k:
            break
    
    # Stage 1c: Quality gate (enrollment-time validation)
    return [a for a in selected if a.quality_score > 0.85]
```

### Vector Store Structure

```
Vector Store (Qdrant / FAISS)
├── subject_id: "alice"
│   ├── anchor_0: {embedding: [512-dim], quality_score: 0.92, pose: "frontal"}
│   ├── anchor_1: {embedding: [512-dim], quality_score: 0.89, pose: "3/4"}
│   ├── anchor_2: {embedding: [512-dim], quality_score: 0.91, pose: "profile"}
│   └── ... (5-20 anchors per subject)
├── subject_id: "bob"
│   ├── anchor_0: {...}
│   └── ...
```

**Retrieval Evaluation:** See `docs/retrieval_design.md` and Section 4.2 of publication.

---

## Stage 2: Generation

**Purpose:** Generate images with retrieved identity embeddings injected via cross-attention.

### Generation with IP-Adapter Conditioning

```python
def generate_with_identity_conditioning(
    anchors: list[np.ndarray],
    prompt: str,
    guidance_scale: float = 7.5
) -> Image:
    
    # 1. Encode prompt
    prompt_embeddings = text_encoder.encode(prompt)
    
    # 2. Prepare identity conditioning
    identity_context = {
        "embeddings": anchors,          # Retrieved identity anchors
        "scale": 1.0                     # IP-Adapter scale
    }
    
    # 3. Run diffusion with cross-attention conditioning
    for step in range(num_inference_steps):
        noise_pred = unet(
            sample=latent,
            timestep=timestep,
            encoder_hidden_states=prompt_embeddings,
            cross_attention_kwargs={
                "ip_adapter": identity_context
            },
            guidance_scale=guidance_scale
        )
        latent = scheduler.step(noise_pred, timestep, latent)
    
    # 4. Decode to image
    image = vae_decoder.decode(latent)
    return image
```

**Critical Property:** Retrieval must meaningfully change outputs (ablation test, Section 4.3).

---

## Stage 3: Guardrails

**Design:** ENFORCEMENT guardrails, not monitoring. Failed checks block delivery.

### Guardrail Decision Tree

```
Per attempt:
┌─────────────────────────────────────┐
│ Identity Validator                  │
│ Quality Validator        ──→ ALL PASS? ──→ GuardrailAction.PASS
│ Safety Validator                    │
└─────────────────────────────────────┘
         │ ANY FAIL
         ▼
  attempt < MAX? ──→ YES ──→ GuardrailAction.REJECT_AND_RETRY
         │
         NO
         ▼
  GuardrailAction.HARD_STOP
  (should_retry = False, permanently)
```

### Identity Consistency Validator

```python
class IdentityConsistencyValidator:
    THRESHOLD = 0.75  # FaceNet-derived
    
    def validate(self, generated_embedding: np.ndarray, anchor_embeddings: list[np.ndarray]):
        scores = [cosine_similarity(generated_embedding, a) for a in anchor_embeddings]
        max_score = max(scores)
        
        return ValidationResult(
            passed=max_score >= self.THRESHOLD,
            identity_score=max_score,
            should_retry=True
        )
```

**Threshold Source:** FaceNet (Schroff et al., 2015). Same-identity pairs in L2-normalized 128-d space cluster ≥ 0.75.

### CRITICAL System Invariant

```python
if decision.action == GuardrailAction.HARD_STOP:
    assert decision.should_retry == False, (
        "INVARIANT VIOLATION: HARD_STOP with should_retry=True. "
        "This creates an unbounded loop."
    )
```

This invariant is enforced in:
- `miva/guardrails.py::GuardrailDecision.__post_init__()`
- `tests/test_guardrails.py::test_hard_stop_is_always_terminal()`

---

## Stage 4: Observability

**Purpose:** Emit structured traces for every session, including failures.

### Session Trace Schema

```python
@dataclass
class SessionTrace:
    session_id: str                    # UUID
    subject_id: str
    timestamp_start: str               # ISO 8601
    
    # Retrieval
    retrieval_latency_ms: float
    anchors_retrieved: int
    
    # Generation attempts (one per attempt)
    attempts: list[AttemptRecord]
    
    # Outcome
    final_action: str                  # PASS, HARD_STOP, COLD_START_REJECTION
    output_delivered: bool
    final_identity_score: Optional[float]
    failure_reason: Optional[str]
    total_latency_ms: float
```

**Key Principle:** Failed sessions have complete traces. This enables post-mortems.

### Operational Metrics

See `docs/observability.md` for:
- `identity_score_p50`: Median identity score (alert if < 0.78)
- `guardrail_pass_rate`: First-attempt pass rate (alert if < 0.70)
- `hard_stop_rate`: Sessions terminated by HARD_STOP (alert if > 0.05)
- `retrieval_p95_latency_ms`: 95th percentile retrieval latency (alert if > 300ms)
- `identity_score_7d_delta`: 7-day trending metric for silent degradation detection

---

## Agentic Loop

The generation loop is bounded by **semantic stopping conditions**, not time/token limits:

```python
def generation_agent(subject_id: str, anchors: list, prompt: str) -> AgentResult:
    for attempt in range(1, MAX_REGENERATION_ATTEMPTS + 1):
        # Generate
        image, embedding = generator.generate(anchors, prompt)
        
        # Evaluate guardrails
        decision = guardrail.evaluate(image, embedding, anchors, attempt)
        
        # Decide
        if decision.action == GuardrailAction.PASS:
            return success(image)
        elif decision.action == GuardrailAction.HARD_STOP:
            # CRITICAL: should_retry == False (invariant enforced)
            return hard_stop(decision.reason)
        else:  # REJECT_AND_RETRY
            continue  # Retry
    
    # Should be unreachable
    raise RuntimeError("Loop exited without terminal decision")
```

**Why 3 attempts?** Three consecutive failures on the same subject_id indicates systematic failure (retrieval quality issue, subject not representable, etc.), not a transient generation hiccup.

---

## Configuration

See `config/miva_default.yaml` for:
- Retrieval parameters (vector_store_type, top_k, diversity_threshold)
- Generation parameters (model, guidance_scale, num_steps)
- Guardrail thresholds (with sources and justification)
- Observability configuration (trace directory, metric alerting)
- Evaluation configuration (dataset path, regression thresholds)

---

## Deployment Architecture

### Single-Instance

```
┌─────────────────────────────────────────────┐
│  MIVA Pipeline                              │
│  ├── Config (YAML)                          │
│  ├── Vector Store (local FAISS / Qdrant)    │
│  ├── Generation Model (cached)              │
│  └── Observability (local traces)           │
└─────────────────────────────────────────────┘
```

### Kubernetes (Production)

```
┌──────────────────────────────────────────────────────────┐
│  MIVA Pipeline Deployment                                │
│  ├── Pod: Pipeline (requests: 4 CPU, 10GB mem)           │
│  ├── Pod: Vector Store (dedicated, persistent volume)    │
│  ├── Service: Pipeline (ClusterIP)                       │
│  ├── ConfigMap: Configuration                            │
│  ├── HPA: Auto-scale 1-10 pods based on demand           │
│  └── Observability: Prometheus + Loki                    │
└──────────────────────────────────────────────────────────┘
```

See `docs/deployment-k8s.md` for details.

---

## Failure Modes

See `docs/failure_analysis.md` for documented real failures and remediation.

Key failure modes:
1. **Identity drift:** Generated output resembles but is not the subject → guardrail blocks (identity score < 0.75)
2. **Over-generation loops:** Agent retries indefinitely → HARD_STOP at max_attempts
3. **Unsafe variation:** Inappropriate context/content → safety validator blocks
4. **Silent degradation:** System-level quality decline → 7-day delta metric detects

---

## References

- **FaceNet:** Schroff et al., 2015. Basis for 0.75 cosine similarity threshold.
- **ArcFace:** Deng et al., 2019. Embedding model architecture.
- **IP-Adapter:** Ye et al., 2023. Cross-attention conditioning mechanism.
- **RAG:** Lewis et al., 2020. Architectural foundation.
- **Max-Marginal Relevance:** Carbonell & Goldstein, 1998. Reranking strategy.

---

**Status:** Production v1.0  
**Last updated:** April 2025
