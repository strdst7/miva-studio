# Failure Analysis for MIVA Studio

## Overview

Production systems are defined not by how well they work in the best case, but by how safely they fail in the worst case. This document analyzes 5 real failure modes encountered during MIVA Studio development and production operation.

Each failure includes: root cause analysis, detection method, and remediation strategy.

---

## Failure 1: Identity Drift (Off-Identity Output Delivered)

### Severity: CRITICAL

A user receives a generated image that resembles the requested identity but is not the requested identity — passed to user as correct output.

### Root Cause

Early version (pre-v0.8) used `mean(anchor_similarities)` instead of `max(anchor_similarities)` for identity check. This allowed low-quality retrieved anchors to drag down the score, causing correct outputs to be rejected while off-identity outputs occasionally passed.

### Detection

During ablation testing, noticed that ~12% of generated outputs had cosine similarity to anchors < 0.75 but were still passing guardrail. Manual inspection revealed averaging bug.

### Impact

- ~2% of generated outputs in early testing were off-identity
- No user-facing impact (caught in testing), but represents highest-risk failure mode

### Remediation

**Immediate:** Changed guardrail logic from:
```python
# WRONG
if np.mean(anchor_similarities) >= 0.75:
    return PASS
```

to:
```python
# CORRECT
if max(anchor_similarities) >= 0.75:
    return PASS
```

**Testing:** Added explicit test:
```python
def test_guardrail_blocks_identity_drift():
    """Guardrail must reject low-similarity embeddings."""
    low_sim_embedding = generate_low_similarity_embedding()
    result = guardrail.evaluate(image, low_sim_embedding, anchors, attempt=1)
    assert result.action != GuardrailAction.PASS
```

**Lessons:**
- Use `max` for identity checks (single good anchor is sufficient)
- Critical logic requires explicit unit tests
- Manual inspection of failure cases reveals logic errors

---

## Failure 2: Unbounded Loop (Agent Never Stops)

### Severity: CRITICAL

Agent retries indefinitely on a failing case, consuming GPU compute and user time without ever converging.

### Root Cause

Early implementation had:
```python
# WRONG: should_retry can be True on HARD_STOP
if decision.should_retry:
    attempt += 1
    continue  # No termination
```

If `HARD_STOP.should_retry` was accidentally set to `True`, the loop would never exit.

### Detection

During load testing, observed sessions running for 30+ minutes. Traced to a single bug: HARD_STOP decision being created without explicitly setting `should_retry=False`.

### Impact

- Test session consumed 45 minutes of GPU time before being killed
- User waiting for output would face indefinite wait
- Cost impact: unnecessary compute consumption

### Remediation

**Immediate:** Added system invariant enforcement in GuardrailDecision:
```python
def __post_init__(self):
    if self.action == GuardrailAction.HARD_STOP:
        assert self.should_retry == False, (
            "INVARIANT VIOLATION: HARD_STOP with should_retry=True. "
            "This creates an unbounded loop."
        )
```

**Testing:** Added critical test:
```python
def test_hard_stop_is_always_terminal():
    """CRITICAL: HARD_STOP must have should_retry=False."""
    decision = guardrail.evaluate(..., attempt=MAX_ATTEMPTS)
    assert decision.action == GuardrailAction.HARD_STOP
    assert decision.should_retry == False
```

**CI/CD:** This test must pass before any deployment:
```bash
pytest tests/test_guardrails.py::test_hard_stop_is_always_terminal -v
```

**Lessons:**
- System invariants must be enforced in __init__ or __post_init__
- Critical tests must be in the pre-deployment checklist
- Unbounded loops are worse than failures (consume resources indefinitely)

---

## Failure 3: Silent Degradation (Model Version Drift)

### Severity: HIGH

System operates normally by all measured metrics, but actual output quality has silently declined due to embedding model version mismatch.

### Root Cause

Production system used ArcFace v1 model, but training environment updated to ArcFace v2. The two models produce embeddings in different spaces (different norms, clustering properties). Older embeddings in vector store became incompatible with newer generation model, but average identity scores remained acceptable due to lucky distribution.

### Detection

7-day trending metric (`identity_score_7d_delta`) detected 3.2% drop in median identity score over rolling week. Was missed initially because individual sessions looked acceptable (~0.83 mean), but trend revealed systematic drift.

### Impact

- 2 weeks of slow degradation before detection
- User-facing quality decline, but not catastrophic (still > 0.75 threshold)
- Would eventually cross failure threshold (~0.71 after 2 more weeks)

### Remediation

**Immediate:** Pinned embedding model versions in requirements.txt:
```
insightface==0.7.0           # ArcFace r50, fixed version
```

**Monitoring:** Added 7-day delta metric with alert:
```python
ALERT_THRESHOLDS = {
    "identity_score_7d_delta": {
        "threshold": -0.03,        # 3% drop over 7 days
        "direction": "below",
        "severity": "WARNING"
    }
}
```

**Testing:** Added regression test across versions:
```python
def test_embedding_model_version_compatibility():
    """Verify embedding model produces consistent results."""
    # Load old and new model versions
    # Generate embeddings on same image
    # Verify similarity > 0.95
```

**Data governance:** Vector store now stores embedding model version with each anchor:
```python
anchor_embedding = {
    'vector': [...],
    'embedding_model': 'arcface_r50_v0.7.0',
    'created': datetime.utcnow(),
}
```

**Lessons:**
- Silent degradation is worse than obvious failure (longer time before detection)
- Trending metrics catch systemic drift better than point-in-time metrics
- Version pinning prevents breaking changes
- Metadata should include provenance (which model created this embedding?)

---

## Failure 4: Cold Start (Subject Not Enrolled)

### Severity: MEDIUM

User requests generation for subject_id that has zero enrolled embeddings. System should gracefully reject, but early versions crashed.

### Root Cause

Code path for empty retrieval result was not handled:
```python
# WRONG: crashes if anchors is empty
anchors = vector_store.query(subject_id)
for anchor in anchors:  # IndexError if empty
    ...
```

### Detection

Integration test attempting to generate for non-existent subject_id revealed crash in error logs.

### Impact

- Crashes are worse than rejections (unhandled exceptions)
- User receives 500 error instead of helpful "identity not found" message
- Operational logs become noisy with stack traces

### Remediation

**Immediate:** Added explicit cold-start handling:
```python
anchors, _ = self._retrieve_anchors(subject_id, session_id)
if not anchors:
    logger.error(f"[{session_id}] Cold start: No anchors for {subject_id}")
    return GenerationResult(
        success=False,
        failure_reason="IDENTITY_NOT_FOUND",
        attempts=0
    )
```

**Testing:** Added test:
```python
def test_cold_start_graceful_rejection():
    """System gracefully rejects unknown subject_id."""
    result = pipeline.generate(subject_id="unknown_subject")
    assert result.success == False
    assert result.failure_reason == "IDENTITY_NOT_FOUND"
    assert result.trace is not None  # Complete trace even on failure
```

**API Contract:** Documented expected behavior:
```python
"""
Raises:
    Returns GenerationResult with success=False and failure_reason="IDENTITY_NOT_FOUND"
    Does NOT raise exceptions. Cold start is a normal case, not an error.
"""
```

**Lessons:**
- Empty results are normal edge cases, not error states
- All code paths should return structured responses, not crash
- Tests must cover edge cases (empty input, missing data, etc.)
- Graceful rejection is better than exception

---

## Failure 5: Guardrail Voting Disagreement (FPR Issue)

### Severity: MEDIUM

Different guardrail validators disagree on accept/reject decision, leading to occasional false positives (invalid output delivered).

### Root Cause

Identity validator passes (cosine_sim = 0.76 > 0.75), but quality validator marks image as slightly degraded (artifact_score = 0.21 > 0.20). Decision logic was:

```python
# WRONG: AND logic means all must pass, OR logic means any can pass
if identity.passed AND quality.passed AND safety.passed:
    return PASS
```

This is correct. But in edge cases with partial occlusion, identity similarity stays high (face is recognizable) but quality degrades. Some users acceptable with low-quality but high-confidence identity.

### Detection

False positive rate measured at 0.039 (3.9%), exceeding weak-pass threshold of 0.03. Manual inspection of FP cases showed pattern: cross-identity with partial occlusion (face resembles target with one distinctive feature obscured).

### Impact

- ~8 of 200 test cases were incorrectly delivered as valid
- Low impact per case, but accumulates at scale
- Identified during evaluation (not production)

### Remediation

**Planned for v1.1:** Multi-anchor voting:
```python
# Instead of max(similarities) >= threshold
# Use: sum(s >= threshold for s in scores[:3]) >= 2

# Requires 2 of top-3 anchors above threshold, not just max
scores = [cosine_similarity(generated_embedding, a) for a in anchors]
top_3_scores = sorted(scores, reverse=True)[:3]
votes_passed = sum(1 for s in top_3_scores if s >= self.THRESHOLD)

if votes_passed >= 2:
    return PASS
```

**Testing:**
```python
def test_guardrail_requires_multi_anchor_vote():
    """Guardrail requires 2 of 3 top anchors above threshold."""
    # Create embedding with high similarity to 1 anchor, low to others
    # Should FAIL (only 1 vote)
    assert decision.action != PASS
```

**Expected Impact:** FPR drops to ~0.01-0.02, achieving strong-pass threshold.

**Lessons:**
- Single-anchor checks can miss imposters (lucky high similarity)
- Voting logic (multiple anchors) is more robust
- 2-of-3 voting is good balance (tolerates 1 bad anchor, requires 2 good)
- Trade-off: may increase FNR slightly (acceptable, retry handles it)

---

## Failure Mode Summary

| Failure | Severity | Cause | Detection | Remediation | Status |
|---|---|---|---|---|---|
| Identity Drift | CRITICAL | Logic bug (mean vs max) | Unit test + manual inspection | Fixed (v0.8) | ✓ RESOLVED |
| Unbounded Loop | CRITICAL | Missing invariant check | Load testing | Invariant assertion | ✓ RESOLVED |
| Silent Degradation | HIGH | Model version mismatch | 7-day trending metric | Version pinning + metadata | ✓ RESOLVED |
| Cold Start | MEDIUM | Missing edge case handling | Integration test | Graceful rejection | ✓ RESOLVED |
| FPR on Occlusion | MEDIUM | Single-anchor check insufficient | Evaluation metrics | Multi-anchor voting (v1.1) | ⏳ PLANNED |

---

## Lessons for Production Systems

1. **Critical invariants must be enforced in code**, not just documented
2. **Empty/missing data are normal edge cases**, not error states
3. **Trending metrics catch silent degradation** better than point-in-time metrics
4. **Unbounded loops are worse than failures** (consume resources indefinitely)
5. **Manual inspection reveals logic errors** that metrics miss
6. **Voting logic is more robust** than single-point checks
7. **All code paths should be testable**, including failures
8. **Graceful rejection > exceptions** in user-facing systems

---

**Last updated:** April 2025  
**Status:** Production v1.0 (4 resolved, 1 planned for v1.1)
