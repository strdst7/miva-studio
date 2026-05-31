"""
Test suite for MIVA Studio guardrail enforcement.

Critical tests that must pass before deployment:
- test_hard_stop_is_always_terminal: Ensures HARD_STOP.should_retry == False
- test_guardrail_blocks_identity_drift: Verifies identity checks block low-similarity outputs
- test_loop_terminates_at_max_attempts: Confirms loop terminates on max attempts

Run with:
    pytest tests/test_guardrails.py -v
"""

import pytest
import numpy as np
from miva.config import MIVAConfig
from miva.guardrails import GuardrailEvaluator, GuardrailAction


@pytest.fixture
def config():
    """Create test configuration."""
    return MIVAConfig()


@pytest.fixture
def evaluator(config):
    """Create guardrail evaluator."""
    return GuardrailEvaluator(config)


@pytest.fixture
def dummy_image():
    """Create dummy generated image."""
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_embedding():
    """Create dummy face embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


@pytest.fixture
def anchor_embeddings():
    """Create dummy anchor embeddings."""
    embeddings = []
    for _ in range(5):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        embeddings.append(emb)
    return embeddings


# ============================================================================
# CRITICAL TESTS — Must pass before any deployment
# ============================================================================

def test_hard_stop_is_always_terminal(evaluator, dummy_image, dummy_embedding, anchor_embeddings):
    """
    CRITICAL: HARD_STOP decisions must have should_retry == False.
    
    Failure: Agent loop becomes unbounded (can retry indefinitely).
    This is a system safety invariant.
    """
    # Create embedding with low similarity (will fail guardrail)
    low_sim_embedding = np.random.randn(512).astype(np.float32)
    low_sim_embedding = low_sim_embedding / (np.linalg.norm(low_sim_embedding) + 1e-8)
    
    # Force HARD_STOP by hitting max attempts
    max_attempts = evaluator.config.guardrails.agent.max_regeneration_attempts
    
    decision = evaluator.evaluate(
        generated_image=dummy_image,
        generated_embedding=low_sim_embedding,
        anchor_embeddings=anchor_embeddings,
        attempt_number=max_attempts  # This will trigger HARD_STOP
    )
    
    # CRITICAL: If decision is HARD_STOP, should_retry MUST be False
    assert decision.action == GuardrailAction.HARD_STOP
    assert decision.should_retry == False, (
        "CRITICAL INVARIANT VIOLATION: HARD_STOP with should_retry=True. "
        "This creates an unbounded loop."
    )


def test_guardrail_blocks_identity_drift(evaluator, dummy_image, anchor_embeddings):
    """
    CRITICAL: Guardrail must block outputs with low identity similarity.
    
    Failure: Identity drift passes through to users (confident wrong answer).
    """
    # Create embedding with low similarity to anchors
    low_sim_embedding = np.random.randn(512).astype(np.float32)
    low_sim_embedding = low_sim_embedding / (np.linalg.norm(low_sim_embedding) + 1e-8)
    
    decision = evaluator.evaluate(
        generated_image=dummy_image,
        generated_embedding=low_sim_embedding,
        anchor_embeddings=anchor_embeddings,
        attempt_number=1
    )
    
    # Must not pass
    assert decision.action != GuardrailAction.PASS, (
        "CRITICAL: Low-similarity embedding was PASSED by identity guardrail. "
        "Identity drift check is not working."
    )
    
    # Should be rejected for retry
    assert decision.action in [GuardrailAction.REJECT_AND_RETRY, GuardrailAction.HARD_STOP]


def test_loop_terminates_at_max_attempts(evaluator, dummy_image, anchor_embeddings):
    """
    CRITICAL: Generation loop must terminate at MAX_REGENERATION_ATTEMPTS.
    
    Failure: Loop continues indefinitely, consuming compute and user time.
    """
    max_attempts = evaluator.config.guardrails.agent.max_regeneration_attempts
    
    # Create persistent low-similarity embedding
    low_sim_embedding = np.random.randn(512).astype(np.float32)
    low_sim_embedding = low_sim_embedding / (np.linalg.norm(low_sim_embedding) + 1e-8)
    
    # Test at each attempt number
    for attempt in range(1, max_attempts + 2):
        decision = evaluator.evaluate(
            generated_image=dummy_image,
            generated_embedding=low_sim_embedding,
            anchor_embeddings=anchor_embeddings,
            attempt_number=attempt
        )
        
        if attempt >= max_attempts:
            # Must be HARD_STOP (terminal)
            assert decision.action == GuardrailAction.HARD_STOP, (
                f"Attempt {attempt} >= max {max_attempts} but decision is {decision.action.value}. "
                f"Loop has not terminated."
            )
            assert decision.should_retry == False


# ============================================================================
# Functional tests — Verify guardrail behavior
# ============================================================================

def test_guardrail_passes_valid_output(evaluator, dummy_image, anchor_embeddings):
    """Guardrail allows valid outputs (high similarity + good quality)."""
    # Create embedding very similar to first anchor
    valid_embedding = anchor_embeddings[0] + np.random.randn(512) * 0.01
    valid_embedding = valid_embedding / (np.linalg.norm(valid_embedding) + 1e-8)
    
    decision = evaluator.evaluate(
        generated_image=dummy_image,
        generated_embedding=valid_embedding,
        anchor_embeddings=anchor_embeddings,
        attempt_number=1
    )
    
    # Should pass
    assert decision.action == GuardrailAction.PASS
    assert decision.identity_score is not None
    assert decision.identity_score >= 0.70  # High similarity


def test_guardrail_rejects_face_not_detected(evaluator, dummy_image, anchor_embeddings):
    """Guardrail correctly handles images with no detectable face."""
    # None embedding signals no face detected
    decision = evaluator.evaluate(
        generated_image=dummy_image,
        generated_embedding=None,
        anchor_embeddings=anchor_embeddings,
        attempt_number=1
    )
    
    assert decision.action == GuardrailAction.REJECT_AND_RETRY
    assert decision.identity_score == 0.0
    assert decision.should_retry == True


def test_guardrail_allows_retry_before_max(evaluator, dummy_image, anchor_embeddings):
    """Guardrail allows retry on first few failures."""
    low_sim_embedding = np.random.randn(512).astype(np.float32)
    low_sim_embedding = low_sim_embedding / (np.linalg.norm(low_sim_embedding) + 1e-8)
    
    max_attempts = evaluator.config.guardrails.agent.max_regeneration_attempts
    
    for attempt in range(1, max_attempts):
        decision = evaluator.evaluate(
            generated_image=dummy_image,
            generated_embedding=low_sim_embedding,
            anchor_embeddings=anchor_embeddings,
            attempt_number=attempt
        )
        
        # Before max, should allow retry
        assert decision.should_retry == True, (
            f"Attempt {attempt}/{max_attempts}: should_retry should be True before max"
        )


# ============================================================================
# Configuration tests
# ============================================================================

def test_config_validate(config):
    """Configuration passes validation."""
    assert config.validate() == True


def test_identity_threshold_in_bounds(config):
    """Identity threshold is within reasonable bounds."""
    threshold = config.guardrails.identity.threshold
    assert 0.65 <= threshold <= 0.85, (
        f"Identity threshold {threshold} outside reasonable range [0.65, 0.85]"
    )


def test_max_attempts_reasonable(config):
    """Max regeneration attempts is in reasonable range."""
    max_attempts = config.guardrails.agent.max_regeneration_attempts
    assert 2 <= max_attempts <= 5, (
        f"Max attempts {max_attempts} outside reasonable range [2, 5]"
    )
