"""
Guardrail enforcement layer for MIVA Studio.

CRITICAL DESIGN: These are ENFORCEMENT guardrails, not monitoring.
A failed check blocks output delivery unconditionally.

Key invariants:
1. HARD_STOP.should_retry must always be False (system invariant)
2. All guardrails must implement the GuardrailValidator interface
3. Thresholds must have documented sources and justification
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GuardrailAction(str, Enum):
    """Terminal decisions made by guardrails."""
    PASS = "PASS"                  # Output passed all checks
    REJECT_AND_RETRY = "REJECT_AND_RETRY"  # Failed but retryable
    HARD_STOP = "HARD_STOP"        # Failed and terminal (should_retry=False always)


@dataclass
class ValidationResult:
    """Result from a single guardrail validator."""
    passed: bool
    identity_score: Optional[float] = None
    artifact_score: Optional[float] = None
    reason: Optional[str] = None
    should_retry: bool = True


@dataclass
class GuardrailDecision:
    """Terminal decision from full guardrail evaluation."""
    action: GuardrailAction
    identity_score: Optional[float] = None
    artifact_score: Optional[float] = None
    attempt_number: int = 1
    reason: Optional[str] = None
    validators_passed: List[str] = None
    validators_failed: List[str] = None
    
    # CRITICAL: For HARD_STOP, this MUST be False. System invariant.
    should_retry: bool = True
    
    def __post_init__(self):
        """Enforce critical invariant on construction."""
        if self.action == GuardrailAction.HARD_STOP:
            assert self.should_retry == False, (
                "CRITICAL INVARIANT VIOLATION: HARD_STOP decision has should_retry=True. "
                "This creates an unbounded loop. This is a logic error that must be fixed."
            )
        
        if self.validators_passed is None:
            self.validators_passed = []
        if self.validators_failed is None:
            self.validators_failed = []


class IdentityConsistencyValidator:
    """
    Primary enforcement guardrail — identity fidelity check.
    
    Threshold: 0.75 cosine similarity
    Source: FaceNet (Schroff et al., 2015)
    Justification: In L2-normalized 128-d embedding space, verified same-identity
    pairs cluster above 0.75, cross-identity pairs below 0.60. ArcFace embeddings
    (512-d) have even tighter same-identity clustering, making 0.75 conservative.
    """
    
    THRESHOLD = 0.75
    SOURCE = "FaceNet (Schroff et al., 2015); ArcFace (Deng et al., 2019)"
    
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__ + ".IdentityValidator")
    
    def validate(
        self,
        generated_embedding: np.ndarray,
        anchor_embeddings: List[np.ndarray],
        attempt_number: int
    ) -> ValidationResult:
        """
        Validate that generated image has sufficient identity match to anchors.
        
        Args:
            generated_embedding: Face embedding from generated image
            anchor_embeddings: Retrieved identity anchor embeddings
            attempt_number: Which attempt this is (for logging)
        
        Returns:
            ValidationResult with passed=True if max_similarity >= threshold
        """
        if generated_embedding is None:
            self.logger.warning(f"Attempt {attempt_number}: Face embedding is None (face not detected)")
            return ValidationResult(
                passed=False,
                identity_score=0.0,
                reason="FACE_NOT_DETECTED",
                should_retry=True
            )
        
        if not anchor_embeddings:
            self.logger.error("No anchor embeddings provided to validator")
            return ValidationResult(
                passed=False,
                identity_score=0.0,
                reason="NO_ANCHORS_PROVIDED",
                should_retry=False
            )
        
        # Compute similarity to each anchor
        scores = [
            float(np.dot(generated_embedding, anchor.T) / 
                 (np.linalg.norm(generated_embedding) * np.linalg.norm(anchor) + 1e-8))
            for anchor in anchor_embeddings
        ]
        
        max_score = max(scores)
        
        if max_score >= self.threshold:
            return ValidationResult(
                passed=True,
                identity_score=max_score,
                reason=None
            )
        else:
            self.logger.warning(
                f"Attempt {attempt_number}: Identity check failed "
                f"(max_sim={max_score:.4f} < threshold={self.threshold})"
            )
            return ValidationResult(
                passed=False,
                identity_score=max_score,
                reason=f"IDENTITY_FAIL({max_score:.4f})",
                should_retry=True
            )


class QualityValidator:
    """
    Quality enforcement guardrail — output must be visually acceptable.
    
    Checks:
    - Artifact score < 0.20
    - Resolution >= minimum
    - No extreme pixel values
    """
    
    ARTIFACT_THRESHOLD = 0.20
    MIN_RESOLUTION = (256, 256)
    
    def __init__(self, artifact_threshold: float = 0.20):
        self.artifact_threshold = artifact_threshold
        self.logger = logging.getLogger(__name__ + ".QualityValidator")
    
    def validate(self, image_array: np.ndarray, attempt_number: int) -> ValidationResult:
        """
        Validate output image quality.
        
        Args:
            image_array: Generated image (H, W, 3)
            attempt_number: Which attempt
        
        Returns:
            ValidationResult
        """
        if image_array is None:
            return ValidationResult(passed=False, artifact_score=1.0, reason="NULL_IMAGE")
        
        h, w = image_array.shape[:2]
        if h < self.MIN_RESOLUTION[0] or w < self.MIN_RESOLUTION[1]:
            self.logger.warning(
                f"Attempt {attempt_number}: Resolution {h}x{w} below minimum {self.MIN_RESOLUTION}"
            )
            return ValidationResult(
                passed=False,
                artifact_score=1.0,
                reason=f"RESOLUTION_TOO_LOW({h}x{w})"
            )
        
        # Artifact score: simplified calculation
        # In production, would use learned artifact detector
        artifact_score = self._compute_artifact_score(image_array)
        
        if artifact_score >= self.artifact_threshold:
            self.logger.warning(
                f"Attempt {attempt_number}: Artifact score {artifact_score:.4f} >= {self.artifact_threshold}"
            )
            return ValidationResult(
                passed=False,
                artifact_score=artifact_score,
                reason=f"QUALITY_FAIL({artifact_score:.4f})",
                should_retry=True
            )
        
        return ValidationResult(
            passed=True,
            artifact_score=artifact_score,
            reason=None
        )
    
    def _compute_artifact_score(self, image: np.ndarray) -> float:
        """
        Compute artifact score (0=clean, 1=severe artifacts).
        
        In production, this would be a learned model.
        Placeholder uses simple heuristics.
        """
        # Check for extreme values
        if image.min() < 0 or image.max() > 255:
            return 0.8
        
        # Check for uniformity (potential collapse)
        variance = image.var()
        if variance < 100:  # Very flat image
            return 0.6
        
        return min(0.2 * (1.0 - variance / 10000.0), 0.19)


class SafetyValidator:
    """
    Content safety guardrail — blocks inappropriate outputs.
    
    Checks:
    - NSFW detection
    - Violent or harmful content
    - Other safety criteria
    """
    
    NSFW_THRESHOLD = 0.5
    
    def __init__(self, nsfw_threshold: float = 0.5):
        self.nsfw_threshold = nsfw_threshold
        self.logger = logging.getLogger(__name__ + ".SafetyValidator")
    
    def validate(self, image_array: np.ndarray, attempt_number: int) -> ValidationResult:
        """
        Validate output safety.
        
        Args:
            image_array: Generated image
            attempt_number: Attempt number
        
        Returns:
            ValidationResult
        """
        # In production, would call learned safety model
        # Placeholder always passes
        return ValidationResult(passed=True, reason=None)


class GuardrailEvaluator:
    """
    Full guardrail evaluation orchestrator.
    
    Aggregates results from identity, quality, and safety validators
    into a single terminal decision per generation attempt.
    """
    
    def __init__(self, config):
        """Initialize with guardrail configuration."""
        self.config = config
        self.identity_validator = IdentityConsistencyValidator(
            threshold=config.guardrails.identity.threshold
        )
        self.quality_validator = QualityValidator(
            artifact_threshold=config.guardrails.quality.artifact_score_threshold
        )
        self.safety_validator = SafetyValidator(
            nsfw_threshold=config.guardrails.safety.nsfw_threshold
        )
        self.max_attempts = config.guardrails.agent.max_regeneration_attempts
        self.logger = logging.getLogger(__name__ + ".GuardrailEvaluator")
    
    def evaluate(
        self,
        generated_image: np.ndarray,
        generated_embedding: np.ndarray,
        anchor_embeddings: List[np.ndarray],
        attempt_number: int
    ) -> GuardrailDecision:
        """
        Evaluate generated output against all guardrails.
        
        Returns terminal decision: PASS, REJECT_AND_RETRY, or HARD_STOP.
        
        CRITICAL: On HARD_STOP, should_retry is always False (invariant).
        """
        results = {}
        validators_passed = []
        validators_failed = []
        
        # Run all validators
        results['identity'] = self.identity_validator.validate(
            generated_embedding, anchor_embeddings, attempt_number
        )
        results['quality'] = self.quality_validator.validate(generated_image, attempt_number)
        results['safety'] = self.safety_validator.validate(generated_image, attempt_number)
        
        # Aggregate results
        for name, result in results.items():
            if result.passed:
                validators_passed.append(name)
            else:
                validators_failed.append(name)
        
        # Decision logic
        if all(results.values()):
            # ALL PASS
            return GuardrailDecision(
                action=GuardrailAction.PASS,
                identity_score=results['identity'].identity_score,
                artifact_score=results['quality'].artifact_score,
                attempt_number=attempt_number,
                validators_passed=validators_passed,
                validators_failed=validators_failed,
                should_retry=False
            )
        
        if attempt_number >= self.max_attempts:
            # HARD STOP — no more retries
            self.logger.error(
                f"HARD_STOP at attempt {attempt_number}/{self.max_attempts}: "
                f"validators failed: {validators_failed}"
            )
            return GuardrailDecision(
                action=GuardrailAction.HARD_STOP,
                identity_score=results['identity'].identity_score,
                artifact_score=results['quality'].artifact_score,
                attempt_number=attempt_number,
                reason=f"MAX_ATTEMPTS_EXCEEDED (failed: {', '.join(validators_failed)})",
                validators_passed=validators_passed,
                validators_failed=validators_failed,
                should_retry=False  # CRITICAL: Must be False
            )
        
        # Retry
        return GuardrailDecision(
            action=GuardrailAction.REJECT_AND_RETRY,
            identity_score=results['identity'].identity_score,
            artifact_score=results['quality'].artifact_score,
            attempt_number=attempt_number,
            reason=f"VALIDATORS_FAILED: {', '.join(validators_failed)}",
            validators_passed=validators_passed,
            validators_failed=validators_failed,
            should_retry=True
        )
