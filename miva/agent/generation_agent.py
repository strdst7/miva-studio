"""
miva/agent/generation_agent.py
Bounded agentic generation loop with semantic stopping conditions.

The loop stops when:
  (A) A generated output passes all guardrail checks — success.
  (B) attempt_number reaches MAX_REGENERATION_ATTEMPTS — HARD_STOP.
  (C) Retrieval returns zero anchors — COLD_START_REJECTION.
  (D) An upstream service returns a non-retryable error — UPSTREAM_ERROR.

Stopping on iteration count alone (without semantic criteria) is not a
controlled agent — it is an uncontrolled agent with a budget. MIVA Studio
uses semantic stopping: the agent knows *why* it is stopping.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from miva.guardrails.identity_guardrail import GuardrailAction, GuardrailDecision, IdentityGuardrail
from miva.retrieval.vector_store import ColdStartResult, RetrievalResult

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

class AgentOutcome(str, Enum):
    SUCCESS              = "SUCCESS"
    HARD_STOP            = "HARD_STOP"
    COLD_START_REJECTION = "COLD_START_REJECTION"
    UPSTREAM_ERROR       = "UPSTREAM_ERROR"


@dataclass
class AttemptRecord:
    attempt_number: int
    generation_latency_ms: float
    identity_score: float
    artifact_score: float
    guardrail_decision: str
    validators_passed: list[str]
    validators_failed: list[str]


@dataclass
class AgentResult:
    outcome: AgentOutcome
    subject_id: str
    output_image: Optional[object] = None     # PIL.Image or None
    delivered: bool = False
    final_identity_score: float = 0.0
    total_attempts: int = 0
    attempt_records: list[AttemptRecord] = field(default_factory=list)
    failure_reason: Optional[str] = None
    total_latency_ms: float = 0.0

    @classmethod
    def success(cls, subject_id, image, score, attempts, records, latency) -> "AgentResult":
        return cls(
            outcome=AgentOutcome.SUCCESS,
            subject_id=subject_id,
            output_image=image,
            delivered=True,
            final_identity_score=score,
            total_attempts=attempts,
            attempt_records=records,
            total_latency_ms=latency,
        )

    @classmethod
    def hard_stop(cls, subject_id, reason, attempts, records, best_score, latency) -> "AgentResult":
        return cls(
            outcome=AgentOutcome.HARD_STOP,
            subject_id=subject_id,
            delivered=False,
            final_identity_score=best_score,
            total_attempts=attempts,
            attempt_records=records,
            failure_reason=reason,
            total_latency_ms=latency,
        )

    @classmethod
    def cold_start(cls, subject_id, latency) -> "AgentResult":
        return cls(
            outcome=AgentOutcome.COLD_START_REJECTION,
            subject_id=subject_id,
            delivered=False,
            failure_reason="IDENTITY_NOT_FOUND",
            total_latency_ms=latency,
        )

    @classmethod
    def upstream_error(cls, subject_id, reason, latency) -> "AgentResult":
        return cls(
            outcome=AgentOutcome.UPSTREAM_ERROR,
            subject_id=subject_id,
            delivered=False,
            failure_reason=reason,
            total_latency_ms=latency,
        )


# ── Agent ─────────────────────────────────────────────────────────────────────

class GenerationAgent:
    """
    Bounded agentic loop for identity-consistent visual generation.

    The agent has exactly four terminal states (see AgentOutcome).
    It cannot run indefinitely. This is enforced both by logic and by test.
    See: tests/unit/test_agent.py::test_hard_stop_is_always_terminal
    """

    def __init__(
        self,
        retriever,
        generator,
        guardrail: IdentityGuardrail,
        tracer,
        max_attempts: int = 3,
    ):
        self.retriever  = retriever
        self.generator  = generator
        self.guardrail  = guardrail
        self.tracer     = tracer
        self.max_attempts = max_attempts

    def run(self, subject_id: str, params: dict) -> AgentResult:
        """
        Execute the full RAG generation loop for one session.

        Retrieval happens once per session (not per attempt).
        The retrieved context is reused across all generation attempts
        because the issue is generation variance, not retrieval variance.
        If retrieval itself is the problem, more retries won't help —
        that's the COLD_START_REJECTION path.
        """
        session_start = time.perf_counter()
        attempt_records: list[AttemptRecord] = []
        best_identity_score = 0.0
        last_failure_reason = "MAX_ATTEMPTS_EXCEEDED"

        # ── Stage 1: Retrieve (once per session) ─────────────────────────────
        retrieval_result = self.retriever.retrieve(subject_id)

        if isinstance(retrieval_result, ColdStartResult) or retrieval_result.is_empty():
            latency = (time.perf_counter() - session_start) * 1000
            logger.warning("Cold start: no anchors for subject_id=%s", subject_id)
            return AgentResult.cold_start(subject_id, latency)

        anchor_embeddings = [a.embedding for a in retrieval_result.anchors]

        # ── Stages 2–3: Generate → Evaluate (agentic loop) ───────────────────
        for attempt in range(1, self.max_attempts + 1):
            attempt_start = time.perf_counter()

            # Stage 2: Generate
            try:
                candidate_image = self.generator.generate(
                    context=retrieval_result,
                    params=params,
                    attempt=attempt,
                )
            except Exception as exc:
                latency = (time.perf_counter() - session_start) * 1000
                logger.error("Generation upstream error at attempt %d: %s", attempt, exc)
                return AgentResult.upstream_error(subject_id, str(exc), latency)

            gen_latency_ms = (time.perf_counter() - attempt_start) * 1000

            # Stage 3: Guardrail evaluation
            decision: GuardrailDecision = self.guardrail.evaluate(
                candidate_image,
                anchor_embeddings,
                attempt_number=attempt,
            )

            best_identity_score = max(best_identity_score, decision.identity_score)
            last_failure_reason = decision.reason or last_failure_reason

            record = AttemptRecord(
                attempt_number=attempt,
                generation_latency_ms=gen_latency_ms,
                identity_score=decision.identity_score,
                artifact_score=decision.artifact_score,
                guardrail_decision=decision.action.value,
                validators_passed=decision.validators_passed,
                validators_failed=decision.validators_failed,
            )
            attempt_records.append(record)
            total_latency = (time.perf_counter() - session_start) * 1000

            if decision.action == GuardrailAction.PASS:
                logger.info(
                    "Session SUCCESS: subject=%s attempt=%d identity=%.4f latency=%.0fms",
                    subject_id, attempt, decision.identity_score, total_latency,
                )
                result = AgentResult.success(
                    subject_id=subject_id,
                    image=candidate_image,
                    score=decision.identity_score,
                    attempts=attempt,
                    records=attempt_records,
                    latency=total_latency,
                )
                self.tracer.record_session(result, retrieval_result)
                return result

            if decision.action == GuardrailAction.HARD_STOP:
                # This assertion is belt-and-suspenders — GuardrailDecision.__post_init__
                # already enforces this, but we verify here too.
                assert decision.should_retry is False, (
                    "HARD_STOP arrived with should_retry=True — guardrail invariant violated"
                )
                logger.warning(
                    "Session HARD_STOP: subject=%s after %d attempts. Best score: %.4f",
                    subject_id, attempt, best_identity_score,
                )
                result = AgentResult.hard_stop(
                    subject_id=subject_id,
                    reason=last_failure_reason,
                    attempts=attempt,
                    records=attempt_records,
                    best_score=best_identity_score,
                    latency=total_latency,
                )
                self.tracer.record_session(result, retrieval_result)
                return result

            if decision.action == GuardrailAction.REJECT:
                # Non-retryable (e.g., NSFW) — stop immediately regardless of attempt count
                total_latency = (time.perf_counter() - session_start) * 1000
                result = AgentResult.hard_stop(
                    subject_id=subject_id,
                    reason=decision.reason,
                    attempts=attempt,
                    records=attempt_records,
                    best_score=best_identity_score,
                    latency=total_latency,
                )
                self.tracer.record_session(result, retrieval_result)
                return result

            # REJECT_AND_RETRY → continue loop
            logger.debug("Attempt %d rejected, retrying...", attempt)

        # This line is intentionally unreachable.
        # HARD_STOP fires at attempt == max_attempts inside the loop.
        # If we reach here, there is a logic error in the guardrail.
        raise RuntimeError(
            "GenerationAgent loop exited without a terminal decision. "
            "This indicates a logic error in IdentityGuardrail. "
            "HARD_STOP should have fired at attempt_number == max_attempts."
        )
