"""
miva/observability/tracer.py
Structured session tracing and metric emission.

Every session produces a complete trace — success and failure alike.
Post-mortems on production failures require the trace of the failed session,
not just the happy-path logs.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MIVASessionTrace:
    """Complete record of one user session — every decision, every score."""
    # Session identity
    session_id: str
    subject_id: str
    pipeline_version: str
    timestamp_utc: str

    # Retrieval
    retrieval_latency_ms: float
    anchors_retrieved: int
    anchor_quality_scores: list[float]
    estimated_recall: float

    # Generation attempts
    attempts: list[dict]

    # Terminal outcome
    final_action: str
    output_delivered: bool
    total_attempts: int
    final_identity_score: float
    failure_reason: Optional[str]
    total_latency_ms: float


class SessionTracer:
    """
    Records structured session traces to the configured backend.
    
    Trace backends:
      jsonl   — One JSON object per line in a rolling file (default, zero dependencies)
      prometheus — Emit metrics to a Prometheus pushgateway
      otlp    — OpenTelemetry trace export (requires opentelemetry packages)
    """

    def __init__(
        self,
        backend: str = "jsonl",
        output_dir: str = "./traces",
        pipeline_version: str = "unknown",
        metrics_port: int = 9090,
    ):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.pipeline_version = pipeline_version
        self.metrics_port = metrics_port
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session_count = 0
        self._success_count = 0
        self._hard_stop_count = 0
        self._identity_scores: list[float] = []

        if backend == "prometheus":
            self._init_prometheus()

    def _init_prometheus(self):
        try:
            from prometheus_client import Counter, Gauge, Histogram, start_http_server
            self._prom_sessions = Counter("miva_sessions_total", "Total sessions", ["outcome"])
            self._prom_identity_score = Histogram(
                "miva_identity_score",
                "Identity cosine similarity of delivered outputs",
                buckets=[0.65, 0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.88, 0.90, 0.95, 1.0],
            )
            self._prom_hard_stop_rate = Gauge("miva_hard_stop_rate_1h", "Hard stop rate (1h window)")
            self._prom_attempts = Histogram(
                "miva_generation_attempts",
                "Attempts per session",
                buckets=[1, 2, 3],
            )
            start_http_server(self.metrics_port)
            logger.info("Prometheus metrics available on port %d", self.metrics_port)
        except ImportError:
            logger.warning("prometheus-client not installed — Prometheus metrics disabled")
            self.backend = "jsonl"

    def record_session(self, agent_result, retrieval_result) -> MIVASessionTrace:
        """Build and persist a session trace from agent and retrieval results."""
        trace = MIVASessionTrace(
            session_id=str(uuid.uuid4()),
            subject_id=agent_result.subject_id,
            pipeline_version=self.pipeline_version,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            retrieval_latency_ms=retrieval_result.latency_ms if hasattr(retrieval_result, 'latency_ms') else 0.0,
            anchors_retrieved=len(retrieval_result.anchors) if hasattr(retrieval_result, 'anchors') else 0,
            anchor_quality_scores=[
                a.quality_score for a in retrieval_result.anchors
            ] if hasattr(retrieval_result, 'anchors') else [],
            estimated_recall=retrieval_result.estimated_recall if hasattr(retrieval_result, 'estimated_recall') else 0.0,
            attempts=[
                {
                    "attempt_number": r.attempt_number,
                    "generation_latency_ms": r.generation_latency_ms,
                    "identity_score": r.identity_score,
                    "artifact_score": r.artifact_score,
                    "guardrail_decision": r.guardrail_decision,
                    "validators_passed": r.validators_passed,
                    "validators_failed": r.validators_failed,
                }
                for r in agent_result.attempt_records
            ],
            final_action=agent_result.outcome.value,
            output_delivered=agent_result.delivered,
            total_attempts=agent_result.total_attempts,
            final_identity_score=agent_result.final_identity_score,
            failure_reason=agent_result.failure_reason,
            total_latency_ms=agent_result.total_latency_ms,
        )

        self._persist(trace)
        self._update_metrics(trace)
        return trace

    def _persist(self, trace: MIVASessionTrace) -> None:
        """Write trace to configured backend."""
        if self.backend == "jsonl":
            self._write_jsonl(trace)
        elif self.backend == "prometheus":
            self._emit_prometheus(trace)
        # otlp: implement with opentelemetry-sdk

    def _write_jsonl(self, trace: MIVASessionTrace) -> None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        trace_file = self.output_dir / f"traces_{date_str}.jsonl"
        with open(trace_file, "a") as f:
            f.write(json.dumps(asdict(trace)) + "\n")

    def _emit_prometheus(self, trace: MIVASessionTrace) -> None:
        try:
            self._prom_sessions.labels(outcome=trace.final_action).inc()
            if trace.output_delivered:
                self._prom_identity_score.observe(trace.final_identity_score)
            if trace.total_attempts > 0:
                self._prom_attempts.observe(trace.total_attempts)
        except Exception as exc:
            logger.warning("Prometheus emit failed: %s", exc)

    def _update_metrics(self, trace: MIVASessionTrace) -> None:
        """Update in-memory aggregates for real-time monitoring."""
        self._session_count += 1
        if trace.output_delivered:
            self._success_count += 1
            self._identity_scores.append(trace.final_identity_score)
        if trace.final_action == "HARD_STOP":
            self._hard_stop_count += 1

    def get_summary(self) -> dict:
        """Current session-level metrics snapshot."""
        import numpy as np
        scores = self._identity_scores or [0.0]
        return {
            "total_sessions": self._session_count,
            "delivered": self._success_count,
            "hard_stops": self._hard_stop_count,
            "hard_stop_rate": self._hard_stop_count / max(self._session_count, 1),
            "identity_score_p50": float(np.percentile(scores, 50)),
            "identity_score_p10": float(np.percentile(scores, 10)),
        }
