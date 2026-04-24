"""
Observability layer for MIVA Studio.

Emits structured session traces for each generation request.
Traces capture all decisions, scores, and outcomes for auditability and debugging.

Key principle: The trace for a failed session is as detailed as for a successful one.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AttemptRecord:
    """Record of a single generation attempt."""
    attempt_number: int
    identity_score: Optional[float]
    artifact_score: Optional[float]
    guardrail_decision: str  # PASS, REJECT_AND_RETRY, HARD_STOP


@dataclass
class SessionTrace:
    """Complete structured trace for a generation session."""
    
    session_id: str
    subject_id: str
    timestamp_start: str
    timestamp_end: Optional[str] = None
    
    # Retrieval
    retrieval_latency_ms: float = 0
    anchors_retrieved: int = 0
    
    # Generation attempts
    attempts: List[AttemptRecord] = None
    
    # Outcome
    final_action: str = "PENDING"  # PASS, HARD_STOP, COLD_START_REJECTION
    output_delivered: bool = False
    final_identity_score: Optional[float] = None
    failure_reason: Optional[str] = None
    total_latency_ms: float = 0
    
    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []
    
    def record_attempt(
        self,
        attempt_number: int,
        identity_score: Optional[float],
        artifact_score: Optional[float],
        decision: str
    ):
        """Record a generation attempt."""
        self.attempts.append(AttemptRecord(
            attempt_number=attempt_number,
            identity_score=identity_score,
            artifact_score=artifact_score,
            guardrail_decision=decision
        ))
    
    def finalize(
        self,
        final_action: str,
        output_delivered: bool = False,
        final_identity_score: Optional[float] = None,
        failure_reason: Optional[str] = None,
        total_latency_ms: float = 0
    ):
        """Finalize the trace with outcome information."""
        self.final_action = final_action
        self.output_delivered = output_delivered
        self.final_identity_score = final_identity_score
        self.failure_reason = failure_reason
        self.total_latency_ms = total_latency_ms
        self.timestamp_end = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['attempts'] = [asdict(a) for a in self.attempts]
        return d
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class SessionTracer:
    """
    Manages session trace generation and persistence.
    
    Emits structured traces to disk for each session.
    Traces enable full auditability and post-mortems on failures.
    """
    
    def __init__(self, config):
        self.config = config
        self.trace_dir = Path(config.observability.trace_output_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def new_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())
    
    def create_trace(self, session_id: str, subject_id: str) -> SessionTrace:
        """Create a new session trace."""
        return SessionTrace(
            session_id=session_id,
            subject_id=subject_id,
            timestamp_start=datetime.utcnow().isoformat() + "Z"
        )
    
    def save_trace(self, trace: SessionTrace):
        """Save trace to disk."""
        trace_path = self.trace_dir / f"{trace.session_id}.json"
        
        with open(trace_path, 'w') as f:
            f.write(trace.to_json())
        
        self.logger.debug(f"Trace saved: {trace_path}")
    
    def load_trace(self, session_id: str) -> Optional[SessionTrace]:
        """Load a trace from disk."""
        trace_path = self.trace_dir / f"{session_id}.json"
        
        if not trace_path.exists():
            return None
        
        with open(trace_path, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_trace(data)
    
    @staticmethod
    def _dict_to_trace(data: Dict[str, Any]) -> SessionTrace:
        """Convert dictionary back to SessionTrace object."""
        attempts = [
            AttemptRecord(**a) for a in data.get('attempts', [])
        ]
        
        return SessionTrace(
            session_id=data['session_id'],
            subject_id=data['subject_id'],
            timestamp_start=data['timestamp_start'],
            timestamp_end=data.get('timestamp_end'),
            retrieval_latency_ms=data.get('retrieval_latency_ms', 0),
            anchors_retrieved=data.get('anchors_retrieved', 0),
            attempts=attempts,
            final_action=data.get('final_action', 'PENDING'),
            output_delivered=data.get('output_delivered', False),
            final_identity_score=data.get('final_identity_score'),
            failure_reason=data.get('failure_reason'),
            total_latency_ms=data.get('total_latency_ms', 0)
        )
    
    def list_traces(self, limit: Optional[int] = None) -> List[SessionTrace]:
        """List all traces on disk."""
        traces = []
        for trace_file in sorted(self.trace_dir.glob("*.json"), reverse=True):
            with open(trace_file, 'r') as f:
                data = json.load(f)
            traces.append(self._dict_to_trace(data))
            
            if limit and len(traces) >= limit:
                break
        
        return traces
    
    def get_trace_summary(self, trace: SessionTrace) -> Dict[str, Any]:
        """Generate human-readable summary of a trace."""
        return {
            'session_id': trace.session_id,
            'subject_id': trace.subject_id,
            'timestamp': trace.timestamp_start,
            'attempts': len(trace.attempts),
            'final_action': trace.final_action,
            'success': trace.output_delivered,
            'identity_score': f"{trace.final_identity_score:.4f}" if trace.final_identity_score else "N/A",
            'latency_ms': f"{trace.total_latency_ms:.0f}",
            'failure_reason': trace.failure_reason or "N/A"
        }


class MetricsEmitter:
    """
    Emit real-time operational metrics.
    
    In production, sends to Prometheus, CloudWatch, or similar.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def emit_generation_result(self, result):
        """Emit metrics for a generation result."""
        metrics = {
            'identity_score': result.final_identity_score or 0,
            'attempts': result.attempts,
            'success': result.success,
            'latency_ms': 0,  # Would come from trace
        }
        
        # In production, would send to Prometheus/CloudWatch
        self.logger.debug(f"Metrics: {metrics}")
