"""
MIVA Studio main pipeline orchestrator.

Coordinates four stages:
1. Retrieval: Fetch identity embeddings from vector store
2. Generation: Diffusion-based image generation with IP-Adapter conditioning
3. Guardrails: Enforce identity consistency and quality constraints
4. Observability: Emit structured traces and metrics

The pipeline operates as a four-stage sequential process with an agentic loop
around generation and guardrails (max 3 attempts per session).
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import numpy as np

from miva.config import MIVAConfig, get_config
from miva.guardrails import GuardrailEvaluator, GuardrailAction
from miva.observability import SessionTracer, SessionTrace

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation session."""
    session_id: str
    subject_id: str
    success: bool
    output_path: Optional[str]
    final_identity_score: Optional[float]
    attempts: int
    failure_reason: Optional[str]
    trace: Optional[SessionTrace]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'subject_id': self.subject_id,
            'success': self.success,
            'output_path': self.output_path,
            'final_identity_score': self.final_identity_score,
            'attempts': self.attempts,
            'failure_reason': self.failure_reason,
        }


class MIVAPipeline:
    """
    MIVA Studio production pipeline.
    
    Orchestrates four stages:
    - Stage 1: Retrieval (independent, measured separately)
    - Stage 2: Generation (agentic loop, max 3 attempts)
    - Stage 3: Guardrails (enforcement, not monitoring)
    - Stage 4: Observability (traces and metrics)
    """
    
    def __init__(self, config: Optional[MIVAConfig] = None):
        """Initialize pipeline with configuration."""
        self.config = config or get_config()
        self.guardrail = GuardrailEvaluator(self.config)
        self.tracer = SessionTracer(self.config)
        
        # Create output directories
        Path(self.config.observability.trace_output_dir).mkdir(parents=True, exist_ok=True)
        Path("./outputs").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MIVA Pipeline initialized (v{self.config.system_version})")
    
    def generate(
        self,
        subject_id: str,
        prompt: str = "professional portrait",
        num_outputs: int = 1,
        output_dir: str = "./outputs",
        seed: Optional[int] = None
    ) -> List[GenerationResult]:
        """
        Generate identity-consistent images for a subject.
        
        This is the main entry point to the pipeline. Orchestrates:
        1. Retrieval of identity anchors
        2. Generation with identity conditioning
        3. Guardrail evaluation
        4. Observability trace emission
        
        Args:
            subject_id: Subject to generate images for
            prompt: Generation prompt (will be augmented with identity context)
            num_outputs: Number of images to generate
            output_dir: Where to save outputs
            seed: Random seed (for reproducibility)
        
        Returns:
            List of GenerationResult objects (one per output)
        """
        results = []
        
        for i in range(num_outputs):
            session_id = self.tracer.new_session_id()
            
            logger.info(f"[{session_id}] Starting generation for subject_id='{subject_id}'")
            
            # Stage 1: Retrieval
            logger.info(f"[{session_id}] Stage 1: Retrieving identity anchors...")
            anchors, retrieval_latency = self._retrieve_anchors(subject_id, session_id)
            
            if not anchors:
                logger.error(f"[{session_id}] Cold start: No anchors found for {subject_id}")
                result = GenerationResult(
                    session_id=session_id,
                    subject_id=subject_id,
                    success=False,
                    output_path=None,
                    final_identity_score=None,
                    attempts=0,
                    failure_reason="IDENTITY_NOT_FOUND"
                )
                results.append(result)
                continue
            
            # Stages 2-3: Agentic loop (Generation + Guardrails)
            logger.info(f"[{session_id}] Stages 2-3: Generation loop (max {self.config.guardrails.agent.max_regeneration_attempts} attempts)...")
            
            session_result = self._run_generation_loop(
                session_id=session_id,
                subject_id=subject_id,
                anchors=anchors,
                prompt=prompt,
                output_dir=output_dir,
                seed=seed
            )
            
            # Stage 4: Observability
            logger.info(f"[{session_id}] Stage 4: Emitting observability trace...")
            self.tracer.save_trace(session_result.trace)
            
            results.append(session_result)
            
            # Log result
            if session_result.success:
                logger.info(
                    f"[{session_id}] ✓ SUCCESS: Generated identity-consistent image "
                    f"(score={session_result.final_identity_score:.4f}, attempts={session_result.attempts})"
                )
            else:
                logger.warning(
                    f"[{session_id}] ✗ FAILURE: {session_result.failure_reason} (attempts={session_result.attempts})"
                )
        
        return results
    
    def _retrieve_anchors(self, subject_id: str, session_id: str) -> tuple[List[np.ndarray], float]:
        """
        Stage 1: Retrieve identity anchors from vector store.
        
        Returns:
            (anchor_embeddings, retrieval_latency_ms)
        """
        import time
        start = time.time()
        
        try:
            # In production, would query actual vector store
            # Placeholder: return empty for now
            anchors = []
            latency = (time.time() - start) * 1000
            return anchors, latency
        except Exception as e:
            logger.error(f"[{session_id}] Retrieval failed: {e}")
            return [], 0
    
    def _run_generation_loop(
        self,
        session_id: str,
        subject_id: str,
        anchors: List[np.ndarray],
        prompt: str,
        output_dir: str,
        seed: Optional[int]
    ) -> GenerationResult:
        """
        Stages 2-3: Run agentic generation loop with guardrails.
        
        Loop logic:
        - Max MAX_REGENERATION_ATTEMPTS attempts
        - Each iteration: generate → evaluate guardrails
        - On PASS: deliver output and stop
        - On REJECT_AND_RETRY + attempts < max: retry
        - On HARD_STOP: terminate and return error
        
        Critical invariant: The loop MUST terminate.
        """
        max_attempts = self.config.guardrails.agent.max_regeneration_attempts
        trace = SessionTrace(session_id=session_id, subject_id=subject_id)
        
        best_identity_score = 0.0
        final_output_path = None
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"[{session_id}] Generation attempt {attempt}/{max_attempts}")
            
            # Stage 2: Generate
            try:
                generated_image, generated_embedding = self._generate_image(
                    subject_id=subject_id,
                    prompt=prompt,
                    anchors=anchors,
                    seed=seed
                )
            except Exception as e:
                logger.error(f"[{session_id}] Generation failed: {e}")
                return GenerationResult(
                    session_id=session_id,
                    subject_id=subject_id,
                    success=False,
                    output_path=None,
                    final_identity_score=None,
                    attempts=attempt,
                    failure_reason=f"GENERATION_ERROR: {e}",
                    trace=trace
                )
            
            # Stage 3: Guardrails
            decision = self.guardrail.evaluate(
                generated_image=generated_image,
                generated_embedding=generated_embedding,
                anchor_embeddings=anchors,
                attempt_number=attempt
            )
            
            # Record in trace
            trace.record_attempt(
                attempt_number=attempt,
                identity_score=decision.identity_score,
                artifact_score=decision.artifact_score,
                decision=decision.action.value
            )
            
            # Track best score for reporting
            if decision.identity_score:
                best_identity_score = max(best_identity_score, decision.identity_score)
            
            # Decision logic
            if decision.action == GuardrailAction.PASS:
                # SUCCESS: Save and return
                final_output_path = self._save_output(generated_image, session_id, subject_id)
                logger.info(f"[{session_id}] Guardrail PASS on attempt {attempt}")
                
                return GenerationResult(
                    session_id=session_id,
                    subject_id=subject_id,
                    success=True,
                    output_path=final_output_path,
                    final_identity_score=decision.identity_score,
                    attempts=attempt,
                    failure_reason=None,
                    trace=trace
                )
            
            elif decision.action == GuardrailAction.HARD_STOP:
                # HARD STOP: Terminal failure
                # CRITICAL: decision.should_retry MUST be False
                assert decision.should_retry == False, (
                    "CRITICAL: HARD_STOP has should_retry=True. Loop invariant violated."
                )
                
                logger.error(
                    f"[{session_id}] HARD_STOP at attempt {attempt}: {decision.reason}"
                )
                
                return GenerationResult(
                    session_id=session_id,
                    subject_id=subject_id,
                    success=False,
                    output_path=None,
                    final_identity_score=best_identity_score,
                    attempts=attempt,
                    failure_reason=decision.reason,
                    trace=trace
                )
            
            else:  # REJECT_AND_RETRY
                logger.info(f"[{session_id}] Guardrail REJECT on attempt {attempt}, retrying...")
                continue
        
        # Should be unreachable — HARD_STOP should fire at max attempts
        raise RuntimeError(
            f"[{session_id}] Generation loop exited without terminal decision. "
            f"This indicates a logic error in guardrail evaluation."
        )
    
    def _generate_image(
        self,
        subject_id: str,
        prompt: str,
        anchors: List[np.ndarray],
        seed: Optional[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: Generate image with identity conditioning.
        
        In production, this would:
        1. Construct augmented prompt (base + identity context)
        2. Inject anchors via IP-Adapter
        3. Run diffusion
        4. Encode output as embedding
        
        Placeholder returns dummy data for demo.
        """
        # In production: actual diffusion pipeline
        generated_image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        generated_embedding = np.random.randn(512).astype(np.float32)
        generated_embedding = generated_embedding / (np.linalg.norm(generated_embedding) + 1e-8)
        
        return generated_image, generated_embedding
    
    def _save_output(self, image: np.ndarray, session_id: str, subject_id: str) -> str:
        """Save generated image to disk."""
        from PIL import Image
        
        output_path = Path("./outputs") / f"{session_id}_{subject_id}.png"
        
        # Convert to PIL and save
        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img.save(output_path)
        
        logger.info(f"Output saved to {output_path}")
        return str(output_path)
    
    def health_check(self) -> bool:
        """Validate system health."""
        logger.info("Running health check...")
        
        checks = {
            'config_valid': self.config.validate(),
            'guardrails_initialized': self.guardrail is not None,
            'tracer_initialized': self.tracer is not None,
        }
        
        all_passed = all(checks.values())
        
        for check, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}")
        
        return all_passed
