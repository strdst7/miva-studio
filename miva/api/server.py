"""
miva/api/server.py
FastAPI server for MIVA Studio.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import uuid
import logging

from miva.pipeline import MIVAPipeline
from miva.config import get_config

app = FastAPI(title="MIVA Studio API", version="1.0.0")
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    config = get_config()
    pipeline = MIVAPipeline(config)
    logger.info("MIVA Studio API started")

class GenerationRequest(BaseModel):
    subject_id: str
    prompt: str = "professional portrait"
    num_outputs: int = 1
    seed: Optional[int] = None

class GenerationResponse(BaseModel):
    session_id: str
    subject_id: str
    success: bool
    output_path: Optional[str] = None
    identity_score: Optional[float] = None
    attempts: int
    failure_reason: Optional[str] = None

@app.get("/health")
async def health_check():
    if pipeline and pipeline.health_check():
        return {"status": "healthy"}
    return {"status": "unhealthy"}

@app.post("/generate", response_model=List[GenerationResponse])
async def generate(request: GenerationRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        results = pipeline.generate(
            subject_id=request.subject_id,
            prompt=request.prompt,
            num_outputs=request.num_outputs,
            seed=request.seed
        )
        
        response = []
        for r in results:
            response.append(GenerationResponse(
                session_id=r.session_id,
                subject_id=r.subject_id,
                success=r.success,
                output_path=r.output_path,
                identity_score=r.final_identity_score,
                attempts=r.attempts,
                failure_reason=r.failure_reason
            ))
        return response
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
