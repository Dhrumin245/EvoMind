from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from typing import Optional, Dict, Any
import numpy as np

# Local imports
from api.schemas import (
    TrainStatus, TrainResumeRequest,
    AgentQuery, AgentResponse, GenomeType
)
from api.trainer import EvoTrainer
from api.interface import AgentInterface
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🧬 Evomind API",
    description="API for Evolutionary AI Training Control & Agentic Interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global trainer instance
trainer: Optional[EvoTrainer] = None
agent_interface: Optional[AgentInterface] = None


async def _auto_start_training() -> None:
    if trainer is None:
        return
    # Yield once so startup can complete before heavy background work begins.
    await asyncio.sleep(0)
    start_result = await trainer.start()
    logger.info(f"🧠 Auto-start training result: {start_result.get('status', 'running')}")

@app.on_event("startup")
async def startup_event():
    """Initialize trainer on startup"""
    global trainer, agent_interface
    
    trainer = EvoTrainer()
    init_success = await trainer.initialize()
    
    if init_success:
        agent_interface = AgentInterface(trainer)
        logger.info("🚀 Evomind API initialized successfully")
        asyncio.create_task(_auto_start_training())
    else:
        logger.error("❌ Trainer initialization failed")

@app.post("/train/start", response_model=TrainStatus)
async def train_start():
    """Start NeuroGenesis training in background"""
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not ready")
    
    result = await trainer.start()
    return TrainStatus(**trainer.last_status)

@app.post("/train/stop", response_model=TrainStatus)
async def train_stop():
    """Stop training gracefully"""
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not ready")
    
    result = await trainer.stop()
    return TrainStatus(**trainer.last_status)

@app.get("/train/status", response_model=TrainStatus)
async def train_status():
    """Get current training status & metrics"""
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not ready")
    
    return trainer.status()

@app.get("/train/insights", response_model=Dict[str, Any])
async def train_insights(last_n: int = Query(default=10, ge=1, le=200)):
    """Get fitness/diversity/learning trends for the latest generations."""
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not ready")

    try:
        return trainer.get_insights(last_n=last_n)
    except Exception as e:
        logger.error(f"Insights retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/resume", response_model=TrainStatus)
async def train_resume(request: TrainResumeRequest):
    """Resume training from checkpoint"""
    if not trainer:
        raise HTTPException(status_code=503, detail="Trainer not ready")
    
    result = await trainer.resume(request.checkpoint_path)
    return TrainStatus(**trainer.last_status)

@app.post("/agent/action", response_model=AgentResponse)
async def agent_action(query: AgentQuery):
    """Inference endpoint: observation → action from best evolved genome"""
    if not agent_interface:
        raise HTTPException(status_code=503, detail="Agent interface not ready")
    
    try:
        obs_array = np.array(query.observation, dtype=np.float32)
        result = agent_interface.query(
            observation=obs_array.tolist(),
            genome_type=query.genome_type,
            generation=query.generation
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return AgentResponse(
            action=result["action"],
            genome_id=result["genome_id"],
            genome_fitness=float(result.get("fitness", 0.0)),
            genome_type=result["genome_type"],
            generation=result["generation"],
            confidence=float(result.get("confidence", 0.0)),
        )
    except Exception as e:
        logger.error(f"Agent action failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/info", response_model=Dict[str, Any])
async def agent_info(
    genome_type: GenomeType = Query(..., description="Genome type: prey or predator"),
    generation: Optional[int] = Query(default=None, ge=0),
):
    """Returns metadata about the currently selected agent genome."""
    if not agent_interface:
        raise HTTPException(status_code=503, detail="Agent interface not ready")

    try:
        genome = agent_interface.get_best_genome(genome_type=genome_type, generation=generation)
        if genome is None:
            return {
                "available": False,
                "genome_type": genome_type,
                "generation": generation,
            }

        return {
            "available": True,
            "genome_id": getattr(genome, "genome_id", "unknown"),
            "genome_type": genome_type,
            "fitness": float(getattr(genome, "fitness", 0.0)),
            "generation": int(getattr(genome, "birth_generation", 0)),
            "gene_count": len(getattr(genome, "genes", [])),
        }
    except Exception as e:
        logger.error(f"Agent info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

