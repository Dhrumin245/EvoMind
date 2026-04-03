from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum

class TrainStatusEnum(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class FitnessPair(BaseModel):
    best: float = 0.0
    average: float = 0.0


class FitnessStatus(BaseModel):
    prey: FitnessPair = Field(default_factory=FitnessPair)
    predator: FitnessPair = Field(default_factory=FitnessPair)


class LearningStatus(BaseModel):
    adaptability: float = 0.0
    meta_effectiveness: float = 0.0
    performance_change: float = 0.0
    instability: float = 0.0


class BehaviorStatus(BaseModel):
    success_rate: float = 0.0
    stability: float = 0.0
    novelty: float = 0.0


class DiversityStatus(BaseModel):
    prey_species: int = 0
    predator_species: int = 0


class NeuralHealthStatus(BaseModel):
    dead_connections: int = 0
    saturation: int = 0


class SystemStatus(BaseModel):
    evaluation_time_sec: float = 0.0
    status: TrainStatusEnum = TrainStatusEnum.STOPPED
    uptime_seconds: float = 0.0
    last_update: str = ""

class TrainStatus(BaseModel):
    generation: int = 0
    stage: str = "unknown"
    fitness: FitnessStatus = Field(default_factory=FitnessStatus)
    learning: LearningStatus = Field(default_factory=LearningStatus)
    behavior: BehaviorStatus = Field(default_factory=BehaviorStatus)
    diversity: DiversityStatus = Field(default_factory=DiversityStatus)
    neural_health: NeuralHealthStatus = Field(default_factory=NeuralHealthStatus)
    system: SystemStatus = Field(default_factory=SystemStatus)

    # Backward-compatible flat fields
    status: TrainStatusEnum = TrainStatusEnum.STOPPED
    best_prey_fitness: float = 0.0
    best_predator_fitness: float = 0.0
    mean_prey_fitness: float = 0.0
    mean_predator_fitness: float = 0.0
    curriculum_stage: str = "unknown"
    total_generations_trained: int = 0
    uptime_seconds: float = 0.0
    last_update: str = Field("", description="ISO timestamp")

class TrainRequest(BaseModel):
    resume_from: Optional[str] = None  # Optional checkpoint path

class TrainResumeRequest(BaseModel):
    checkpoint_path: str  # Required for resume

class GenomeType(str, Enum):
    PREY = "prey"
    PREDATOR = "predator"

class AgentQuery(BaseModel):
    observation: List[float] = Field(..., min_length=1, description="Environment observation vector")
    genome_type: GenomeType
    generation: Optional[int] = None  # Specific generation, None for latest best
    max_action_length: Optional[int] = 10

class AgentResponse(BaseModel):
    action: List[float] = Field(..., max_length=10, description="Agent action vector")
    genome_id: str
    genome_fitness: float
    genome_type: GenomeType
    generation: int
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Action confidence based on fitness")

class HealthCheck(BaseModel):
    status: Literal["healthy", "warning", "error"]
    message: str
    uptime_seconds: float

