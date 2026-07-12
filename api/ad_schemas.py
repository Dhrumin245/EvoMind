"""
Phase 6 -- API schemas for the ad campaign endpoints.

Mirrors the SHAPE of EvoMind's real api/schemas.py (TrainStatusEnum,
TrainStatus, TrainRequest) rather than copying its fields verbatim -- those
fields are genuinely NN-training-specific (FitnessPair with prey/predator,
NeuralHealthStatus, curriculum_stage). What's worth reusing is the pattern:
a status enum + a generation counter + a resume-from-checkpoint request.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CampaignStatusEnum(str, Enum):
    """Same states as EvoMind's TrainStatusEnum (QUEUED/RUNNING/PAUSED/
    STOPPED/ERROR) -- these state semantics are genuinely domain-agnostic,
    unlike the fitness/health fields elsewhere in the original schema."""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class CampaignConfig(BaseModel):
    """What a person configures when launching a campaign. Deliberately
    small -- this is the piece that replaces EvoMind's NN hyperparameters
    (layer sizes, learning rates) with ad-campaign concepts."""
    name: str = Field(..., description="Human-readable campaign name")
    population_size: int = Field(40, ge=10, le=200)
    total_generations: int = Field(30, ge=1, le=500)
    budget_per_generation: int = Field(6000, ge=100, description="Simulated impressions available per generation")
    rounds_per_generation: int = Field(15, ge=1, le=100, description="Bandit learning rounds within each generation")
    compatibility_threshold: float = Field(0.65, gt=0, description="Speciation distance threshold, tuned in Phase 3")
    mutation_rate: float = Field(0.15, ge=0, le=1)
    mutation_strength: float = Field(0.2, ge=0, le=2)
    architecture_mutation_rate: float = Field(0.1, ge=0, le=1)


class CreativeSummary(BaseModel):
    genome_id: str
    headline: str
    image_style: str
    cta: str
    tone: str
    color_scheme: str
    optional_traits: Dict[str, bool]
    fitness: float


class CampaignStatusResponse(BaseModel):
    campaign_id: str
    tenant_id: str
    name: str
    status: CampaignStatusEnum
    generation: int
    total_generations: int
    species_count: int = 0
    best_fitness: float = 0.0
    best_creative: Optional[CreativeSummary] = None
    total_impressions_served: int = 0
    created_at: str
    updated_at: str


class GenerationHistoryEntry(BaseModel):
    generation: int
    best_fitness: float
    species_count: int
    impressions_this_generation: int


class CampaignListResponse(BaseModel):
    campaigns: List[CampaignStatusResponse]
