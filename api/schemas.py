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

class BatchAgentQuery(BaseModel):
    observations: List[List[float]] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Batch of environment observation vectors",
    )
    genome_type: GenomeType
    generation: Optional[int] = None
    max_action_length: Optional[int] = Field(10, ge=1, le=10)

class BatchAgentResponse(BaseModel):
    genome_id: str
    genome_type: GenomeType
    generation: int
    genome_fitness: float
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    batch_size: int
    actions: List[List[float]]

class GenomeSummary(BaseModel):
    genome_id: str
    genome_type: GenomeType
    fitness: float
    generation: int
    source: str
    gene_count: int = 0
    input_size: int = 0
    output_size: int = 0
    architecture: Optional[str] = None

class GenomeListResponse(BaseModel):
    count: int
    items: List[GenomeSummary]

class CheckpointSummary(BaseModel):
    checkpoint_path: str
    generation: int
    saved_at_utc: str = ""
    config_path: Optional[str] = None
    experiment_path: Optional[str] = None
    metrics_path: Optional[str] = None
    marker_exists: bool = True

class CheckpointListResponse(BaseModel):
    count: int
    items: List[CheckpointSummary]

class CreateCheckpointRequest(BaseModel):
    path: Optional[str] = Field(
        default=None,
        description="Optional custom checkpoint marker path",
    )

class CreateCheckpointResponse(BaseModel):
    checkpoint: CheckpointSummary

class MetricsResponse(BaseModel):
    source: str
    total: int
    count: int
    limit: Optional[int] = None
    offset: int = 0
    since_generation: Optional[int] = None
    items: List[Dict[str, Any]]

class JobCreateRequest(BaseModel):
    job_id: Optional[str] = Field(
        default=None,
        description="Optional custom job identifier",
    )
    name: Optional[str] = Field(
        default=None,
        description="Friendly display name for the job",
    )

class JobSummary(BaseModel):
    job_id: str
    tenant_id: str
    name: str
    base_dir: str
    created_at: str
    updated_at: str
    status: str
    generation: int = 0

class JobListResponse(BaseModel):
    count: int
    items: List[JobSummary]

class TenantLimitsResponse(BaseModel):
    tenant_id: str
    requests_per_minute: int
    requests_per_day: int
    max_jobs: int

class UsageSummaryResponse(BaseModel):
    tenant_id: str
    requests_last_minute: int
    requests_last_day: int
    requests_total: int
    requests_per_minute_limit: int
    requests_per_day_limit: int
    max_jobs: int
    remaining_this_minute: int
    remaining_today: int
    estimated_cost_last_day_usd: float = 0.0
    estimated_cost_total_usd: float = 0.0

class BillingTierInfo(BaseModel):
    method: str
    route_template: str
    billing_tier: str
    unit_name: str
    unit_price_usd: float
    description: str

class BillingTiersResponse(BaseModel):
    count: int
    items: List[BillingTierInfo]

class UsageExportRow(BaseModel):
    created_at: str
    tenant_id: str
    key_id: str
    method: str
    path: str
    route_template: Optional[str] = None
    status_code: int
    duration_ms: float
    job_id: Optional[str] = None
    billing_tier: str
    billed_units: int
    unit_price_usd: float
    estimated_cost_usd: float

class UsageExportResponse(BaseModel):
    tenant_id: str
    count: int
    items: List[UsageExportRow]

class JobEvent(BaseModel):
    event_id: str
    tenant_id: str
    job_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: str

class JobEventListResponse(BaseModel):
    count: int
    items: List[JobEvent]

class WebhookCreateRequest(BaseModel):
    url: str = Field(..., min_length=8, description="Webhook target URL")
    description: Optional[str] = Field(default=None, description="Optional display name")
    subscribed_events: List[str] = Field(
        default_factory=list,
        description="Optional event filter list. Empty means all job events.",
    )
    secret: Optional[str] = Field(
        default=None,
        description="Optional signing secret for X-Evomind-Signature",
    )

class WebhookSummary(BaseModel):
    webhook_id: str
    tenant_id: str
    url: str
    description: str
    subscribed_events: List[str]
    status: str
    created_at: str
    updated_at: str
    last_delivery_at: Optional[str] = None
    last_delivery_status: Optional[str] = None
    last_delivery_error: Optional[str] = None

class WebhookListResponse(BaseModel):
    count: int
    items: List[WebhookSummary]

class WebhookDeleteResponse(BaseModel):
    webhook_id: str
    deleted: bool

class WebhookDeliveryAttempt(BaseModel):
    attempt_id: str
    delivery_id: str
    attempt_number: int
    status: str
    response_status_code: Optional[int] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class WebhookDelivery(BaseModel):
    delivery_id: str
    webhook_id: str
    event_id: str
    tenant_id: str
    job_id: str
    event_type: str
    status: str
    attempt_count: int
    max_attempts: int
    next_retry_at: Optional[str] = None
    delivered_at: Optional[str] = None
    last_error: Optional[str] = None
    created_at: str
    updated_at: str
    attempts: List[WebhookDeliveryAttempt] = Field(default_factory=list)

class WebhookDeliveryListResponse(BaseModel):
    count: int
    items: List[WebhookDelivery]

class HealthCheck(BaseModel):
    status: Literal["healthy", "warning", "error"]
    message: str
    uptime_seconds: float

