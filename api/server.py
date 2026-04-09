import csv
import logging
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.auth import APIKeyPrincipal, api_key_store, require_api_key
from api.events import EventManager
from api.interface import AgentInterface
from api.job_manager import JobManager, JobRecord
from api.schemas import (
    AgentQuery,
    AgentResponse,
    BillingTierInfo,
    BillingTiersResponse,
    BatchAgentQuery,
    BatchAgentResponse,
    CheckpointListResponse,
    CreateCheckpointRequest,
    CreateCheckpointResponse,
    GenomeListResponse,
    GenomeSummary,
    GenomeType,
    HealthCheck,
    JobEvent,
    JobEventListResponse,
    JobCreateRequest,
    JobListResponse,
    JobSummary,
    MetricsResponse,
    TenantLimitsResponse,
    TrainResumeRequest,
    TrainStatus,
    UsageExportResponse,
    UsageExportRow,
    UsageSummaryResponse,
    WebhookCreateRequest,
    WebhookDelivery,
    WebhookDeliveryListResponse,
    WebhookDeleteResponse,
    WebhookListResponse,
    WebhookSummary,
)
from api.trainer import EvoTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evomind API",
    description=(
        "API for evolutionary AI training control and agent inference. "
        "Protected endpoints require `X-API-Key: <your-key>` or `Authorization: Bearer <your-key>`. "
        "Each tenant can create isolated jobs with separate state, checkpoints, and metrics. "
        "Tenant request limits, billing tiers, usage exports, job events, and webhook delivery history are exposed through the API."
    ),
    version="1.6.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager: Optional[JobManager] = None
event_manager: Optional[EventManager] = None


@app.middleware("http")
async def usage_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)

    principal = getattr(request.state, "principal", None)
    if principal is not None:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        job_id = None
        path_params = getattr(request, "path_params", None)
        if isinstance(path_params, dict):
            job_id = path_params.get("job_id")
        route = request.scope.get("route")
        route_template = getattr(route, "path", request.url.path)
        billed_units = getattr(request.state, "billable_units", 1)
        api_key_store.log_usage(
            tenant_id=principal.tenant_id,
            key_id=principal.key_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            route_template=route_template,
            job_id=job_id,
            billed_units=billed_units,
        )

    return response


def _require_job_manager() -> JobManager:
    if job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager not ready")
    return job_manager


def _require_event_manager() -> EventManager:
    if event_manager is None:
        raise HTTPException(status_code=503, detail="Event manager not ready")
    return event_manager


def _job_summary(record: JobRecord) -> JobSummary:
    return JobSummary(**record.to_dict())


def _checkpoint_summary(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint_path": checkpoint["checkpoint_path"],
        "generation": int(checkpoint.get("generation", 0) or 0),
        "saved_at_utc": str(checkpoint.get("saved_at_utc", "")),
        "config_path": checkpoint.get("config_path"),
        "experiment_path": checkpoint.get("experiment_path"),
        "metrics_path": checkpoint.get("metrics_path"),
        "marker_exists": bool(checkpoint.get("marker_exists", True)),
    }


async def _get_job_components(
    principal: APIKeyPrincipal,
    job_id: str,
) -> Tuple[JobRecord, EvoTrainer, AgentInterface]:
    manager = _require_job_manager()

    record = manager.get_job(principal.tenant_id, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        trainer = await manager.get_trainer(principal.tenant_id, job_id)
        agent = await manager.get_agent(principal.tenant_id, job_id)
        updated_record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Job load failed for tenant=%s job=%s: %s", principal.tenant_id, job_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    return updated_record, trainer, agent


async def _get_default_job_components(
    principal: APIKeyPrincipal,
) -> Tuple[JobRecord, EvoTrainer, AgentInterface]:
    manager = _require_job_manager()
    manager.ensure_default_job(principal.tenant_id)
    return await _get_job_components(principal, "default")


async def _emit_job_event(
    principal: APIKeyPrincipal,
    job_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    current_event_manager = _require_event_manager()
    await current_event_manager.emit_event(
        tenant_id=principal.tenant_id,
        job_id=job_id,
        event_type=event_type,
        payload=payload,
    )


async def _run_agent_action(
    agent: AgentInterface,
    query: AgentQuery,
) -> AgentResponse:
    try:
        result = agent.query(
            observation=query.observation,
            genome_type=query.genome_type.value,
            generation=query.generation,
            max_action_length=query.max_action_length,
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Agent action failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_agent_batch_action(
    agent: AgentInterface,
    query: BatchAgentQuery,
) -> BatchAgentResponse:
    try:
        result = agent.query_batch(
            observations=query.observations,
            genome_type=query.genome_type.value,
            generation=query.generation,
            max_action_length=query.max_action_length,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return BatchAgentResponse(
            genome_id=result["genome_id"],
            genome_type=result["genome_type"],
            generation=result["generation"],
            genome_fitness=float(result.get("fitness", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            batch_size=int(result.get("batch_size", len(result.get("actions", [])))),
            actions=result["actions"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch agent action failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    global job_manager, event_manager
    job_manager = JobManager()
    event_manager = EventManager()
    await event_manager.start_worker()
    logger.info("Evomind API initialized with job isolation and webhook events")


@app.on_event("shutdown")
async def shutdown_event():
    current_event_manager = event_manager
    if current_event_manager is not None:
        await current_event_manager.stop_worker()


@app.get("/health", response_model=HealthCheck)
async def health_check():
    current_manager = job_manager
    if current_manager is None:
        return HealthCheck(
            status="error",
            message="Job manager has not been initialized",
            uptime_seconds=0.0,
        )

    return HealthCheck(
        status="healthy",
        message="Service is ready",
        uptime_seconds=0.0,
    )


@app.post("/jobs", response_model=JobSummary)
async def create_job(
    request: Optional[JobCreateRequest] = None,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    manager = _require_job_manager()
    limits = api_key_store.get_tenant_limits(principal.tenant_id)
    existing_jobs = manager.list_jobs(principal.tenant_id)

    if len(existing_jobs) >= limits["max_jobs"]:
        raise HTTPException(
            status_code=403,
            detail="Maximum job quota reached for this tenant",
        )

    try:
        record = manager.create_job(
            tenant_id=principal.tenant_id,
            name=request.name if request else None,
            job_id=request.job_id if request else None,
        )
        await _emit_job_event(
            principal,
            record.job_id,
            "job.created",
            {
                "job": record.to_dict(),
            },
        )
        return _job_summary(record)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/usage/limits", response_model=TenantLimitsResponse)
async def usage_limits(principal: APIKeyPrincipal = Depends(require_api_key)):
    limits = api_key_store.get_tenant_limits(principal.tenant_id)
    return TenantLimitsResponse(tenant_id=principal.tenant_id, **limits)


@app.get("/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(principal: APIKeyPrincipal = Depends(require_api_key)):
    summary = api_key_store.get_usage_summary(principal.tenant_id)
    return UsageSummaryResponse(tenant_id=principal.tenant_id, **summary)


@app.get("/usage/billing-tiers", response_model=BillingTiersResponse)
async def usage_billing_tiers(principal: APIKeyPrincipal = Depends(require_api_key)):
    items = [BillingTierInfo(**item) for item in api_key_store.get_billing_catalog()]
    return BillingTiersResponse(count=len(items), items=items)


@app.get("/usage/export", response_model=UsageExportResponse)
async def usage_export(
    principal: APIKeyPrincipal = Depends(require_api_key),
    days: int = Query(default=7, ge=1, le=365),
    limit: int = Query(default=5000, ge=1, le=50000),
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    items = api_key_store.export_usage(
        tenant_id=principal.tenant_id,
        days=days,
        limit=limit,
    )

    if format == "csv":
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "created_at",
                "tenant_id",
                "key_id",
                "method",
                "path",
                "route_template",
                "status_code",
                "duration_ms",
                "job_id",
                "billing_tier",
                "billed_units",
                "unit_price_usd",
                "estimated_cost_usd",
            ],
        )
        writer.writeheader()
        for item in items:
            writer.writerow(item)

        return PlainTextResponse(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{principal.tenant_id}-usage-export.csv"'
            },
        )

    return UsageExportResponse(
        tenant_id=principal.tenant_id,
        count=len(items),
        items=[UsageExportRow(**item) for item in items],
    )


@app.post("/webhooks", response_model=WebhookSummary)
async def create_webhook(
    request: WebhookCreateRequest,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    current_event_manager = _require_event_manager()
    try:
        webhook = current_event_manager.create_webhook(
            tenant_id=principal.tenant_id,
            url=request.url,
            description=request.description,
            subscribed_events=request.subscribed_events,
            secret=request.secret,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return WebhookSummary(**webhook.to_dict())


@app.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(principal: APIKeyPrincipal = Depends(require_api_key)):
    current_event_manager = _require_event_manager()
    items = [WebhookSummary(**item.to_dict()) for item in current_event_manager.list_webhooks(principal.tenant_id)]
    return WebhookListResponse(count=len(items), items=items)


@app.delete("/webhooks/{webhook_id}", response_model=WebhookDeleteResponse)
async def delete_webhook(
    webhook_id: str,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    current_event_manager = _require_event_manager()
    deleted = current_event_manager.delete_webhook(principal.tenant_id, webhook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return WebhookDeleteResponse(webhook_id=webhook_id, deleted=True)


@app.get("/webhooks/{webhook_id}/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    webhook_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[str] = Query(default=None),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    current_event_manager = _require_event_manager()
    webhook_ids = {item.webhook_id for item in current_event_manager.list_webhooks(principal.tenant_id)}
    if webhook_id not in webhook_ids:
        raise HTTPException(status_code=404, detail="Webhook not found")

    items = [
        WebhookDelivery(**item.to_dict())
        for item in current_event_manager.list_deliveries(
            tenant_id=principal.tenant_id,
            webhook_id=webhook_id,
            limit=limit,
            status=status,
        )
    ]
    return WebhookDeliveryListResponse(count=len(items), items=items)


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    items = [_job_summary(item) for item in manager.list_jobs(principal.tenant_id)]
    return JobListResponse(count=len(items), items=items)


@app.get("/jobs/{job_id}", response_model=JobSummary)
async def get_job(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    record, _, _ = await _get_job_components(principal, job_id)
    return _job_summary(record)


@app.get("/jobs/{job_id}/events", response_model=JobEventListResponse)
async def list_job_events(
    job_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    event_type: Optional[str] = Query(default=None),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    current_event_manager = _require_event_manager()
    manager = _require_job_manager()
    record = manager.get_job(principal.tenant_id, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    items = [
        JobEvent(**item.to_dict())
        for item in current_event_manager.list_events(
            tenant_id=principal.tenant_id,
            job_id=job_id,
            limit=limit,
            event_type=event_type,
        )
    ]
    return JobEventListResponse(count=len(items), items=items)


@app.post("/jobs/{job_id}/train/start", response_model=TrainStatus)
async def job_train_start(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    await trainer.start()
    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        job_id,
        "job.started",
        {
            "status": trainer.last_status,
            "job": record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.post("/jobs/{job_id}/train/stop", response_model=TrainStatus)
async def job_train_stop(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    await trainer.stop()
    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        job_id,
        "job.stopped",
        {
            "status": trainer.last_status,
            "job": record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.get("/jobs/{job_id}/train/status", response_model=TrainStatus)
async def job_train_status(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    return trainer.status()


@app.get("/jobs/{job_id}/train/insights", response_model=Dict[str, Any])
async def job_train_insights(
    job_id: str,
    last_n: int = Query(default=10, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_job_components(principal, job_id)
    try:
        return trainer.get_insights(last_n=last_n)
    except Exception as e:
        logger.error("Insights retrieval failed for job=%s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/train/metrics", response_model=MetricsResponse)
async def job_train_metrics(
    job_id: str,
    limit: Optional[int] = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    since_generation: Optional[int] = Query(default=None, ge=0),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_job_components(principal, job_id)
    return MetricsResponse(**trainer.get_metrics_payload(
        limit=limit,
        offset=offset,
        since_generation=since_generation,
    ))


@app.post("/jobs/{job_id}/train/resume", response_model=TrainStatus)
async def job_train_resume(
    job_id: str,
    request: TrainResumeRequest,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    result = await trainer.resume(request.checkpoint_path)

    if result.get("status") == "resume_failed":
        raise HTTPException(status_code=400, detail=result.get("error", "Resume failed"))

    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        job_id,
        "job.resumed",
        {
            "checkpoint_path": request.checkpoint_path,
            "status": trainer.last_status,
            "job": record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.get("/jobs/{job_id}/train/checkpoints", response_model=CheckpointListResponse)
async def job_list_checkpoints(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_job_components(principal, job_id)
    items = [_checkpoint_summary(item) for item in trainer.list_checkpoints(limit=limit)]
    return CheckpointListResponse(count=len(items), items=items)


@app.post("/jobs/{job_id}/train/checkpoints", response_model=CreateCheckpointResponse)
async def job_create_checkpoint(
    job_id: str,
    request: Optional[CreateCheckpointRequest] = None,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_job_components(principal, job_id)
    requested_path: Optional[str] = None

    if request and request.path:
        candidate_path = Path(request.path)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise HTTPException(status_code=400, detail="Custom checkpoint path must stay within the job checkpoint directory")
        requested_path = str(trainer.checkpoint_dir / candidate_path)

    try:
        checkpoint_path = await trainer.save_checkpoint(path=requested_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Checkpoint creation failed for job=%s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    checkpoint_items = trainer.list_checkpoints(limit=None)
    created_checkpoint = next(
        (item for item in checkpoint_items if item["checkpoint_path"] == checkpoint_path),
        {
            "checkpoint_path": checkpoint_path,
            "generation": int(getattr(trainer.state, "generation", 0) or 0),
            "saved_at_utc": "",
            "config_path": None,
            "experiment_path": None,
            "metrics_path": None,
            "marker_exists": True,
        },
    )
    await _emit_job_event(
        principal,
        job_id,
        "checkpoint.created",
        {
            "checkpoint": _checkpoint_summary(created_checkpoint),
        },
    )
    return CreateCheckpointResponse(checkpoint=_checkpoint_summary(created_checkpoint))


@app.post("/jobs/{job_id}/agent/action", response_model=AgentResponse)
async def job_agent_action(
    job_id: str,
    query: AgentQuery,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_job_components(principal, job_id)
    return await _run_agent_action(agent, query)


@app.post("/jobs/{job_id}/agent/action/batch", response_model=BatchAgentResponse)
async def job_agent_action_batch(
    request: Request,
    job_id: str,
    query: BatchAgentQuery,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    request.state.billable_units = len(query.observations)
    _, _, agent = await _get_job_components(principal, job_id)
    return await _run_agent_batch_action(agent, query)


@app.get("/jobs/{job_id}/agent/info", response_model=Dict[str, Any])
async def job_agent_info(
    job_id: str,
    genome_type: GenomeType = Query(..., description="Genome type: prey or predator"),
    generation: Optional[int] = Query(default=None, ge=0),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_job_components(principal, job_id)

    try:
        genome = agent.get_best_genome(genome_type=genome_type.value, generation=generation)
        if genome is None:
            return {
                "available": False,
                "genome_type": genome_type,
                "generation": generation,
            }

        summary = agent._genome_summary(genome, genome_type.value, "selected_best")
        return {"available": True, **summary}
    except Exception as e:
        logger.error("Agent info failed for job=%s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/genomes", response_model=GenomeListResponse)
async def job_list_genomes(
    job_id: str,
    genome_type: Optional[GenomeType] = Query(default=None),
    limit_per_type: int = Query(default=10, ge=1, le=100),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_job_components(principal, job_id)

    try:
        items = agent.list_available_genomes(
            genome_type=genome_type.value if genome_type is not None else None,
            limit_per_type=limit_per_type,
        )
        return GenomeListResponse(
            count=len(items),
            items=[GenomeSummary(**item) for item in items],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Genome listing failed for job=%s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/genomes/{genome_id}", response_model=GenomeSummary)
async def job_get_genome(
    job_id: str,
    genome_id: str,
    genome_type: Optional[GenomeType] = Query(default=None),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_job_components(principal, job_id)

    try:
        result = agent.get_genome_by_id(
            genome_id=genome_id,
            genome_type=genome_type.value if genome_type is not None else None,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Genome not found")

        return GenomeSummary(**result["summary"])
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Genome lookup failed for job=%s: %s", job_id, e)
        raise HTTPException(status_code=500, detail=str(e))


# Compatibility routes. These now operate on the tenant-scoped "default" job.
@app.post("/train/start", response_model=TrainStatus)
async def train_start(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    await trainer.start()
    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        record.job_id,
        "job.started",
        {
            "status": trainer.last_status,
            "job": updated_record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.post("/train/stop", response_model=TrainStatus)
async def train_stop(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    await trainer.stop()
    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        record.job_id,
        "job.stopped",
        {
            "status": trainer.last_status,
            "job": updated_record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.get("/train/status", response_model=TrainStatus)
async def train_status(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    return trainer.status()


@app.get("/train/insights", response_model=Dict[str, Any])
async def train_insights(
    last_n: int = Query(default=10, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_default_job_components(principal)
    try:
        return trainer.get_insights(last_n=last_n)
    except Exception as e:
        logger.error("Insights retrieval failed on default job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/metrics", response_model=MetricsResponse)
async def train_metrics(
    limit: Optional[int] = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    since_generation: Optional[int] = Query(default=None, ge=0),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_default_job_components(principal)
    return MetricsResponse(**trainer.get_metrics_payload(
        limit=limit,
        offset=offset,
        since_generation=since_generation,
    ))


@app.post("/train/resume", response_model=TrainStatus)
async def train_resume(
    request: TrainResumeRequest,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    result = await trainer.resume(request.checkpoint_path)

    if result.get("status") == "resume_failed":
        raise HTTPException(status_code=400, detail=result.get("error", "Resume failed"))

    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        record.job_id,
        "job.resumed",
        {
            "checkpoint_path": request.checkpoint_path,
            "status": trainer.last_status,
            "job": updated_record.to_dict(),
        },
    )
    return TrainStatus(**trainer.last_status)


@app.get("/train/checkpoints", response_model=CheckpointListResponse)
async def list_train_checkpoints(
    limit: int = Query(default=20, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_default_job_components(principal)
    items = [_checkpoint_summary(item) for item in trainer.list_checkpoints(limit=limit)]
    return CheckpointListResponse(count=len(items), items=items)


@app.post("/train/checkpoints", response_model=CreateCheckpointResponse)
async def create_train_checkpoint(
    request: Optional[CreateCheckpointRequest] = None,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_default_job_components(principal)
    requested_path: Optional[str] = None

    if request and request.path:
        candidate_path = Path(request.path)
        if candidate_path.is_absolute() or ".." in candidate_path.parts:
            raise HTTPException(status_code=400, detail="Custom checkpoint path must stay within the job checkpoint directory")
        requested_path = str(trainer.checkpoint_dir / candidate_path)

    try:
        checkpoint_path = await trainer.save_checkpoint(path=requested_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Checkpoint creation failed on default job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    checkpoint_items = trainer.list_checkpoints(limit=None)
    created_checkpoint = next(
        (item for item in checkpoint_items if item["checkpoint_path"] == checkpoint_path),
        {
            "checkpoint_path": checkpoint_path,
            "generation": int(getattr(trainer.state, "generation", 0) or 0),
            "saved_at_utc": "",
            "config_path": None,
            "experiment_path": None,
            "metrics_path": None,
            "marker_exists": True,
        },
    )
    await _emit_job_event(
        principal,
        "default",
        "checkpoint.created",
        {
            "checkpoint": _checkpoint_summary(created_checkpoint),
        },
    )
    return CreateCheckpointResponse(checkpoint=_checkpoint_summary(created_checkpoint))


@app.post("/agent/action", response_model=AgentResponse)
async def agent_action(
    query: AgentQuery,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_default_job_components(principal)
    return await _run_agent_action(agent, query)


@app.post("/agent/action/batch", response_model=BatchAgentResponse)
async def agent_action_batch(
    request: Request,
    query: BatchAgentQuery,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    request.state.billable_units = len(query.observations)
    _, _, agent = await _get_default_job_components(principal)
    return await _run_agent_batch_action(agent, query)


@app.get("/agent/info", response_model=Dict[str, Any])
async def agent_info(
    genome_type: GenomeType = Query(..., description="Genome type: prey or predator"),
    generation: Optional[int] = Query(default=None, ge=0),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_default_job_components(principal)

    try:
        genome = agent.get_best_genome(genome_type=genome_type.value, generation=generation)
        if genome is None:
            return {
                "available": False,
                "genome_type": genome_type,
                "generation": generation,
            }

        summary = agent._genome_summary(genome, genome_type.value, "selected_best")
        return {"available": True, **summary}
    except Exception as e:
        logger.error("Agent info failed on default job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genomes", response_model=GenomeListResponse)
async def list_genomes(
    genome_type: Optional[GenomeType] = Query(default=None),
    limit_per_type: int = Query(default=10, ge=1, le=100),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_default_job_components(principal)

    try:
        items = agent.list_available_genomes(
            genome_type=genome_type.value if genome_type is not None else None,
            limit_per_type=limit_per_type,
        )
        return GenomeListResponse(
            count=len(items),
            items=[GenomeSummary(**item) for item in items],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Genome listing failed on default job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genomes/{genome_id}", response_model=GenomeSummary)
async def get_genome(
    genome_id: str,
    genome_type: Optional[GenomeType] = Query(default=None),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, _, agent = await _get_default_job_components(principal)

    try:
        result = agent.get_genome_by_id(
            genome_id=genome_id,
            genome_type=genome_type.value if genome_type is not None else None,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Genome not found")

        return GenomeSummary(**result["summary"])
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Genome lookup failed on default job: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
