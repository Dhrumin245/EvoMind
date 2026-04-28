import asyncio
import csv
import logging
import os
import secrets
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from api.auth import APIKeyPrincipal, api_key_store, require_api_key
from api.events import EventManager
from api.interface import AgentInterface
from api.job_manager import JobControlConflictError, JobManager, JobRecord
from api.logging_utils import configure_logging, merge_log_context, push_log_context, reset_log_context
from api.payments import RazorpayClient
from api.schemas import (
    AgentQuery,
    AgentResponse,
    BillingAccountResponse,
    BillingLedgerEntry,
    BillingLedgerResponse,
    BillingTierInfo,
    BillingTiersResponse,
    BillingTopupConfirmRequest,
    BillingTopupConfirmResponse,
    BillingTopupRequest,
    BillingTopupResponse,
    BatchAgentQuery,
    BatchAgentResponse,
    CheckpointListResponse,
    CreateCheckpointRequest,
    CreateCheckpointResponse,
    GenomeListResponse,
    GenomeSummary,
    GenomeType,
    HealthCheck,
    ReadinessCheck,
    ReadinessComponent,
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


configure_logging(service_name="evomind", component="api")
logger = logging.getLogger(__name__)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_int_env(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return max(minimum, default)
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default=%s", name, raw_value, default)
        return max(minimum, default)


APP_ENV = os.getenv("EVOMIND_ENV", "development").strip().lower() or "development"
_docs_override = os.getenv("EVOMIND_ENABLE_API_DOCS")
if _docs_override is None:
    API_DOCS_ENABLED = APP_ENV != "production"
else:
    API_DOCS_ENABLED = _is_truthy(_docs_override)

_default_cors_origins = "http://localhost:3000,http://127.0.0.1:3000" if APP_ENV != "production" else ""
CORS_ALLOW_ORIGINS = _parse_csv(os.getenv("EVOMIND_CORS_ALLOW_ORIGINS", _default_cors_origins))
CORS_ALLOW_METHODS = _parse_csv(os.getenv("EVOMIND_CORS_ALLOW_METHODS", "GET,POST,DELETE,OPTIONS"))
CORS_ALLOW_HEADERS = _parse_csv(
    os.getenv("EVOMIND_CORS_ALLOW_HEADERS", "Authorization,Content-Type,X-API-Key")
)
CORS_ALLOW_CREDENTIALS = _is_truthy(os.getenv("EVOMIND_CORS_ALLOW_CREDENTIALS", "false"))
if "*" in CORS_ALLOW_ORIGINS and CORS_ALLOW_CREDENTIALS:
    logger.warning("Disabling CORS credentials because wildcard origin is configured")
    CORS_ALLOW_CREDENTIALS = False

MAX_REQUEST_BYTES = _read_int_env("EVOMIND_MAX_REQUEST_BYTES", 1_048_576)
USAGE_LOG_QUEUE_SIZE = _read_int_env("EVOMIND_USAGE_LOG_QUEUE_SIZE", 5000)
USAGE_LOG_DRAIN_TIMEOUT_SECONDS = _read_int_env("EVOMIND_USAGE_LOG_DRAIN_TIMEOUT_SECONDS", 5)
SERVICE_STARTED_AT = time.time()
REQUEST_ID_HEADER = "X-Request-ID"


class RequestBodyTooLargeError(RuntimeError):
    def __init__(self, max_bytes: int):
        self.max_bytes = max(1, int(max_bytes))
        super().__init__(f"Request body too large (max {self.max_bytes} bytes)")


def _resolve_request_id(value: Optional[str]) -> str:
    normalized = str(value or "").strip()
    if normalized and len(normalized) <= 128 and re.fullmatch(r"[A-Za-z0-9._:-]+", normalized):
        return normalized
    return secrets.token_hex(12)

app = FastAPI(
    title="Evomind API",
    description=(
        "API for evolutionary AI training control and agent inference. "
        "Protected endpoints require `X-API-Key: <your-key>` or `Authorization: Bearer <your-key>`. "
        "Each tenant can create isolated jobs with separate state, checkpoints, and metrics. "
        "Tenant request limits, prepaid credits, billing tiers, usage exports, job events, and webhook delivery history are exposed through the API."
    ),
    version="1.7.0",
    docs_url="/docs" if API_DOCS_ENABLED else None,
    redoc_url="/redoc" if API_DOCS_ENABLED else None,
    openapi_url="/openapi.json" if API_DOCS_ENABLED else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS or ["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=CORS_ALLOW_HEADERS or ["Authorization", "Content-Type", "X-API-Key"],
)

job_manager: Optional[JobManager] = None
event_manager: Optional[EventManager] = None
usage_log_queue: Optional[asyncio.Queue[Optional[Dict[str, Any]]]] = None
usage_log_worker_task: Optional[asyncio.Task] = None


def _service_uptime_seconds() -> float:
    return max(0.0, time.time() - SERVICE_STARTED_AT)


def _internal_server_error(log_message: str, exc: Optional[Exception] = None) -> HTTPException:
    if exc is not None:
        logger.exception(log_message)
    else:
        logger.error(log_message)
    return HTTPException(status_code=500, detail="Internal server error")


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[no-any-return]


def _request_body_too_large_response(max_bytes: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body too large (max {max_bytes} bytes)"},
    )


def _estimate_tokens_from_size(payload_size: int) -> int:
    # Approximate tokens from payload size when the API payload is arbitrary JSON
    # rather than model-native text with a tokenizer available.
    normalized_size = max(0, int(payload_size))
    if normalized_size == 0:
        return 0
    return (normalized_size + 3) // 4


def _response_body_size(response: Any) -> int:
    body = getattr(response, "body", b"") or b""
    if isinstance(body, str):
        return len(body.encode("utf-8"))
    if isinstance(body, memoryview):
        return len(body.tobytes())
    return len(body)


def _install_request_body_limit(request: Request, max_bytes: int) -> None:
    original_receive = request._receive
    bytes_read = 0
    request.state.request_body_bytes_for_billing = 0

    async def limited_receive() -> Dict[str, Any]:
        nonlocal bytes_read
        message = await original_receive()
        if message.get("type") != "http.request":
            return dict(message)

        body = message.get("body", b"") or b""
        bytes_read += len(body)
        request.state.request_body_bytes_for_billing += len(body)
        if bytes_read > max_bytes:
            raise RequestBodyTooLargeError(max_bytes)
        return dict(message)

    # Starlette Request reads from this receive hook when parsing the body/stream.
    request._receive = limited_receive


async def _readiness_components() -> list[ReadinessComponent]:
    components: list[ReadinessComponent] = []

    auth_db_healthy = await asyncio.to_thread(api_key_store.is_available)
    components.append(
        ReadinessComponent(
            name="api_auth_db",
            healthy=auth_db_healthy,
            detail="reachable" if auth_db_healthy else "unreachable",
        )
    )

    current_job_manager = job_manager
    runtime_db_healthy = False
    training_worker_count = 0
    if current_job_manager is not None:
        runtime_db_healthy = bool(await asyncio.to_thread(current_job_manager.is_available))
        if runtime_db_healthy:
            training_worker_count = len(
                await asyncio.to_thread(current_job_manager.list_active_workers, "training")
            )
    components.append(
        ReadinessComponent(
            name="job_runtime_db",
            healthy=runtime_db_healthy,
            detail="reachable" if runtime_db_healthy else "unreachable",
        )
    )

    components.append(
        ReadinessComponent(
            name="training_worker",
            healthy=training_worker_count > 0,
            detail=f"{training_worker_count} active worker(s)" if training_worker_count > 0 else "no active workers",
        )
    )

    current_event_manager = event_manager
    events_db_healthy = False
    worker_healthy = False
    if current_event_manager is not None:
        events_db_healthy = bool(await asyncio.to_thread(current_event_manager.is_available))
        worker_healthy = current_event_manager.is_worker_running()

    components.append(
        ReadinessComponent(
            name="api_events_db",
            healthy=events_db_healthy,
            detail="reachable" if events_db_healthy else "unreachable",
        )
    )

    components.append(
        ReadinessComponent(
            name="webhook_worker",
            healthy=worker_healthy,
            detail="running" if worker_healthy else "not running",
        )
    )

    return components


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    token = push_log_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    try:
        logger.exception("Unhandled exception for %s %s", request.method, request.url.path, extra={"event": "request_error"})
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        if request_id:
            response.headers[REQUEST_ID_HEADER] = str(request_id)
        return response
    finally:
        reset_log_context(token)


async def _usage_log_worker_loop() -> None:
    queue = usage_log_queue
    if queue is None:
        return

    while True:
        payload = await queue.get()
        try:
            if payload is None:
                return
            await asyncio.to_thread(api_key_store.log_usage, **payload)
        except Exception:
            logger.exception(
                "Usage logging failed tenant=%s key_id=%s path=%s",
                payload.get("tenant_id") if isinstance(payload, dict) else "<unknown>",
                payload.get("key_id") if isinstance(payload, dict) else "<unknown>",
                payload.get("path") if isinstance(payload, dict) else "<unknown>",
            )
        finally:
            queue.task_done()


def _enqueue_usage_log(payload: Dict[str, Any]) -> None:
    queue = usage_log_queue
    if queue is None:
        logger.warning("Usage log queue unavailable; dropping record for path=%s", payload.get("path"))
        return
    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        logger.error(
            "Usage log queue full; dropping record tenant=%s key_id=%s path=%s",
            payload.get("tenant_id"),
            payload.get("key_id"),
            payload.get("path"),
        )


def _attach_rate_limit_headers(response, rate_limits: Any) -> None:
    if not isinstance(rate_limits, dict):
        return

    def _read_non_negative_int(key: str) -> int:
        raw_value = rate_limits.get(key)
        if raw_value is None:
            return 0
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return 0

    minute_limit = _read_non_negative_int("requests_per_minute")
    minute_used = _read_non_negative_int("minute_count")
    day_limit = _read_non_negative_int("requests_per_day")
    day_used = _read_non_negative_int("day_count")

    response.headers["X-RateLimit-Limit-Minute"] = str(minute_limit)
    response.headers["X-RateLimit-Used-Minute"] = str(minute_used)
    response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, minute_limit - minute_used))
    response.headers["X-RateLimit-Limit-Day"] = str(day_limit)
    response.headers["X-RateLimit-Used-Day"] = str(day_used)
    response.headers["X-RateLimit-Remaining-Day"] = str(max(0, day_limit - day_used))


def _finalize_request_response(request: Request, response, started_at: float):
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    if request_id:
        response.headers[REQUEST_ID_HEADER] = str(request_id)

    _attach_rate_limit_headers(response, getattr(request.state, "rate_limits", None))

    duration_ms = (time.perf_counter() - started_at) * 1000.0
    principal = getattr(request.state, "principal", None)
    job_id = None
    path_params = getattr(request, "path_params", None)
    if isinstance(path_params, dict):
        job_id = path_params.get("job_id")
    route = request.scope.get("route")
    route_template = getattr(route, "path", request.url.path)
    merge_log_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 3),
        route_template=route_template,
        job_id=job_id,
    )
    if principal is not None:
        merge_log_context(
            tenant_id=principal.tenant_id,
            key_id=principal.key_id,
        )
    logger.info("Request completed", extra={"event": "request_completed"})

    if principal is not None:
        request_body_tokens = _estimate_tokens_from_size(
            getattr(request.state, "request_body_bytes_for_billing", 0)
        )
        response_body_tokens = _estimate_tokens_from_size(_response_body_size(response))
        billed_tokens = request_body_tokens + response_body_tokens
        _enqueue_usage_log(
            {
                "tenant_id": principal.tenant_id,
                "key_id": principal.key_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "route_template": route_template,
                "job_id": job_id,
                "billed_tokens": billed_tokens,
            }
        )

    return response


@app.middleware("http")
async def usage_logging_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    request_id = _resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
    request.state.request_id = request_id
    token = push_log_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                content_length_value = int(content_length)
            except ValueError:
                return _finalize_request_response(
                    request,
                    JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"}),
                    started_at,
                )

            if content_length_value > MAX_REQUEST_BYTES:
                return _finalize_request_response(
                    request,
                    _request_body_too_large_response(MAX_REQUEST_BYTES),
                    started_at,
                )

        _install_request_body_limit(request, MAX_REQUEST_BYTES)

        try:
            response = await call_next(request)
        except RequestBodyTooLargeError as exc:
            response = _request_body_too_large_response(exc.max_bytes)

        return _finalize_request_response(request, response, started_at)
    finally:
        reset_log_context(token)


def _require_job_manager() -> JobManager:
    if job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager not ready")
    return job_manager


def _require_event_manager() -> EventManager:
    if event_manager is None:
        raise HTTPException(status_code=503, detail="Event manager not ready")
    return event_manager


def _job_summary(record: JobRecord) -> JobSummary:
    return JobSummary(**record.to_dict())  # type: ignore


def _serialize_train_status(status: TrainStatus) -> Dict[str, Any]:
    if hasattr(status, "model_dump"):
        return status.model_dump()  # type: ignore[no-any-return]
    return status.dict()  # type: ignore[no-any-return]


def _job_control_conflict(job_id: str, exc: JobControlConflictError) -> HTTPException:
    detail = (
        f"Job '{job_id}' is currently controlled by another API instance "
        f"({exc.owner_id}) until {exc.lease_expires_at}"
    )
    return HTTPException(status_code=409, detail=detail)


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
    merge_log_context(job_id=job_id)

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
        raise _internal_server_error("Internal request handling failed", exc=e)

    return updated_record, trainer, agent


async def _get_default_job_components(
    principal: APIKeyPrincipal,
) -> Tuple[JobRecord, EvoTrainer, AgentInterface]:
    manager = _require_job_manager()
    manager.ensure_default_job(principal.tenant_id)
    merge_log_context(job_id="default")
    return await _get_job_components(principal, "default")


def _build_queued_train_status_payload(
    trainer: EvoTrainer,
    queued_at: str,
) -> Dict[str, Any]:
    status_payload = _serialize_train_status(trainer.status())
    status_payload["status"] = "queued"
    status_payload["last_update"] = queued_at
    system_payload = dict(status_payload.get("system", {})) if isinstance(status_payload.get("system"), dict) else {}
    system_payload["status"] = "queued"
    system_payload["last_update"] = queued_at
    status_payload["system"] = system_payload
    return status_payload


async def _queue_training_command(
    manager: JobManager,
    principal: APIKeyPrincipal,
    job_id: str,
    command_type: str,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    active_claim = await asyncio.to_thread(manager.get_runtime_claim, principal.tenant_id, job_id)
    pending_command = await asyncio.to_thread(
        manager.get_pending_command,
        principal.tenant_id,
        job_id,
        [command_type],
    )

    if command_type in {"start", "resume"} and active_claim is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Job '{job_id}' is already running on worker "
                f"{active_claim.owner_id} until {active_claim.lease_expires_at}"
            ),
        )

    if pending_command is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A '{command_type}' command is already queued for job '{job_id}'",
        )

    if command_type == "stop" and active_claim is None:
        return {}, None

    payload: Dict[str, Any] = {}
    if checkpoint_path is not None:
        payload["checkpoint_path"] = checkpoint_path

    command = await asyncio.to_thread(
        manager.enqueue_job_command,
        principal.tenant_id,
        job_id,
        command_type,
        payload,
    )
    return _model_dump(command), command.created_at


def _effective_train_status_payload(
    manager: JobManager,
    principal: APIKeyPrincipal,
    job_id: str,
    trainer: EvoTrainer,
    queued_at: Optional[str] = None,
) -> Dict[str, Any]:
    status_payload = _serialize_train_status(trainer.status())
    status_payload = manager.apply_runtime_status_overlay(principal.tenant_id, job_id, status_payload)
    if queued_at is not None and status_payload.get("status") != "running":
        status_payload = _build_queued_train_status_payload(trainer, queued_at)
    return status_payload


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
        raise _internal_server_error("Internal request handling failed", exc=e)


@app.on_event("startup")
async def startup_event():
    global event_manager, job_manager, usage_log_queue, usage_log_worker_task
    job_manager = JobManager()
    event_manager = EventManager()
    usage_log_queue = asyncio.Queue(maxsize=USAGE_LOG_QUEUE_SIZE)
    usage_log_worker_task = asyncio.create_task(_usage_log_worker_loop())
    await event_manager.start_worker()
    logger.info(
        "Evomind API initialized env=%s docs_enabled=%s cors_origins=%s max_request_bytes=%s usage_log_queue_size=%s",
        APP_ENV,
        API_DOCS_ENABLED,
        CORS_ALLOW_ORIGINS or ["<none>"],
        MAX_REQUEST_BYTES,
        USAGE_LOG_QUEUE_SIZE,
    )


@app.on_event("shutdown")
async def shutdown_event():
    global usage_log_queue, usage_log_worker_task

    queue = usage_log_queue
    task = usage_log_worker_task
    if queue is not None and task is not None:
        try:
            await queue.put(None)
            await asyncio.wait_for(queue.join(), timeout=USAGE_LOG_DRAIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(
                "Usage log drain timed out after %ss; forcing shutdown",
                USAGE_LOG_DRAIN_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Unexpected failure while draining usage log queue")

        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Usage log worker stopped with error")

    usage_log_worker_task = None
    usage_log_queue = None

    current_job_manager = job_manager
    if current_job_manager is not None:
        await current_job_manager.shutdown()

    current_event_manager = event_manager
    if current_event_manager is not None:
        await current_event_manager.stop_worker()


@app.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(
        status="healthy",
        message="Service is alive",
        uptime_seconds=_service_uptime_seconds(),
    )


@app.get("/health/readiness", response_model=ReadinessCheck)
async def readiness_check():
    components = await _readiness_components()
    all_healthy = all(component.healthy for component in components)
    payload = ReadinessCheck(
        status="ready" if all_healthy else "not_ready",
        message=(
            "Service dependencies are ready"
            if all_healthy
            else "Dependencies not ready: "
            + ", ".join(component.name for component in components if not component.healthy)
        ),
        uptime_seconds=_service_uptime_seconds(),
        components=components,
    )
    if all_healthy:
        return payload
    return JSONResponse(status_code=503, content=_model_dump(payload))


@app.post("/jobs", response_model=JobSummary)
async def create_job(
    request: Optional[JobCreateRequest] = None,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    manager = _require_job_manager()
    limits = await asyncio.to_thread(api_key_store.get_tenant_limits, principal.tenant_id)
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


@app.get("/billing/account", response_model=BillingAccountResponse)
async def billing_account(principal: APIKeyPrincipal = Depends(require_api_key)):
    account = await asyncio.to_thread(api_key_store.get_billing_account, principal.tenant_id)
    return BillingAccountResponse(**account)


@app.get("/billing/ledger", response_model=BillingLedgerResponse)
async def billing_ledger(
    principal: APIKeyPrincipal = Depends(require_api_key),
    limit: int = Query(default=100, ge=1, le=1000),
):
    items = await asyncio.to_thread(api_key_store.list_billing_ledger, principal.tenant_id, limit)
    return BillingLedgerResponse(
        tenant_id=principal.tenant_id,
        count=len(items),
        items=[BillingLedgerEntry(**item) for item in items],
    )


@app.post("/billing/topups", response_model=BillingTopupResponse)
async def billing_topup_create(
    request: BillingTopupRequest,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    receipt = f"tp_{secrets.token_hex(8)}"
    notes = {
        "tenant_id": principal.tenant_id,
        "requested_amount_inr": f"{request.amount_inr:.2f}",
    }
    if request.description:
        notes["description"] = request.description
    try:
        order = await asyncio.to_thread(
            RazorpayClient.create_order,
            amount_inr=request.amount_inr,
            receipt=receipt,
            notes=notes,
        )
        topup = await asyncio.to_thread(
            api_key_store.create_topup,
            principal.tenant_id,
            provider="razorpay",
            amount_inr=request.amount_inr,
            provider_order_id=str(order["id"]),
            receipt=receipt,
            description=request.description,
            metadata={
                "razorpay_order_status": order.get("status"),
                "razorpay_amount_subunits": order.get("amount"),
                "notes": order.get("notes") if isinstance(order.get("notes"), dict) else {},
            },
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BillingTopupResponse(
        tenant_id=principal.tenant_id,
        topup_id=topup["topup_id"],
        provider=topup["provider"],
        status=topup["status"],
        amount_inr=topup["amount_inr"],
        currency=topup["currency"],
        receipt=topup["receipt"],
        provider_order_id=topup["provider_order_id"],
        checkout_key_id=RazorpayClient.checkout_key_id(),
        created_at=topup["created_at"],
    )


@app.post("/billing/topups/confirm", response_model=BillingTopupConfirmResponse)
async def billing_topup_confirm(
    request: BillingTopupConfirmRequest,
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    try:
        await asyncio.to_thread(
            RazorpayClient.verify_checkout_signature,
            order_id=request.razorpay_order_id,
            payment_id=request.razorpay_payment_id,
            signature=request.razorpay_signature,
        )
        result = await asyncio.to_thread(
            api_key_store.confirm_topup_payment,
            provider="razorpay",
            provider_order_id=request.razorpay_order_id,
            provider_payment_id=request.razorpay_payment_id,
            expected_tenant_id=principal.tenant_id,
            metadata={"confirmed_via": "checkout_callback"},
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    topup = result["topup"]
    account = result["account"]
    return BillingTopupConfirmResponse(
        tenant_id=principal.tenant_id,
        topup_id=topup["topup_id"],
        provider=topup["provider"],
        status=topup["status"],
        amount_inr=topup["amount_inr"],
        payment_id=topup["provider_payment_id"],
        balance_inr=account["available_credit_inr"],
        credited=bool(result["credited"]),
    )


@app.post("/billing/providers/razorpay/webhook")
async def billing_razorpay_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature")
    if not signature:
        raise HTTPException(status_code=401, detail="Missing Razorpay signature header")
    try:
        await asyncio.to_thread(
            RazorpayClient.verify_webhook_signature,
            body=raw_body,
            signature=signature,
        )
        payload = RazorpayClient.parse_webhook(raw_body)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payment = RazorpayClient.extract_captured_payment(payload)
    if payment is None:
        return {"ok": True, "ignored": True}

    try:
        await asyncio.to_thread(
            api_key_store.confirm_topup_payment,
            provider="razorpay",
            provider_order_id=payment["order_id"],
            provider_payment_id=payment["payment_id"],
            amount_inr=payment["amount_inr"],
            metadata={"confirmed_via": "webhook", "event": payment["event"]},
        )
    except ValueError as e:
        logger.warning("Ignored Razorpay webhook order=%s payment=%s: %s", payment["order_id"], payment["payment_id"], e)
        return {"ok": True, "ignored": True}

    return {"ok": True, "processed": True}


@app.get("/usage/limits", response_model=TenantLimitsResponse)
async def usage_limits(principal: APIKeyPrincipal = Depends(require_api_key)):
    limits = await asyncio.to_thread(api_key_store.get_tenant_limits, principal.tenant_id)
    return TenantLimitsResponse(tenant_id=principal.tenant_id, **limits)


@app.get("/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(principal: APIKeyPrincipal = Depends(require_api_key)):
    summary = await asyncio.to_thread(api_key_store.get_usage_summary, principal.tenant_id)
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
    items = await asyncio.to_thread(
        api_key_store.export_usage,
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
                "billed_tokens",
                "unit_price_inr",
                "estimated_cost_inr",
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
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=job_id,
        command_type="start",
    )
    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        job_id,
        "job.start_requested",
        {
            "queued_at": queued_at,
            "job": record.to_dict(),
        },
    )
    return TrainStatus(**_effective_train_status_payload(manager, principal, job_id, trainer, queued_at))


@app.post("/jobs/{job_id}/train/stop", response_model=TrainStatus)
async def job_train_stop(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=job_id,
        command_type="stop",
    )
    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    if queued_at is not None:
        await _emit_job_event(
            principal,
            job_id,
            "job.stop_requested",
            {
                "queued_at": queued_at,
                "job": record.to_dict(),
            },
        )
    return TrainStatus(**_effective_train_status_payload(manager, principal, job_id, trainer))


@app.get("/jobs/{job_id}/train/status", response_model=TrainStatus)
async def job_train_status(job_id: str, principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    _, trainer, _ = await _get_job_components(principal, job_id)
    manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    return TrainStatus(**_effective_train_status_payload(manager, principal, job_id, trainer))


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
    try:
        checkpoint_path = str(trainer.resolve_checkpoint_path(request.checkpoint_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=job_id,
        command_type="resume",
        checkpoint_path=checkpoint_path,
    )
    record = manager.update_job_status(principal.tenant_id, job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        job_id,
        "job.resume_requested",
        {
            "checkpoint_path": checkpoint_path,
            "queued_at": queued_at,
            "job": record.to_dict(),
        },
    )
    return TrainStatus(**_effective_train_status_payload(manager, principal, job_id, trainer, queued_at))


@app.get("/jobs/{job_id}/train/checkpoints", response_model=CheckpointListResponse)
async def job_list_checkpoints(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_job_components(principal, job_id)
    from api.schemas import CheckpointSummary
    items = [CheckpointSummary(**_checkpoint_summary(item)) for item in trainer.list_checkpoints(limit=limit)]
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
        raise _internal_server_error("Internal request handling failed", exc=e)

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
    checkpoint_summary = _checkpoint_summary(created_checkpoint)
    await _emit_job_event(
        principal,
        job_id,
        "checkpoint.created",
        {
            "checkpoint": checkpoint_summary,
        },
    )
    from api.schemas import CheckpointSummary
    return CreateCheckpointResponse(checkpoint=CheckpointSummary(**checkpoint_summary))


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
        raise _internal_server_error("Internal request handling failed", exc=e)


# Compatibility routes. These now operate on the tenant-scoped "default" job.
@app.post("/train/start", response_model=TrainStatus)
async def train_start(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=record.job_id,
        command_type="start",
    )
    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        record.job_id,
        "job.start_requested",
        {
            "queued_at": queued_at,
            "job": updated_record.to_dict(),
        },
    )
    return TrainStatus(**_effective_train_status_payload(manager, principal, record.job_id, trainer, queued_at))


@app.post("/train/stop", response_model=TrainStatus)
async def train_stop(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=record.job_id,
        command_type="stop",
    )
    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    if queued_at is not None:
        await _emit_job_event(
            principal,
            record.job_id,
            "job.stop_requested",
            {
                "queued_at": queued_at,
                "job": updated_record.to_dict(),
            },
        )
    return TrainStatus(**_effective_train_status_payload(manager, principal, record.job_id, trainer))


@app.get("/train/status", response_model=TrainStatus)
async def train_status(principal: APIKeyPrincipal = Depends(require_api_key)):
    manager = _require_job_manager()
    record, trainer, _ = await _get_default_job_components(principal)
    manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    return TrainStatus(**_effective_train_status_payload(manager, principal, record.job_id, trainer))


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
    try:
        checkpoint_path = str(trainer.resolve_checkpoint_path(request.checkpoint_path))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _, queued_at = await _queue_training_command(
        manager=manager,
        principal=principal,
        job_id=record.job_id,
        command_type="resume",
        checkpoint_path=checkpoint_path,
    )
    updated_record = manager.update_job_status(principal.tenant_id, record.job_id, trainer=trainer)
    await _emit_job_event(
        principal,
        record.job_id,
        "job.resume_requested",
        {
            "checkpoint_path": checkpoint_path,
            "queued_at": queued_at,
            "job": updated_record.to_dict(),
        },
    )
    return TrainStatus(**_effective_train_status_payload(manager, principal, record.job_id, trainer, queued_at))


@app.get("/train/checkpoints", response_model=CheckpointListResponse)
async def list_train_checkpoints(
    limit: int = Query(default=20, ge=1, le=200),
    principal: APIKeyPrincipal = Depends(require_api_key),
):
    _, trainer, _ = await _get_default_job_components(principal)
    from api.schemas import CheckpointSummary
    items = [CheckpointSummary(**_checkpoint_summary(item)) for item in trainer.list_checkpoints(limit=limit)]
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
        raise _internal_server_error("Internal request handling failed", exc=e)

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
    checkpoint_summary = _checkpoint_summary(created_checkpoint)
    await _emit_job_event(
        principal,
        "default",
        "checkpoint.created",
        {
            "checkpoint": checkpoint_summary,
        },
    )
    from api.schemas import CheckpointSummary
    return CreateCheckpointResponse(checkpoint=CheckpointSummary(**checkpoint_summary))


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
        raise _internal_server_error("Internal request handling failed", exc=e)


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
        raise _internal_server_error("Internal request handling failed", exc=e)


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=APP_ENV == "development",
        log_level="info",
    )
