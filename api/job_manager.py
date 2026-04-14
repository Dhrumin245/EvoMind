import asyncio
import json
import logging
import re
import secrets
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api.interface import AgentInterface
from api.storage import api_jobs_db_path, connect_database, resolve_db_target, tenant_root_dir
from api.trainer import EvoTrainer


DEFAULT_JOB_LEASE_SECONDS = 30
DEFAULT_JOB_HEARTBEAT_INTERVAL_SECONDS = 10
DEFAULT_WORKER_HEARTBEAT_TTL_SECONDS = 30

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add_seconds(iso_timestamp: str, seconds: int) -> str:
    base = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return _iso_from_timestamp(base.timestamp() + max(1, int(seconds)))


def _iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_timestamp(iso_timestamp: str) -> float:
    return datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _sanitize_identifier(value: str, field_name: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{field_name} must contain only letters, numbers, '_' or '-'")
    return value


def _sanitize_worker_id(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("worker_id must be a non-empty string")
    return normalized


@dataclass
class JobRecord:
    job_id: str
    tenant_id: str
    name: str
    base_dir: str
    created_at: str
    updated_at: str
    status: str
    generation: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "base_dir": self.base_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "generation": self.generation,
        }


@dataclass
class JobRuntimeClaim:
    tenant_id: str
    job_id: str
    owner_id: str
    lease_expires_at: str
    last_heartbeat_at: str
    acquired_at: str
    updated_at: str


@dataclass
class JobCommandRecord:
    command_id: str
    tenant_id: str
    job_id: str
    command_type: str
    payload: Dict[str, Any]
    status: str
    worker_id: Optional[str]
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    updated_at: str


@dataclass
class JobRuntimeStatusRecord:
    tenant_id: str
    job_id: str
    worker_id: Optional[str]
    status_payload: Dict[str, Any]
    last_error: Optional[str]
    active_command_id: Optional[str]
    command_type: Optional[str]
    updated_at: str


@dataclass
class WorkerHeartbeatRecord:
    worker_id: str
    worker_type: str
    lease_expires_at: str
    last_heartbeat_at: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class JobControlConflictError(RuntimeError):
    def __init__(self, tenant_id: str, job_id: str, owner_id: str, lease_expires_at: str):
        self.tenant_id = tenant_id
        self.job_id = job_id
        self.owner_id = owner_id
        self.lease_expires_at = lease_expires_at
        super().__init__(
            f"Job '{job_id}' is currently controlled by another API instance "
            f"({owner_id}) until {lease_expires_at}"
        )


class JobManager:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        runtime_db_path: Optional[str] = None,
        runtime_db_url: Optional[str] = None,
        instance_id: Optional[str] = None,
        lease_seconds: int = DEFAULT_JOB_LEASE_SECONDS,
        heartbeat_interval_seconds: int = DEFAULT_JOB_HEARTBEAT_INTERVAL_SECONDS,
    ):
        self.root_dir = Path(root_dir) if root_dir is not None else tenant_root_dir()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_db_target = resolve_db_target(
            context="API jobs",
            explicit_path=Path(runtime_db_path) if runtime_db_path is not None else None,
            explicit_url=runtime_db_url,
            env_url_names=("EVOMIND_API_JOBS_DB_URL",),
            default_path=api_jobs_db_path(),
        )
        self.runtime_db_path = self.runtime_db_target.path
        self.runtime_db_url = self.runtime_db_target.url
        self.runtime_db_backend = self.runtime_db_target.backend
        self.instance_id = instance_id or self._build_instance_id()
        self.lease_seconds = max(5, int(lease_seconds))
        self.heartbeat_interval_seconds = max(1, int(heartbeat_interval_seconds))
        self._trainer_cache: Dict[Tuple[str, str], EvoTrainer] = {}
        self._agent_cache: Dict[Tuple[str, str], AgentInterface] = {}
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._lease_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._init_runtime_db()

    @staticmethod
    def _build_instance_id() -> str:
        return f"{socket.gethostname()}:{secrets.token_hex(6)}"

    def _runtime_connect(self):
        return connect_database(self.runtime_db_target, timeout=30.0)

    def _init_runtime_db(self) -> None:
        with self._runtime_connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runtime_claims (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, job_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_runtime_claims_lease
                ON job_runtime_claims (lease_expires_at, updated_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_commands (
                    command_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    command_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    worker_id TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_commands_status_created
                ON job_commands (status, created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runtime_status (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    worker_id TEXT,
                    status_payload_json TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    active_command_id TEXT,
                    command_type TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, job_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_workers (
                    worker_id TEXT NOT NULL,
                    worker_type TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (worker_id, worker_type)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runtime_workers_type_lease
                ON runtime_workers (worker_type, lease_expires_at, updated_at)
                """
            )
            conn.commit()

    def _tenant_dir(self, tenant_id: str) -> Path:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        return self.root_dir / tenant / "jobs"

    def _job_dir(self, tenant_id: str, job_id: str) -> Path:
        job = _sanitize_identifier(job_id, "job_id")
        return self._tenant_dir(tenant_id) / job

    def _job_file(self, tenant_id: str, job_id: str) -> Path:
        return self._job_dir(tenant_id, job_id) / "job.json"

    def _lock_for(self, tenant_id: str, job_id: str) -> asyncio.Lock:
        key = (tenant_id, job_id)
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    @staticmethod
    def _row_to_runtime_claim(row: Any) -> JobRuntimeClaim:
        return JobRuntimeClaim(
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            owner_id=str(row["owner_id"]),
            lease_expires_at=str(row["lease_expires_at"]),
            last_heartbeat_at=str(row["last_heartbeat_at"]),
            acquired_at=str(row["acquired_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_command(row: Any) -> JobCommandRecord:
        payload_raw = row["payload_json"] or "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return JobCommandRecord(
            command_id=str(row["command_id"]),
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            command_type=str(row["command_type"]),
            payload=payload,
            status=str(row["status"]),
            worker_id=row["worker_id"],
            error_message=row["error_message"],
            created_at=str(row["created_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_runtime_status(row: Any) -> JobRuntimeStatusRecord:
        payload_raw = row["status_payload_json"] or "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return JobRuntimeStatusRecord(
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            worker_id=row["worker_id"],
            status_payload=payload,
            last_error=row["last_error"],
            active_command_id=row["active_command_id"],
            command_type=row["command_type"],
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _row_to_worker_heartbeat(row: Any) -> WorkerHeartbeatRecord:
        metadata_raw = row["metadata_json"] or "{}"
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        return WorkerHeartbeatRecord(
            worker_id=str(row["worker_id"]),
            worker_type=str(row["worker_type"]),
            lease_expires_at=str(row["lease_expires_at"]),
            last_heartbeat_at=str(row["last_heartbeat_at"]),
            metadata=metadata,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def is_available(self) -> bool:
        try:
            with self._runtime_connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def get_runtime_claim(self, tenant_id: str, job_id: str) -> Optional[JobRuntimeClaim]:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        now = _utc_now()
        with self._runtime_connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM job_runtime_claims
                WHERE tenant_id = ? AND job_id = ? AND lease_expires_at > ?
                """,
                (tenant, job, now),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_runtime_claim(row)

    def get_runtime_status(self, tenant_id: str, job_id: str) -> Optional[JobRuntimeStatusRecord]:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        with self._runtime_connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM job_runtime_status
                WHERE tenant_id = ? AND job_id = ?
                """,
                (tenant, job),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_runtime_status(row)

    def upsert_runtime_status(
        self,
        tenant_id: str,
        job_id: str,
        status_payload: Dict[str, Any],
        worker_id: Optional[str] = None,
        active_command_id: Optional[str] = None,
        command_type: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> JobRuntimeStatusRecord:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        now = _utc_now()
        with self._runtime_connect() as conn:
            conn.execute(
                """
                INSERT INTO job_runtime_status (
                    tenant_id,
                    job_id,
                    worker_id,
                    status_payload_json,
                    last_error,
                    active_command_id,
                    command_type,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, job_id) DO UPDATE SET
                    worker_id = excluded.worker_id,
                    status_payload_json = excluded.status_payload_json,
                    last_error = excluded.last_error,
                    active_command_id = excluded.active_command_id,
                    command_type = excluded.command_type,
                    updated_at = excluded.updated_at
                """,
                (
                    tenant,
                    job,
                    worker_id,
                    json.dumps(status_payload, default=str),
                    last_error,
                    active_command_id,
                    command_type,
                    now,
                ),
            )
            conn.commit()
        status = self.get_runtime_status(tenant, job)
        if status is None:
            raise RuntimeError(f"Failed to persist runtime status for job '{job}'")
        return status

    def record_worker_heartbeat(
        self,
        worker_id: str,
        worker_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        lease_seconds: int = DEFAULT_WORKER_HEARTBEAT_TTL_SECONDS,
    ) -> WorkerHeartbeatRecord:
        normalized_worker_id = _sanitize_worker_id(worker_id)
        normalized_worker_type = _sanitize_identifier(worker_type, "worker_type")
        now = _utc_now()
        lease_expires_at = _add_seconds(now, max(5, int(lease_seconds)))
        with self._runtime_connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_workers (
                    worker_id,
                    worker_type,
                    lease_expires_at,
                    last_heartbeat_at,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id, worker_type) DO UPDATE SET
                    lease_expires_at = excluded.lease_expires_at,
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_worker_id,
                    normalized_worker_type,
                    lease_expires_at,
                    now,
                    json.dumps(metadata or {}, default=str),
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT *
                FROM runtime_workers
                WHERE worker_id = ? AND worker_type = ?
                """,
                (normalized_worker_id, normalized_worker_type),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to persist worker heartbeat for '{normalized_worker_id}'")
        return self._row_to_worker_heartbeat(row)

    def list_active_workers(self, worker_type: Optional[str] = None) -> List[WorkerHeartbeatRecord]:
        now = _utc_now()
        clauses = ["lease_expires_at > ?"]
        params: List[Any] = [now]
        if worker_type is not None:
            clauses.append("worker_type = ?")
            params.append(_sanitize_identifier(worker_type, "worker_type"))
        query = (
            "SELECT * FROM runtime_workers WHERE "
            + " AND ".join(clauses)
            + " ORDER BY worker_type ASC, last_heartbeat_at DESC"
        )
        with self._runtime_connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_worker_heartbeat(row) for row in rows]

    def has_active_worker(self, worker_type: str) -> bool:
        return bool(self.list_active_workers(worker_type=worker_type))

    def remove_worker_registration(self, worker_id: str, worker_type: Optional[str] = None) -> int:
        normalized_worker_id = _sanitize_worker_id(worker_id)
        clauses = ["worker_id = ?"]
        params: List[Any] = [normalized_worker_id]
        if worker_type is not None:
            clauses.append("worker_type = ?")
            params.append(_sanitize_identifier(worker_type, "worker_type"))
        query = "DELETE FROM runtime_workers WHERE " + " AND ".join(clauses)
        with self._runtime_connect() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
        return int(cursor.rowcount)

    def get_pending_command(
        self,
        tenant_id: str,
        job_id: str,
        command_types: Optional[List[str]] = None,
    ) -> Optional[JobCommandRecord]:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        clauses = ["tenant_id = ?", "job_id = ?", "status IN ('queued', 'processing')"]
        params: List[Any] = [tenant, job]
        if command_types:
            normalized_types = [str(item).strip().lower() for item in command_types if str(item).strip()]
            if normalized_types:
                placeholders = ",".join("?" for _ in normalized_types)
                clauses.append(f"command_type IN ({placeholders})")
                params.extend(normalized_types)
        params.append(1)
        query = (
            "SELECT * FROM job_commands WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at ASC LIMIT ?"
        )
        with self._runtime_connect() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return self._row_to_command(row)

    def enqueue_job_command(
        self,
        tenant_id: str,
        job_id: str,
        command_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> JobCommandRecord:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        normalized_command = str(command_type).strip().lower()
        if normalized_command not in {"start", "resume", "stop"}:
            raise ValueError(f"Unsupported job command: {command_type}")

        now = _utc_now()
        command_id = secrets.token_hex(12)
        with self._runtime_connect() as conn:
            conn.execute(
                """
                INSERT INTO job_commands (
                    command_id,
                    tenant_id,
                    job_id,
                    command_type,
                    payload_json,
                    status,
                    worker_id,
                    error_message,
                    created_at,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'queued', NULL, NULL, ?, NULL, NULL, ?)
                """,
                (
                    command_id,
                    tenant,
                    job,
                    normalized_command,
                    json.dumps(payload or {}, default=str),
                    now,
                    now,
                ),
            )
            conn.commit()
        with self._runtime_connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_commands WHERE command_id = ?",
                (command_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to enqueue job command for job '{job}'")
        return self._row_to_command(row)

    def claim_next_job_command(self) -> Optional[JobCommandRecord]:
        now = _utc_now()
        with self._runtime_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT q.*
                FROM job_commands q
                LEFT JOIN job_runtime_claims c
                    ON c.tenant_id = q.tenant_id
                   AND c.job_id = q.job_id
                   AND c.lease_expires_at > ?
                WHERE q.status = 'queued'
                  AND (
                    (q.command_type IN ('start', 'resume') AND c.owner_id IS NULL)
                    OR (q.command_type = 'stop' AND c.owner_id = ?)
                  )
                ORDER BY
                    CASE q.command_type
                        WHEN 'stop' THEN 0
                        ELSE 1
                    END,
                    q.created_at ASC
                LIMIT 1
                """,
                (now, self.instance_id),
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            cursor = conn.execute(
                """
                UPDATE job_commands
                SET
                    status = 'processing',
                    worker_id = ?,
                    started_at = ?,
                    updated_at = ?
                WHERE command_id = ? AND status = 'queued'
                """,
                (self.instance_id, now, now, row["command_id"]),
            )
            if cursor.rowcount == 0:
                conn.commit()
                return None

            refreshed = conn.execute(
                """
                SELECT *
                FROM job_commands
                WHERE command_id = ?
                """,
                (row["command_id"],),
            ).fetchone()
            conn.commit()

        if refreshed is None:
            return None
        return self._row_to_command(refreshed)

    def complete_job_command(
        self,
        command_id: str,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        with self._runtime_connect() as conn:
            conn.execute(
                """
                UPDATE job_commands
                SET
                    status = 'completed',
                    error_message = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (error_message, now, now, command_id),
            )
            conn.commit()

    def fail_job_command(self, command_id: str, error_message: str) -> None:
        now = _utc_now()
        with self._runtime_connect() as conn:
            conn.execute(
                """
                UPDATE job_commands
                SET
                    status = 'failed',
                    error_message = ?,
                    completed_at = ?,
                    updated_at = ?
                WHERE command_id = ?
                """,
                (error_message, now, now, command_id),
            )
            conn.commit()

    def acquire_job_control(self, tenant_id: str, job_id: str) -> JobRuntimeClaim:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        now = _utc_now()
        lease_expires_at = _add_seconds(now, self.lease_seconds)

        with self._runtime_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT *
                FROM job_runtime_claims
                WHERE tenant_id = ? AND job_id = ?
                """,
                (tenant, job),
            ).fetchone()

            if row is not None:
                existing_claim = self._row_to_runtime_claim(row)
                if (
                    existing_claim.owner_id != self.instance_id
                    and existing_claim.lease_expires_at > now
                ):
                    conn.rollback()
                    raise JobControlConflictError(
                        tenant_id=tenant,
                        job_id=job,
                        owner_id=existing_claim.owner_id,
                        lease_expires_at=existing_claim.lease_expires_at,
                    )
                conn.execute(
                    """
                    UPDATE job_runtime_claims
                    SET
                        owner_id = ?,
                        lease_expires_at = ?,
                        last_heartbeat_at = ?,
                        acquired_at = ?,
                        updated_at = ?
                    WHERE tenant_id = ? AND job_id = ?
                    """,
                    (
                        self.instance_id,
                        lease_expires_at,
                        now,
                        now,
                        now,
                        tenant,
                        job,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO job_runtime_claims (
                        tenant_id,
                        job_id,
                        owner_id,
                        lease_expires_at,
                        last_heartbeat_at,
                        acquired_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant,
                        job,
                        self.instance_id,
                        lease_expires_at,
                        now,
                        now,
                        now,
                    ),
                )
            conn.commit()

        claim = self.get_runtime_claim(tenant, job)
        if claim is None:
            raise RuntimeError(f"Failed to persist runtime claim for job '{job}'")
        return claim

    def renew_job_control(self, tenant_id: str, job_id: str) -> Optional[JobRuntimeClaim]:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        now = _utc_now()
        lease_expires_at = _add_seconds(now, self.lease_seconds)
        with self._runtime_connect() as conn:
            cursor = conn.execute(
                """
                UPDATE job_runtime_claims
                SET
                    lease_expires_at = ?,
                    last_heartbeat_at = ?,
                    updated_at = ?
                WHERE tenant_id = ? AND job_id = ? AND owner_id = ?
                """,
                (
                    lease_expires_at,
                    now,
                    now,
                    tenant,
                    job,
                    self.instance_id,
                ),
            )
            conn.commit()

        if cursor.rowcount == 0:
            return None
        return self.get_runtime_claim(tenant, job)

    def release_job_control(self, tenant_id: str, job_id: str) -> bool:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        with self._runtime_connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM job_runtime_claims
                WHERE tenant_id = ? AND job_id = ? AND owner_id = ?
                """,
                (tenant, job, self.instance_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def ensure_local_job_control(self, tenant_id: str, job_id: str) -> Optional[JobRuntimeClaim]:
        claim = self.get_runtime_claim(tenant_id, job_id)
        if claim is None:
            return None
        if claim.owner_id != self.instance_id:
            raise JobControlConflictError(
                tenant_id=claim.tenant_id,
                job_id=claim.job_id,
                owner_id=claim.owner_id,
                lease_expires_at=claim.lease_expires_at,
            )
        return claim

    def apply_runtime_status_overlay(
        self,
        tenant_id: str,
        job_id: str,
        payload: Dict[str, object],
    ) -> Dict[str, object]:
        overlay = dict(payload)
        runtime_status = self.get_runtime_status(tenant_id, job_id)
        if runtime_status is not None and runtime_status.status_payload:
            overlay = dict(runtime_status.status_payload)
            if runtime_status.last_error and "error" not in overlay:
                overlay["error"] = runtime_status.last_error

        claim = self.get_runtime_claim(tenant_id, job_id)
        if claim is not None and claim.owner_id != self.instance_id:
            overlay["status"] = "running"
            overlay["last_update"] = claim.last_heartbeat_at
            system_payload: Dict[str, Any] = {}
            system_dict = overlay.get("system", {})
            if isinstance(system_dict, dict):
                system_payload = {str(k): v for k, v in system_dict.items()}
            system_payload["status"] = "running"
            system_payload["last_update"] = claim.last_heartbeat_at
            overlay["system"] = system_payload

        pending_command = self.get_pending_command(tenant_id, job_id, ["start", "resume"])
        if pending_command is not None and claim is None:
            overlay["status"] = "queued"
            overlay["last_update"] = pending_command.created_at
            system_payload: Dict[str, Any] = {}
            system_dict = overlay.get("system", {})
            if isinstance(system_dict, dict):
                system_payload = {str(k): v for k, v in system_dict.items()}
            system_payload["status"] = "queued"
            system_payload["last_update"] = pending_command.created_at
            overlay["system"] = system_payload
        return overlay

    def _lease_key(self, tenant_id: str, job_id: str) -> Tuple[str, str]:
        return (_sanitize_identifier(tenant_id, "tenant_id"), _sanitize_identifier(job_id, "job_id"))

    def ensure_runtime_lease_task(self, tenant_id: str, job_id: str, trainer: EvoTrainer) -> None:
        key = self._lease_key(tenant_id, job_id)
        existing_task = self._lease_tasks.get(key)
        if existing_task is not None and not existing_task.done():
            return

        async def _lease_loop() -> None:
            try:
                while trainer.is_running:
                    await asyncio.sleep(self.heartbeat_interval_seconds)
                    if not trainer.is_running:
                        break
                    renewed = await asyncio.to_thread(self.renew_job_control, tenant_id, job_id)
                    if renewed is None:
                        logger.warning(
                            "Lost runtime claim for tenant=%s job=%s; stopping local trainer",
                            tenant_id,
                            job_id,
                        )
                        trainer.is_running = False
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Lease heartbeat failed for tenant=%s job=%s: %s",
                    tenant_id,
                    job_id,
                    exc,
                )
                trainer.is_running = False
            finally:
                self._lease_tasks.pop(key, None)
                await asyncio.to_thread(self.release_job_control, tenant_id, job_id)

        self._lease_tasks[key] = asyncio.create_task(_lease_loop())

    async def stop_runtime_lease_task(self, tenant_id: str, job_id: str) -> None:
        key = self._lease_key(tenant_id, job_id)
        task = self._lease_tasks.pop(key, None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.to_thread(self.release_job_control, tenant_id, job_id)

    async def shutdown(self) -> None:
        owned_keys = list(self._lease_tasks.keys())
        for tenant_id, job_id in owned_keys:
            await self.stop_runtime_lease_task(tenant_id, job_id)

    def _write_job(self, record: JobRecord) -> None:
        job_dir = Path(record.base_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._job_file(record.tenant_id, record.job_id).write_text(
            json.dumps(record.to_dict(), indent=2),
            encoding="utf-8",
        )

    def _read_job(self, tenant_id: str, job_id: str) -> Optional[JobRecord]:
        job_path = self._job_file(tenant_id, job_id)
        if not job_path.exists():
            return None
        payload = json.loads(job_path.read_text(encoding="utf-8"))
        return JobRecord(
            job_id=str(payload["job_id"]),
            tenant_id=str(payload["tenant_id"]),
            name=str(payload.get("name", payload["job_id"])),
            base_dir=str(payload["base_dir"]),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            status=str(payload.get("status", "created")),
            generation=int(payload.get("generation", 0) or 0),
        )

    def create_job(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> JobRecord:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        if job_id is None:
            job_id = f"job_{secrets.token_hex(6)}"
        job = _sanitize_identifier(job_id, "job_id")

        existing = self._read_job(tenant, job)
        if existing is not None:
            raise ValueError(f"Job '{job}' already exists")

        base_dir = self._job_dir(tenant, job)
        now = _utc_now()
        record = JobRecord(
            job_id=job,
            tenant_id=tenant,
            name=name or job,
            base_dir=str(base_dir),
            created_at=now,
            updated_at=now,
            status="created",
            generation=0,
        )
        self._write_job(record)
        return record

    def ensure_default_job(self, tenant_id: str) -> JobRecord:
        existing = self._read_job(tenant_id, "default")
        if existing is not None:
            return existing
        return self.create_job(tenant_id=tenant_id, name="Default Job", job_id="default")

    def list_jobs(self, tenant_id: str) -> List[JobRecord]:
        tenant_jobs_dir = self._tenant_dir(tenant_id)
        if not tenant_jobs_dir.exists():
            return []

        items: List[JobRecord] = []
        for job_path in sorted(tenant_jobs_dir.glob("*/job.json")):
            payload = json.loads(job_path.read_text(encoding="utf-8"))
            items.append(
                JobRecord(
                    job_id=str(payload["job_id"]),
                    tenant_id=str(payload["tenant_id"]),
                    name=str(payload.get("name", payload["job_id"])),
                    base_dir=str(payload["base_dir"]),
                    created_at=str(payload.get("created_at", "")),
                    updated_at=str(payload.get("updated_at", "")),
                    status=str(payload.get("status", "created")),
                    generation=int(payload.get("generation", 0) or 0),
                )
            )

        for item in items:
            runtime_status = self.get_runtime_status(item.tenant_id, item.job_id)
            if runtime_status is not None:
                status_value = runtime_status.status_payload.get("status")
                generation_value = runtime_status.status_payload.get("generation")
                if status_value is not None:
                    item.status = str(status_value)
                if generation_value is not None:
                    try:
                        item.generation = int(generation_value)
                    except (TypeError, ValueError):
                        pass
                item.updated_at = runtime_status.updated_at
            runtime_claim = self.get_runtime_claim(item.tenant_id, item.job_id)
            if runtime_claim is not None:
                item.status = "running"
                item.updated_at = runtime_claim.last_heartbeat_at
            else:
                pending_command = self.get_pending_command(item.tenant_id, item.job_id, ["start", "resume"])
                if pending_command is not None:
                    item.status = "queued"
                    item.updated_at = pending_command.created_at

        items.sort(key=lambda item: item.created_at, reverse=True)
        return items

    def get_job(self, tenant_id: str, job_id: str) -> Optional[JobRecord]:
        return self._read_job(tenant_id, job_id)

    def update_job_status(
        self,
        tenant_id: str,
        job_id: str,
        trainer: Optional[EvoTrainer] = None,
        status: Optional[str] = None,
    ) -> JobRecord:
        record = self._read_job(tenant_id, job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' does not exist")

        record.updated_at = _utc_now()
        runtime_status = self.get_runtime_status(tenant_id, job_id)
        if runtime_status is not None:
            status_value = runtime_status.status_payload.get("status")
            generation_value = runtime_status.status_payload.get("generation")
            if status_value is not None:
                record.status = str(status_value)
            if generation_value is not None:
                try:
                    record.generation = int(generation_value)
                except (TypeError, ValueError):
                    pass
            record.updated_at = runtime_status.updated_at
        runtime_claim = self.get_runtime_claim(tenant_id, job_id)
        if runtime_claim is not None and runtime_claim.owner_id != self.instance_id:
            record.status = "running"
            record.updated_at = runtime_claim.last_heartbeat_at
        elif runtime_claim is None:
            pending_command = self.get_pending_command(tenant_id, job_id, ["start", "resume"])
            if pending_command is not None:
                record.status = "queued"
                record.updated_at = pending_command.created_at
        elif trainer is not None:
            trainer.update_status()
            record.status = str(trainer.last_status.get("status", record.status))
            record.generation = int(trainer.last_status.get("generation", record.generation) or 0)
        elif status is not None:
            record.status = status

        self._write_job(record)
        return record

    async def get_trainer(self, tenant_id: str, job_id: str) -> EvoTrainer:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")

        record = self._read_job(tenant, job)
        if record is None:
            raise ValueError(f"Job '{job}' does not exist")

        runtime_claim = self.get_runtime_claim(tenant, job)
        if runtime_claim is None or runtime_claim.owner_id != self.instance_id:
            trainer = EvoTrainer(base_dir=record.base_dir)
            initialized = await trainer.initialize()
            if not initialized:
                raise RuntimeError(f"Failed to initialize trainer for job '{job}'")
            return trainer

        key = (tenant, job)
        if key in self._trainer_cache:
            return self._trainer_cache[key]

        async with self._lock_for(tenant, job):
            if key in self._trainer_cache:
                return self._trainer_cache[key]

            trainer = EvoTrainer(base_dir=record.base_dir)
            initialized = await trainer.initialize()
            if not initialized:
                raise RuntimeError(f"Failed to initialize trainer for job '{job}'")
            self._trainer_cache[key] = trainer
            self._agent_cache[key] = AgentInterface(trainer)
            self.update_job_status(tenant, job, trainer=trainer)
            return trainer

    async def get_agent(self, tenant_id: str, job_id: str) -> AgentInterface:
        tenant = _sanitize_identifier(tenant_id, "tenant_id")
        job = _sanitize_identifier(job_id, "job_id")
        trainer = await self.get_trainer(tenant, job)
        key = (tenant, job)
        if key not in self._trainer_cache:
            return AgentInterface(trainer)

        agent = self._agent_cache.get(key)
        if agent is None:
            agent = AgentInterface(trainer)
            self._agent_cache[key] = agent
        return agent
