import asyncio
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from api.interface import AgentInterface
from api.trainer import EvoTrainer


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_identifier(value: str, field_name: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{field_name} must contain only letters, numbers, '_' or '-'")
    return value


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


class JobManager:
    def __init__(self, root_dir: str = "data/tenants"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._trainer_cache: Dict[Tuple[str, str], EvoTrainer] = {}
        self._agent_cache: Dict[Tuple[str, str], AgentInterface] = {}
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}

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
        if trainer is not None:
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
        agent = self._agent_cache.get(key)
        if agent is None:
            agent = AgentInterface(trainer)
            self._agent_cache[key] = agent
        return agent
