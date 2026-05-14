import argparse
import asyncio
import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.auth import APIKeyStore
from api.backup import create_backup, restore_backup
from api.events import EventManager
from api.job_manager import JobManager


def _require_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Create, restore, and verify an EvoMind backup archive")


def main() -> int:
    parser = _build_parser()
    parser.parse_args()

    _require_env("EVOMIND_CONTROL_PLANE_DB_URL")
    data_dir = Path(_require_env("EVOMIND_DATA_DIR"))
    backups_dir = Path(_require_env("EVOMIND_BACKUP_DIR"))
    data_dir.mkdir(parents=True, exist_ok=True)
    backups_dir.mkdir(parents=True, exist_ok=True)

    auth_store = APIKeyStore()
    event_manager = EventManager()
    job_manager = JobManager(instance_id="backup-restore-drill")

    principal, _ = auth_store.create_key(name="drill-key", tenant_id="tenant-drill")
    job = job_manager.create_job("tenant-drill", job_id="job-drill", name="Restore Drill Job")
    asyncio.run(
        event_manager.emit_event(
            tenant_id="tenant-drill",
            job_id="job-drill",
            event_type="job.created",
            payload={"job_id": "job-drill"},
        )
    )
    artifact_file = Path(job.base_dir) / "config.json"
    artifact_file.write_text('{"restored": true}', encoding="utf-8")

    archive_path = create_backup()

    auth_store.create_key(name="drill-key-mutated", tenant_id="tenant-drill")
    asyncio.run(
        event_manager.emit_event(
            tenant_id="tenant-drill",
            job_id="job-drill",
            event_type="job.updated",
            payload={"job_id": "job-drill"},
        )
    )
    artifact_file.write_text('{"restored": false}', encoding="utf-8")

    auth_store = None
    event_manager = None
    asyncio.run(job_manager.shutdown())
    job_manager = None
    gc.collect()

    restore_backup(str(archive_path), force=True)

    restored_auth = APIKeyStore()
    restored_events = EventManager()
    restored_jobs = JobManager(instance_id="backup-restore-drill-verify")

    keys = restored_auth.list_keys(tenant_id="tenant-drill")
    if len(keys) != 1 or keys[0].tenant_id != principal.tenant_id:
        raise RuntimeError("Backup restore drill failed: API keys were not restored correctly")

    restored_job = restored_jobs.get_job("tenant-drill", "job-drill")
    if restored_job is None:
        raise RuntimeError("Backup restore drill failed: job metadata missing after restore")

    restored_artifact = Path(restored_job.base_dir) / "config.json"
    if restored_artifact.read_text(encoding="utf-8") != '{"restored": true}':
        raise RuntimeError("Backup restore drill failed: tenant artifact did not restore correctly")

    events = restored_events.list_events("tenant-drill", "job-drill")
    if len(events) != 1 or events[0].event_type != "job.created":
        raise RuntimeError("Backup restore drill failed: event log was not restored correctly")

    asyncio.run(restored_jobs.shutdown())
    print(f"restore drill archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
