import argparse
import asyncio
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from api.events import EventManager
from api.job_manager import JobCommandRecord, JobManager
from api.logging_utils import configure_logging, push_log_context, reset_log_context
from api.trainer import EvoTrainer
from api.storage import tenant_root_dir
from api.schemas import TrainStatus


logger = logging.getLogger(__name__)


def _serialize_train_status(status: TrainStatus) -> Dict[str, Any]:
    if hasattr(status, "model_dump"):
        return status.model_dump()  # type: ignore[no-any-return]
    return status.dict()  # type: ignore[no-any-return]


class TrainingWorker:
    def __init__(
        self,
        root_dir: Optional[str] = None,
        runtime_db_url: Optional[str] = None,
        events_db_url: Optional[str] = None,
        poll_interval_seconds: float = 2.0,
        status_interval_seconds: float = 2.0,
        worker_heartbeat_interval_seconds: float = 10.0,
        worker_heartbeat_ttl_seconds: int = 30,
    ):
        resolved_root_dir = str(Path(root_dir) if root_dir is not None else tenant_root_dir())
        self.job_manager = JobManager(
            root_dir=resolved_root_dir,
            runtime_db_url=runtime_db_url,
        )
        self.event_manager = EventManager(db_url=events_db_url)
        self.poll_interval_seconds = max(0.2, float(poll_interval_seconds))
        self.status_interval_seconds = max(0.2, float(status_interval_seconds))
        self.worker_heartbeat_interval_seconds = max(1.0, float(worker_heartbeat_interval_seconds))
        self.worker_heartbeat_ttl_seconds = max(
            5,
            int(
                max(
                    worker_heartbeat_ttl_seconds,
                    self.worker_heartbeat_interval_seconds * 2,
                )
            ),
        )
        self.worker_id = self.job_manager.instance_id
        self.worker_metadata = {
            "pid": os.getpid(),
            "root_dir": resolved_root_dir,
            "runtime_db_url": runtime_db_url,
            "events_db_url": events_db_url,
        }
        self._active_trainers: Dict[Tuple[str, str], EvoTrainer] = {}
        self._status_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._worker_heartbeat_task: Optional[asyncio.Task] = None
        self._worker_log_context_token = None
        self._shutdown = False

    async def start(self) -> None:
        self._worker_log_context_token = push_log_context(worker_id=self.worker_id)
        await self.event_manager.start_worker()
        await asyncio.to_thread(
            self.job_manager.record_worker_heartbeat,
            self.worker_id,
            "training",
            self.worker_metadata,
            self.worker_heartbeat_ttl_seconds,
        )
        self._worker_heartbeat_task = asyncio.create_task(self._worker_heartbeat_loop())
        logger.info("Training worker started worker_id=%s", self.worker_id)

    async def stop(self) -> None:
        self._shutdown = True
        heartbeat_task = self._worker_heartbeat_task
        self._worker_heartbeat_task = None
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        for task in list(self._status_tasks.values()):
            task.cancel()
        for task in list(self._status_tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass

        for (tenant_id, job_id), trainer in list(self._active_trainers.items()):
            try:
                if trainer.is_running:
                    await trainer.stop()
            except Exception as exc:
                logger.warning(
                    "Worker shutdown stop failed tenant=%s job=%s error=%s",
                    tenant_id,
                    job_id,
                    exc,
                )
            finally:
                await self.job_manager.stop_runtime_lease_task(tenant_id, job_id)
                self._active_trainers.pop((tenant_id, job_id), None)

        await self.job_manager.shutdown()
        await asyncio.to_thread(self.job_manager.remove_worker_registration, self.worker_id, "training")
        await self.event_manager.stop_worker()
        if self._worker_log_context_token is not None:
            reset_log_context(self._worker_log_context_token)
            self._worker_log_context_token = None

    async def run_forever(self) -> None:
        await self.start()
        try:
            while not self._shutdown:
                command = await asyncio.to_thread(self.job_manager.claim_next_job_command)
                if command is None:
                    await asyncio.sleep(self.poll_interval_seconds)
                    continue
                await self._handle_command(command)
        finally:
            await self.stop()

    async def _emit_event(
        self,
        tenant_id: str,
        job_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            await self.event_manager.emit_event(
                tenant_id=tenant_id,
                job_id=job_id,
                event_type=event_type,
                payload=payload,
            )
        except Exception as exc:
            logger.warning(
                "Event emission failed tenant=%s job=%s event=%s error=%s",
                tenant_id,
                job_id,
                event_type,
                exc,
            )

    async def _worker_heartbeat_loop(self) -> None:
        try:
            while not self._shutdown:
                await asyncio.sleep(self.worker_heartbeat_interval_seconds)
                if self._shutdown:
                    break
                await asyncio.to_thread(
                    self.job_manager.record_worker_heartbeat,
                    self.worker_id,
                    "training",
                    self.worker_metadata,
                    self.worker_heartbeat_ttl_seconds,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Training worker heartbeat failed worker_id=%s error=%s", self.worker_id, exc)

    async def _handle_command(self, command: JobCommandRecord) -> None:
        token = push_log_context(
            worker_id=self.worker_id,
            tenant_id=command.tenant_id,
            job_id=command.job_id,
        )
        try:
            if command.command_type in {"start", "resume"}:
                await self._handle_start_or_resume(command)
                return
            if command.command_type == "stop":
                await self._handle_stop(command)
                return

            self.job_manager.fail_job_command(
                command.command_id,
                f"Unsupported command type: {command.command_type}",
            )
        finally:
            reset_log_context(token)

    async def _build_trainer(self, tenant_id: str, job_id: str) -> EvoTrainer:
        record = self.job_manager.get_job(tenant_id, job_id)
        if record is None:
            raise ValueError(f"Job '{job_id}' does not exist")
        trainer = EvoTrainer(base_dir=record.base_dir)
        initialized = await trainer.initialize()
        if not initialized:
            raise RuntimeError(f"Failed to initialize trainer for job '{job_id}'")
        return trainer

    async def _publish_runtime_status(
        self,
        tenant_id: str,
        job_id: str,
        trainer: EvoTrainer,
        active_command_id: Optional[str] = None,
        command_type: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        payload = _serialize_train_status(trainer.status())
        await asyncio.to_thread(
            self.job_manager.upsert_runtime_status,
            tenant_id,
            job_id,
            payload,
            self.worker_id,
            active_command_id,
            command_type,
            last_error,
        )

    async def _publish_error_status(
        self,
        tenant_id: str,
        job_id: str,
        error_message: str,
        active_command_id: Optional[str],
        command_type: Optional[str],
    ) -> None:
        payload = {
            "status": "error",
            "generation": 0,
            "last_update": "",
            "system": {
                "status": "error",
                "last_update": "",
                "uptime_seconds": 0.0,
                "evaluation_time_sec": 0.0,
            },
        }
        await asyncio.to_thread(
            self.job_manager.upsert_runtime_status,
            tenant_id,
            job_id,
            payload,
            self.worker_id,
            active_command_id,
            command_type,
            error_message,
        )

    def _ensure_status_task(self, tenant_id: str, job_id: str, trainer: EvoTrainer) -> None:
        key = (tenant_id, job_id)
        existing = self._status_tasks.get(key)
        if existing is not None and not existing.done():
            return

        async def _status_loop() -> None:
            try:
                while True:
                    await self._publish_runtime_status(tenant_id, job_id, trainer)
                    if trainer.training_task is None or trainer.training_task.done():
                        break
                    await asyncio.sleep(self.status_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Runtime status publish failed tenant=%s job=%s error=%s",
                    tenant_id,
                    job_id,
                    exc,
                )
            finally:
                try:
                    await self._publish_runtime_status(tenant_id, job_id, trainer)
                except Exception:
                    pass
                self._status_tasks.pop(key, None)
                self._active_trainers.pop(key, None)
                await self.job_manager.stop_runtime_lease_task(tenant_id, job_id)

        self._status_tasks[key] = asyncio.create_task(_status_loop())

    async def _handle_start_or_resume(self, command: JobCommandRecord) -> None:
        tenant_id = command.tenant_id
        job_id = command.job_id
        key = (tenant_id, job_id)

        try:
            await asyncio.to_thread(self.job_manager.acquire_job_control, tenant_id, job_id)
            trainer = await self._build_trainer(tenant_id, job_id)

            if command.command_type == "start":
                result = await trainer.start()
                if result.get("status") == "initialization_failed":
                    raise RuntimeError("Training initialization failed")
                await self._emit_event(
                    tenant_id,
                    job_id,
                    "job.started",
                    {"status": trainer.last_status},
                )
            else:
                checkpoint_path = str(command.payload.get("checkpoint_path", "") or "")
                result = await trainer.resume(checkpoint_path)
                if result.get("status") == "resume_failed":
                    raise RuntimeError(str(result.get("error", "Resume failed")))
                await self._emit_event(
                    tenant_id,
                    job_id,
                    "job.resumed",
                    {
                        "checkpoint_path": result.get("checkpoint_path", checkpoint_path),
                        "status": trainer.last_status,
                    },
                )

            self._active_trainers[key] = trainer
            self.job_manager.ensure_runtime_lease_task(tenant_id, job_id, trainer)
            await self._publish_runtime_status(
                tenant_id,
                job_id,
                trainer,
                active_command_id=command.command_id,
                command_type=command.command_type,
            )
            self.job_manager.complete_job_command(command.command_id)
            self._ensure_status_task(tenant_id, job_id, trainer)
        except Exception as exc:
            logger.error(
                "Worker command failed worker_id=%s tenant=%s job=%s command=%s error=%s",
                self.worker_id,
                tenant_id,
                job_id,
                command.command_type,
                exc,
            )
            await self.job_manager.stop_runtime_lease_task(tenant_id, job_id)
            await self._publish_error_status(
                tenant_id,
                job_id,
                str(exc),
                active_command_id=command.command_id,
                command_type=command.command_type,
            )
            self.job_manager.fail_job_command(command.command_id, str(exc))

    async def _handle_stop(self, command: JobCommandRecord) -> None:
        tenant_id = command.tenant_id
        job_id = command.job_id
        key = (tenant_id, job_id)
        trainer = self._active_trainers.get(key)

        try:
            if trainer is None:
                self.job_manager.complete_job_command(command.command_id)
                return

            await trainer.stop()
            await self._publish_runtime_status(
                tenant_id,
                job_id,
                trainer,
                active_command_id=command.command_id,
                command_type=command.command_type,
            )
            self.job_manager.complete_job_command(command.command_id)
            await self._emit_event(
                tenant_id,
                job_id,
                "job.stopped",
                {"status": trainer.last_status},
            )
        except Exception as exc:
            logger.error(
                "Worker stop failed worker_id=%s tenant=%s job=%s error=%s",
                self.worker_id,
                tenant_id,
                job_id,
                exc,
            )
            await self._publish_error_status(
                tenant_id,
                job_id,
                str(exc),
                active_command_id=command.command_id,
                command_type=command.command_type,
            )
            self.job_manager.fail_job_command(command.command_id, str(exc))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EvoMind training worker")
    parser.add_argument("--root-dir", default=str(tenant_root_dir()), help="Tenant job root directory")
    parser.add_argument("--runtime-db-url", default=None, help="Optional PostgreSQL runtime coordination DB URL")
    parser.add_argument("--events-db-url", default=None, help="Optional PostgreSQL event DB URL")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Command poll interval in seconds")
    parser.add_argument("--status-interval", type=float, default=2.0, help="Status publish interval in seconds")
    parser.add_argument(
        "--worker-heartbeat-interval",
        type=float,
        default=10.0,
        help="Training worker heartbeat interval in seconds",
    )
    parser.add_argument(
        "--worker-heartbeat-ttl",
        type=int,
        default=30,
        help="Training worker readiness lease TTL in seconds",
    )
    return parser


async def _async_main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    configure_logging(service_name="evomind", component="training_worker")
    worker = TrainingWorker(
        root_dir=args.root_dir,
        runtime_db_url=args.runtime_db_url,
        events_db_url=args.events_db_url,
        poll_interval_seconds=args.poll_interval,
        status_interval_seconds=args.status_interval,
        worker_heartbeat_interval_seconds=args.worker_heartbeat_interval,
        worker_heartbeat_ttl_seconds=args.worker_heartbeat_ttl,
    )
    await worker.run_forever()
    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
