import asyncio
import unittest
import uuid
from pathlib import Path
from typing import Any, cast

from api.job_manager import JobControlConflictError, JobManager
from tests.postgres_utils import postgres_url, reset_tables
from tests.tmp_utils import cleanup_path


class JobRuntimeClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.db_url = postgres_url()
        reset_tables(self.db_url)
        self.root_dir = Path(f"tests/.tmp/job-runtime-root-{suffix}")
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.manager_a = JobManager(
            root_dir=str(self.root_dir),
            runtime_db_url=self.db_url,
            instance_id="instance-a",
            lease_seconds=30,
            heartbeat_interval_seconds=1,
        )
        self.manager_b = JobManager(
            root_dir=str(self.root_dir),
            runtime_db_url=self.db_url,
            instance_id="instance-b",
            lease_seconds=30,
            heartbeat_interval_seconds=1,
        )
        self.tenant_id = "tenant1"
        self.job_id = "job1"

    def tearDown(self) -> None:
        asyncio.run(self.manager_a.shutdown())
        asyncio.run(self.manager_b.shutdown())
        reset_tables(self.db_url)
        cleanup_path(self.root_dir)

    def test_second_instance_cannot_claim_active_job(self) -> None:
        claim = self.manager_a.acquire_job_control(self.tenant_id, self.job_id)

        self.assertEqual(claim.owner_id, "instance-a")
        with self.assertRaises(JobControlConflictError):
            self.manager_b.acquire_job_control(self.tenant_id, self.job_id)

    def test_expired_claim_can_be_taken_over(self) -> None:
        self.manager_a.acquire_job_control(self.tenant_id, self.job_id)
        with self.manager_a._runtime_connect() as conn:
            conn.execute(
                """
                UPDATE job_runtime_claims
                SET lease_expires_at = ?, last_heartbeat_at = ?, updated_at = ?
                WHERE tenant_id = ? AND job_id = ?
                """,
                (
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                    self.tenant_id,
                    self.job_id,
                ),
            )
            conn.commit()

        claim = self.manager_b.acquire_job_control(self.tenant_id, self.job_id)

        self.assertEqual(claim.owner_id, "instance-b")

    def test_runtime_overlay_marks_remote_job_as_running(self) -> None:
        self.manager_a.acquire_job_control(self.tenant_id, self.job_id)
        payload = {
            "status": "stopped",
            "last_update": "",
            "system": {
                "status": "stopped",
                "last_update": "",
            },
        }

        overlaid = self.manager_b.apply_runtime_status_overlay(
            self.tenant_id,
            self.job_id,
            payload,
        )

        self.assertEqual(overlaid.get("status"), "running")
        system_payload = cast(dict[str, Any], overlaid.get("system", {}))
        self.assertEqual(system_payload.get("status"), "running")
        self.assertTrue(overlaid["last_update"])

    def test_expired_training_worker_heartbeat_is_not_reported_active(self) -> None:
        heartbeat = self.manager_a.record_worker_heartbeat(
            worker_id="training-worker-1",
            worker_type="training",
            metadata={"pid": 1234},
            lease_seconds=30,
        )
        self.assertEqual(heartbeat.worker_type, "training")
        self.assertTrue(self.manager_b.has_active_worker("training"))

        with self.manager_a._runtime_connect() as conn:
            conn.execute(
                """
                UPDATE runtime_workers
                SET lease_expires_at = ?, last_heartbeat_at = ?, updated_at = ?
                WHERE worker_id = ? AND worker_type = ?
                """,
                (
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                    "2000-01-01T00:00:00Z",
                    "training-worker-1",
                    "training",
                ),
            )
            conn.commit()

        self.assertFalse(self.manager_b.has_active_worker("training"))


if __name__ == "__main__":
    unittest.main()
