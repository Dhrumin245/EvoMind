import asyncio
import unittest
import uuid
from pathlib import Path

from api.job_manager import JobManager
from tests.tmp_utils import cleanup_path


class JobCommandQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.runtime_db_path = Path(f"tests/.tmp/job-commands-{suffix}.db")
        self.root_dir = Path(f"tests/.tmp/job-commands-root-{suffix}")
        self.runtime_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.manager = JobManager(
            root_dir=str(self.root_dir),
            runtime_db_path=str(self.runtime_db_path),
            instance_id="worker-a",
            lease_seconds=30,
            heartbeat_interval_seconds=1,
        )

    def tearDown(self) -> None:
        asyncio.run(self.manager.shutdown())
        cleanup_path(self.runtime_db_path)
        cleanup_path(self.root_dir)

    def test_start_command_is_claimed_when_job_is_unowned(self) -> None:
        command = self.manager.enqueue_job_command("tenant1", "job1", "start")

        claimed = self.manager.claim_next_job_command()

        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.command_id, command.command_id)
        self.assertEqual(claimed.status, "processing")
        self.assertEqual(claimed.worker_id, "worker-a")

    def test_stop_command_is_claimed_only_by_current_owner(self) -> None:
        self.manager.acquire_job_control("tenant1", "job1")
        command = self.manager.enqueue_job_command("tenant1", "job1", "stop")

        claimed = self.manager.claim_next_job_command()

        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.command_id, command.command_id)
        self.assertEqual(claimed.command_type, "stop")

    def test_start_command_is_not_claimed_while_job_is_owned(self) -> None:
        self.manager.acquire_job_control("tenant1", "job1")
        self.manager.enqueue_job_command("tenant1", "job1", "start")

        claimed = self.manager.claim_next_job_command()

        self.assertIsNone(claimed)

    def test_runtime_overlay_marks_unowned_queued_start_as_queued(self) -> None:
        self.manager.enqueue_job_command("tenant1", "job1", "start")

        overlay: dict = self.manager.apply_runtime_status_overlay(
            "tenant1",
            "job1",
            {
                "status": "stopped",
                "last_update": "",
                "system": {
                    "status": "stopped",
                    "last_update": "",
                },
            },
        )

        self.assertEqual(overlay["status"], "queued")
        self.assertEqual(overlay["system"]["status"], "queued")


if __name__ == "__main__":
    unittest.main()
