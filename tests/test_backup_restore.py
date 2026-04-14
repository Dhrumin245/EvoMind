import asyncio
import gc
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from api.auth import APIKeyStore
from api.backup import create_backup, restore_backup
from api.events import EventManager
from api.job_manager import JobManager
from api.storage import api_auth_db_path, tenant_root_dir
from tests.tmp_utils import cleanup_path


class BackupRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path(f"tests/.tmp/backup-restore-{uuid.uuid4().hex}")
        self.data_dir = self.base_dir / "data"
        self.backups_dir = self.base_dir / "backups"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        cleanup_path(self.base_dir)

    def test_round_trip_restores_sqlite_state_and_tenant_artifacts(self) -> None:
        env = {
            "EVOMIND_ENV": "development",
            "EVOMIND_DATA_DIR": str(self.data_dir),
            "EVOMIND_BACKUP_DIR": str(self.backups_dir),
        }
        with patch.dict(os.environ, env, clear=False):
            auth_store = APIKeyStore()
            event_manager = EventManager()
            job_manager = JobManager(instance_id="backup-test")

            principal, _ = auth_store.create_key(name="backup-key", tenant_id="tenant-a")
            job = job_manager.create_job("tenant-a", job_id="job-a", name="Backup Job")
            asyncio.run(
                event_manager.emit_event(
                    tenant_id="tenant-a",
                    job_id="job-a",
                    event_type="job.created",
                    payload={"job_id": "job-a"},
                )
            )
            artifact_file = Path(job.base_dir) / "config.json"
            artifact_file.write_text('{"restored": true}', encoding="utf-8")

            archive = create_backup()
            auth_store.create_key(name="backup-key-2", tenant_id="tenant-a")
            asyncio.run(
                event_manager.emit_event(
                    tenant_id="tenant-a",
                    job_id="job-a",
                    event_type="job.updated",
                    payload={"job_id": "job-a"},
                )
            )
            artifact_file.write_text('{"restored": false}', encoding="utf-8")
            auth_store = None
            event_manager = None
            job_manager = None
            gc.collect()
            restore_backup(str(archive), force=True)

            restored_auth = APIKeyStore()
            restored_events = EventManager()
            restored_jobs = JobManager(instance_id="restore-test")

            keys = restored_auth.list_keys()
            self.assertEqual(len(keys), 1)
            self.assertEqual(keys[0].tenant_id, principal.tenant_id)

            restored_job = restored_jobs.get_job("tenant-a", "job-a")
            self.assertIsNotNone(restored_job)
            assert restored_job is not None
            self.assertTrue(Path(restored_job.base_dir, "config.json").exists())
            self.assertEqual(
                Path(restored_job.base_dir, "config.json").read_text(encoding="utf-8"),
                '{"restored": true}',
            )

            events = restored_events.list_events("tenant-a", "job-a")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "job.created")

            self.assertTrue(api_auth_db_path().exists())
            self.assertTrue(tenant_root_dir().exists())


if __name__ == "__main__":
    unittest.main()
