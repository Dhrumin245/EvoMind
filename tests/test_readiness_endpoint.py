import asyncio
import json
import unittest
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from fastapi.responses import JSONResponse

from api import server
from api.events import EventManager
from api.job_manager import JobManager
from api.schemas import ReadinessCheck
from tests.postgres_utils import postgres_url, reset_tables
from tests.tmp_utils import cleanup_path


class ReadinessEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.db_url = postgres_url()
        reset_tables(self.db_url)
        self.root_dir = Path(f"tests/.tmp/readiness-root-{suffix}")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.manager = JobManager(
            root_dir=str(self.root_dir),
            runtime_db_url=self.db_url,
            instance_id="api-instance",
            lease_seconds=30,
            heartbeat_interval_seconds=1,
        )
        self.event_manager = EventManager(db_url=self.db_url)
        self.original_job_manager = server.job_manager
        self.original_event_manager = server.event_manager
        server.job_manager = self.manager
        server.event_manager = self.event_manager

    def tearDown(self) -> None:
        server.job_manager = self.original_job_manager
        server.event_manager = self.original_event_manager
        asyncio.run(self.manager.shutdown())
        reset_tables(self.db_url)
        cleanup_path(self.root_dir)

    def _run_readiness_check(self) -> ReadinessCheck | JSONResponse:
        readiness_call = cast(
            Callable[[], Coroutine[Any, Any, ReadinessCheck | JSONResponse]],
            server.readiness_check,
        )
        return asyncio.run(readiness_call())

    def test_readiness_returns_503_when_training_worker_missing(self) -> None:
        with patch.object(server.api_key_store, "is_available", return_value=True):
            with patch.object(self.event_manager, "is_worker_running", return_value=True):
                response = self._run_readiness_check()

        self.assertIsInstance(response, JSONResponse)
        assert isinstance(response, JSONResponse)
        self.assertEqual(response.status_code, 503)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "not_ready")
        component_map = {item["name"]: item for item in payload["components"]}
        self.assertFalse(component_map["training_worker"]["healthy"])
        self.assertEqual(component_map["training_worker"]["detail"], "no active workers")

    def test_readiness_returns_ready_when_all_dependencies_are_healthy(self) -> None:
        self.manager.record_worker_heartbeat(
            worker_id="training-worker-1",
            worker_type="training",
            metadata={"pid": 1234},
            lease_seconds=30,
        )
        with patch.object(server.api_key_store, "is_available", return_value=True):
            with patch.object(self.event_manager, "is_worker_running", return_value=True):
                response = self._run_readiness_check()

        assert not isinstance(response, JSONResponse)
        self.assertEqual(response.status, "ready")
        component_map = {item.name: item for item in response.components}
        self.assertTrue(component_map["training_worker"].healthy)
        self.assertIn("1 active worker", component_map["training_worker"].detail)


if __name__ == "__main__":
    unittest.main()
