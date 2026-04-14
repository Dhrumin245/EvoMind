import asyncio
import io
import json
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Request
from fastapi.responses import JSONResponse

from api import auth, server
from api.auth import APIKeyPrincipal
from api.logging_utils import ContextFieldsFilter, JsonLogFormatter, get_log_context, push_log_context, reset_log_context


def _build_request(headers: dict[str, str] | None = None) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/jobs/job-1/train/status",
        "raw_path": b"/jobs/job-1/train/status",
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
        "path_params": {"job_id": "job-1"},
    }
    return Request(scope, receive)


class LoggingContextTests(unittest.TestCase):
    def test_json_formatter_includes_context_fields(self) -> None:
        logger = logging.getLogger("tests.logging_context")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        handler.addFilter(ContextFieldsFilter({"service": "evomind", "component": "api"}))
        logger.handlers = [handler]

        token = push_log_context(request_id="req-123", tenant_id="tenant-a", path="/jobs")
        try:
            logger.info("Structured log", extra={"event": "unit_test", "job_id": "job-1"})
        finally:
            reset_log_context(token)
            logger.handlers = []

        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["service"], "evomind")
        self.assertEqual(payload["component"], "api")
        self.assertEqual(payload["request_id"], "req-123")
        self.assertEqual(payload["tenant_id"], "tenant-a")
        self.assertEqual(payload["job_id"], "job-1")
        self.assertEqual(payload["event"], "unit_test")
        self.assertEqual(payload["path"], "/jobs")

    def test_usage_logging_middleware_sets_request_id_header_and_logs_completion(self) -> None:
        request = _build_request(headers={"x-request-id": "req-abc123"})
        original_handlers = list(server.logger.handlers)
        original_propagate = server.logger.propagate
        original_level = server.logger.level
        original_usage_log_queue = server.usage_log_queue
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonLogFormatter())
        handler.addFilter(ContextFieldsFilter({"service": "evomind", "component": "api"}))
        server.logger.handlers = [handler]
        server.logger.propagate = False
        server.logger.setLevel(logging.INFO)
        server.usage_log_queue = asyncio.Queue()

        async def call_next(req: Request) -> JSONResponse:
            req.state.principal = SimpleNamespace(tenant_id="tenant-a", key_id="key-1")
            req.state.rate_limits = {"requests_per_minute": 10, "minute_count": 1, "requests_per_day": 20, "day_count": 2}
            return JSONResponse({"ok": True})

        try:
            response = asyncio.run(server.usage_logging_middleware(request, call_next))
        finally:
            server.logger.handlers = original_handlers
            server.logger.propagate = original_propagate
            server.logger.setLevel(original_level)
            server.usage_log_queue = original_usage_log_queue

        self.assertEqual(response.headers["X-Request-ID"], "req-abc123")
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["event"], "request_completed")
        self.assertEqual(payload["request_id"], "req-abc123")
        self.assertEqual(payload["tenant_id"], "tenant-a")
        self.assertEqual(payload["job_id"], "job-1")
        self.assertEqual(payload["status_code"], 200)
        self.assertEqual(payload["method"], "GET")

    def test_require_api_key_binds_tenant_context(self) -> None:
        request = _build_request(headers={"x-api-key": "secret-key"})
        principal = APIKeyPrincipal(
            key_id="key-1",
            name="test",
            tenant_id="tenant-a",
            status="active",
        )

        with patch.object(auth.api_key_store, "resolve_key", return_value=principal):
            with patch.object(auth.api_key_store, "consume_rate_limit", return_value={"requests_per_minute": 10}):
                async def _run():
                    result = await auth.require_api_key(request, header_key="secret-key", bearer=None)
                    return result, get_log_context()

                result, context = asyncio.run(_run())

        self.assertEqual(result.tenant_id, "tenant-a")
        self.assertEqual(context["tenant_id"], "tenant-a")
        self.assertEqual(context["key_id"], "key-1")


if __name__ == "__main__":
    unittest.main()
