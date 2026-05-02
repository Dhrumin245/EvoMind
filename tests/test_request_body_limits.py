import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi import Request
from fastapi.responses import JSONResponse

from api import server


def _build_request(chunks: list[bytes], headers: dict[str, str] | None = None) -> Request:
    message_queue = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < (len(chunks) - 1),
        }
        for index, chunk in enumerate(chunks)
    ]
    if not message_queue:
        message_queue.append({"type": "http.request", "body": b"", "more_body": False})

    async def receive():
        if message_queue:
            return message_queue.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    raw_headers = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode("latin-1"), value.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/test",
        "raw_path": b"/test",
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }
    return Request(scope, receive)


async def _consume_body_and_respond(request: Request) -> JSONResponse:
    body = await request.body()
    return JSONResponse({"size": len(body)})


class RequestBodyLimitTests(unittest.TestCase):
    def test_invalid_content_length_returns_400(self) -> None:
        request = _build_request([b"abc"], headers={"content-length": "nope"})

        with patch.object(server, "MAX_REQUEST_BYTES", 8):
            response = asyncio.run(server.usage_logging_middleware(request, _consume_body_and_respond))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body), {"detail": "Invalid Content-Length header"})

    def test_oversized_content_length_returns_413_before_body_read(self) -> None:
        request = _build_request([b"abc"], headers={"content-length": "99"})

        with patch.object(server, "MAX_REQUEST_BYTES", 8):
            response = asyncio.run(server.usage_logging_middleware(request, _consume_body_and_respond))

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            json.loads(response.body),
            {"detail": "Request body too large (max 8 bytes)"},
        )

    def test_streamed_body_over_limit_returns_413_without_content_length(self) -> None:
        request = _build_request([b"1234", b"5678", b"9"])

        with patch.object(server, "MAX_REQUEST_BYTES", 8):
            response = asyncio.run(server.usage_logging_middleware(request, _consume_body_and_respond))

        self.assertEqual(response.status_code, 413)
        self.assertEqual(
            json.loads(response.body),
            {"detail": "Request body too large (max 8 bytes)"},
        )

    def test_streamed_body_within_limit_is_passed_through(self) -> None:
        request = _build_request([b"1234", b"56"])

        with patch.object(server, "MAX_REQUEST_BYTES", 8):
            response = asyncio.run(server.usage_logging_middleware(request, _consume_body_and_respond))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {"size": 6})

    def test_usage_logging_middleware_bills_combined_request_and_response_tokens(self) -> None:
        request = _build_request([b"1234"])
        original_usage_log_queue = server.usage_log_queue
        server.usage_log_queue = asyncio.Queue()

        async def call_next(req: Request) -> JSONResponse:
            req.state.principal = type("Principal", (), {"tenant_id": "tenant-a", "key_id": "key-1"})()
            await req.body()
            return JSONResponse({"ok": True})

        try:
            response = asyncio.run(server.usage_logging_middleware(request, call_next))
            payload = server.usage_log_queue.get_nowait()
        finally:
            server.usage_log_queue = original_usage_log_queue

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["billed_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
