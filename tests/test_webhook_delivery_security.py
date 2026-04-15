import ipaddress
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

from api.events import EventManager, JobEventRecord, ResolvedWebhookTarget, WebhookRecord
from tests.tmp_utils import cleanup_path


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.closed = False

    def read(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, status: int | None = None, error: Exception | None = None) -> None:
        self.status = status
        self.error = error
        self.closed = False
        self.requests: list[tuple[str, str, bytes | None, dict[str, str] | None]] = []

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.requests.append((method, url, body, headers))
        if self.error is not None:
            raise self.error

    def getresponse(self) -> _FakeResponse:
        if self.status is None:
            raise AssertionError("Expected a configured status code")
        return _FakeResponse(self.status)

    def close(self) -> None:
        self.closed = True


class WebhookDeliverySecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(f"tests/.tmp/webhook-security-{uuid.uuid4().hex}.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.manager = EventManager(db_path=str(self.db_path))

    def tearDown(self) -> None:
        cleanup_path(self.db_path)

    @staticmethod
    def _build_webhook(url: str) -> WebhookRecord:
        return WebhookRecord(
            webhook_id="webhook-1",
            tenant_id="tenant-1",
            url=url,
            description="security test webhook",
            subscribed_events=[],
            status="active",
            created_at="2026-04-15T00:00:00Z",
            updated_at="2026-04-15T00:00:00Z",
            last_delivery_at=None,
            last_delivery_status=None,
            last_delivery_error=None,
            secret="test-secret",
        )

    @staticmethod
    def _build_event() -> JobEventRecord:
        return JobEventRecord(
            event_id="event-1",
            tenant_id="tenant-1",
            job_id="job-1",
            event_type="job.started",
            payload={"state": "queued"},
            created_at="2026-04-15T00:00:00Z",
        )

    @staticmethod
    def _target(ip_text: str, port: int) -> ResolvedWebhookTarget:
        address = ipaddress.ip_address(ip_text)
        return ResolvedWebhookTarget(
            family=0,
            socktype=0,
            proto=0,
            sockaddr=(ip_text, port),
            address=address,
        )

    def test_send_time_revalidation_blocks_private_dns_rebind(self) -> None:
        webhook = self._build_webhook("https://hooks.example.com/incoming")

        with patch.object(
            EventManager,
            "_resolve_validated_webhook_targets",
            side_effect=ValueError("Webhook host must resolve to publicly routable IP addresses"),
        ) as mock_resolve, patch.object(EventManager, "_build_webhook_connection") as mock_build:
            with self.assertRaisesRegex(ValueError, "publicly routable IP addresses"):
                self.manager._post_webhook(webhook, self._build_event())

        mock_resolve.assert_called_once_with("hooks.example.com", 443)
        mock_build.assert_not_called()

    def test_redirect_response_is_rejected_without_following_target(self) -> None:
        webhook = self._build_webhook("https://hooks.example.com/incoming?attempt=1")
        target = self._target("93.184.216.34", 443)
        connection = _FakeConnection(status=302)

        with patch.object(
            EventManager,
            "_resolve_validated_webhook_targets",
            return_value=[target],
        ), patch.object(
            EventManager,
            "_build_webhook_connection",
            return_value=connection,
        ):
            with self.assertRaises(urllib_error.HTTPError) as exc_info:
                self.manager._post_webhook(webhook, self._build_event())

        self.assertEqual(exc_info.exception.code, 302)
        self.assertEqual(len(connection.requests), 1)
        method, request_target, _, headers = connection.requests[0]
        self.assertEqual(method, "POST")
        self.assertEqual(request_target, "/incoming?attempt=1")
        self.assertIsNotNone(headers)
        assert headers is not None
        self.assertEqual(headers["Host"], "hooks.example.com")
        self.assertTrue(connection.closed)

    def test_delivery_fails_over_across_revalidated_public_ips(self) -> None:
        webhook = self._build_webhook("http://hooks.example.com:8080/webhooks")
        first_target = self._target("93.184.216.34", 8080)
        second_target = self._target("93.184.216.35", 8080)
        first_connection = _FakeConnection(error=OSError("connect failed"))
        second_connection = _FakeConnection(status=204)

        with patch.object(
            EventManager,
            "_resolve_validated_webhook_targets",
            return_value=[first_target, second_target],
        ), patch.object(
            EventManager,
            "_build_webhook_connection",
            side_effect=[first_connection, second_connection],
        ) as mock_build:
            status_code = self.manager._post_webhook(webhook, self._build_event())

        self.assertEqual(status_code, 204)
        self.assertTrue(first_connection.closed)
        self.assertTrue(second_connection.closed)
        self.assertEqual(mock_build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
