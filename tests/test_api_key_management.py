import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request

from api import auth
from api.auth import (
    API_KEY_ROLE_OPERATOR,
    API_KEY_ROLE_READER,
    API_KEY_SCOPE_JOBS_READ,
    API_KEY_SCOPE_TRAINING_READ,
    API_KEY_SCOPE_TRAINING_WRITE,
    APIKeyStore,
)
from tests.postgres_utils import postgres_url, reset_tables


def _future_expiry(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_request(method: str, path: str, route_template: str) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
        "path_params": {"job_id": "job-1"},
        "route": SimpleNamespace(path=route_template),
    }
    return Request(scope, receive)


class APIKeyManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_url = postgres_url()
        reset_tables(self.db_url)
        self.store = APIKeyStore(db_url=self.db_url)
        self.future_expiry = _future_expiry(365)
        self.later_future_expiry = _future_expiry(395)

    def tearDown(self) -> None:
        reset_tables(self.db_url)

    def test_create_key_persists_role_scopes_and_expiry(self) -> None:
        principal, raw_key = self.store.create_key(
            name="reader-key",
            tenant_id="tenant-a",
            role=API_KEY_ROLE_READER,
            scopes=[API_KEY_SCOPE_JOBS_READ, API_KEY_SCOPE_TRAINING_READ],
            expires_at=self.future_expiry,
        )

        resolved = self.store.resolve_key(raw_key)
        listed = self.store.list_keys(tenant_id="tenant-a")

        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.role, API_KEY_ROLE_READER)
        self.assertEqual(resolved.scopes, [API_KEY_SCOPE_JOBS_READ, API_KEY_SCOPE_TRAINING_READ])
        self.assertEqual(resolved.expires_at, self.future_expiry)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0].key_id, principal.key_id)
        self.assertEqual(listed[0].role, API_KEY_ROLE_READER)

    def test_resolve_key_marks_expired_keys_as_expired(self) -> None:
        principal, raw_key = self.store.create_key(
            name="expiring-key",
            tenant_id="tenant-a",
            expires_at=self.future_expiry,
        )

        with self.store._connect() as conn:
            conn.execute(
                "UPDATE api_keys SET expires_at = ?, updated_at = ? WHERE key_id = ?",
                ("2026-04-01T00:00:00Z", "2026-04-01T00:00:00Z", principal.key_id),
            )
            conn.commit()

        resolved = self.store.resolve_key(raw_key)
        stored = self.store.get_key(principal.key_id)

        self.assertIsNone(resolved)
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.status, "expired")
        self.assertIsNotNone(stored.expired_at)

    def test_require_api_key_denies_missing_scope_without_consuming_rate_limit(self) -> None:
        _, raw_key = self.store.create_key(
            name="reader-key",
            tenant_id="tenant-a",
            role=API_KEY_ROLE_READER,
        )
        request = _build_request("POST", "/jobs/job-1/train/start", "/jobs/{job_id}/train/start")

        with patch.object(auth, "api_key_store", self.store):
            with patch.object(self.store, "consume_rate_limit", wraps=self.store.consume_rate_limit) as mock_consume:
                with self.assertRaises(HTTPException) as exc_info:
                    asyncio.run(auth.require_api_key(request, header_key=raw_key, bearer=None))

        self.assertEqual(exc_info.exception.status_code, 403)
        self.assertIn(API_KEY_SCOPE_TRAINING_WRITE, str(exc_info.exception.detail))
        mock_consume.assert_not_called()

    def test_require_api_key_allows_permitted_scope_and_consumes_rate_limit(self) -> None:
        _, raw_key = self.store.create_key(
            name="operator-key",
            tenant_id="tenant-a",
            role=API_KEY_ROLE_OPERATOR,
            scopes=[API_KEY_SCOPE_TRAINING_WRITE],
            expires_at=self.future_expiry,
        )
        request = _build_request("POST", "/jobs/job-1/train/start", "/jobs/{job_id}/train/start")

        with patch.object(auth, "api_key_store", self.store):
            with patch.object(self.store, "consume_rate_limit", wraps=self.store.consume_rate_limit) as mock_consume:
                principal = asyncio.run(auth.require_api_key(request, header_key=raw_key, bearer=None))

        self.assertEqual(principal.tenant_id, "tenant-a")
        self.assertEqual(principal.role, API_KEY_ROLE_OPERATOR)
        self.assertEqual(request.state.principal.key_id, principal.key_id)
        self.assertIn("minute_count", request.state.rate_limits)
        mock_consume.assert_called_once_with("tenant-a")

    def test_rotate_key_replaces_old_credential_and_preserves_lineage(self) -> None:
        original, raw_key = self.store.create_key(
            name="rotating-key",
            tenant_id="tenant-a",
            role=API_KEY_ROLE_OPERATOR,
            scopes=[API_KEY_SCOPE_TRAINING_WRITE],
            expires_at=self.future_expiry,
        )

        rotated = self.store.rotate_key(original.key_id, name="rotated-key")

        self.assertIsNotNone(rotated)
        assert rotated is not None
        replacement, replacement_raw_key = rotated
        old_record = self.store.get_key(original.key_id)
        new_record = self.store.get_key(replacement.key_id)

        self.assertIsNone(self.store.resolve_key(raw_key))
        self.assertIsNotNone(self.store.resolve_key(replacement_raw_key))
        self.assertIsNotNone(old_record)
        self.assertIsNotNone(new_record)
        assert old_record is not None
        assert new_record is not None
        self.assertEqual(old_record.status, "rotated")
        self.assertEqual(old_record.replaced_by_key_id, replacement.key_id)
        self.assertEqual(new_record.rotated_from_key_id, original.key_id)

    def test_update_key_can_reduce_permissions(self) -> None:
        principal, _ = self.store.create_key(
            name="operator-key",
            tenant_id="tenant-a",
            role=API_KEY_ROLE_OPERATOR,
            expires_at=self.future_expiry,
        )

        updated = self.store.update_key(
            principal.key_id,
            role=API_KEY_ROLE_READER,
            scopes=[API_KEY_SCOPE_JOBS_READ],
            expires_at=self.later_future_expiry,
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated.role, API_KEY_ROLE_READER)
        self.assertEqual(updated.scopes, [API_KEY_SCOPE_JOBS_READ])
        self.assertEqual(updated.expires_at, self.later_future_expiry)
        with self.assertRaises(HTTPException) as exc_info:
            self.store.require_permission(updated, "POST", "/jobs/{job_id}/train/start")
        self.assertEqual(exc_info.exception.status_code, 403)

        cleared = self.store.update_key(principal.key_id, expires_at="")
        self.assertIsNotNone(cleared)
        assert cleared is not None
        self.assertIsNone(cleared.expires_at)

    def test_log_usage_records_token_pricing_in_inr(self) -> None:
        self.store.log_usage(
            tenant_id="tenant-a",
            key_id="key-1",
            method="POST",
            path="/agent/action",
            route_template="/agent/action",
            status_code=200,
            duration_ms=12.5,
            billed_tokens=1000,
        )

        summary = self.store.get_usage_summary("tenant-a")
        rows = self.store.export_usage("tenant-a", days=7, limit=10)

        self.assertEqual(summary["requests_total"], 1)
        self.assertAlmostEqual(summary["estimated_cost_total_inr"], 2.79, places=6)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["billing_tier"], "inference_single")
        self.assertEqual(rows[0]["billed_tokens"], 1000)
        self.assertAlmostEqual(rows[0]["unit_price_inr"], 0.00279, places=6)
        self.assertAlmostEqual(rows[0]["estimated_cost_inr"], 2.79, places=6)

    def test_log_usage_does_not_charge_failed_request(self) -> None:
        self.store.log_usage(
            tenant_id="tenant-a",
            key_id="key-1",
            method="POST",
            path="/agent/action",
            route_template="/agent/action",
            status_code=500,
            duration_ms=8.0,
            billed_tokens=1000,
        )

        summary = self.store.get_usage_summary("tenant-a")
        rows = self.store.export_usage("tenant-a", days=7, limit=10)

        self.assertEqual(summary["requests_total"], 1)
        self.assertAlmostEqual(summary["estimated_cost_total_inr"], 0.0, places=6)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["estimated_cost_inr"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
