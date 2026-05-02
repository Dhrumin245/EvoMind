import asyncio
import hmac
import json
import os
import unittest
import uuid
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, Request

from api import auth
from api.auth import APIKeyStore
from api.payments import RazorpayClient
from tests.tmp_utils import cleanup_path


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
        "path_params": {},
        "route": SimpleNamespace(path=route_template),
    }
    return Request(scope, receive)


class BillingPaymentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(f"tests/.tmp/billing-payments-{uuid.uuid4().hex}.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = APIKeyStore(db_path=str(self.db_path))

    def tearDown(self) -> None:
        cleanup_path(self.db_path)

    def test_log_usage_debits_prepaid_balance_and_records_ledger(self) -> None:
        self.store.add_credit(
            tenant_id="tenant-a",
            amount_inr=10.0,
            description="Initial prepaid credit",
            reference_type="manual",
            reference_id="seed-credit",
        )

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

        account = self.store.get_billing_account("tenant-a")
        ledger = self.store.list_billing_ledger("tenant-a", limit=10)

        self.assertAlmostEqual(account["available_credit_inr"], 7.21, places=6)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["entry_type"], "usage_charge")
        self.assertAlmostEqual(ledger[0]["amount_inr"], -2.79, places=6)
        self.assertEqual(ledger[1]["entry_type"], "credit")

    def test_confirm_topup_payment_is_idempotent(self) -> None:
        topup = self.store.create_topup(
            tenant_id="tenant-a",
            provider="razorpay",
            amount_inr=250.0,
            provider_order_id="order_test_1",
            receipt="receipt_test_1",
            description="Top-up",
        )

        first = self.store.confirm_topup_payment(
            provider="razorpay",
            provider_order_id="order_test_1",
            provider_payment_id="pay_test_1",
            amount_inr=250.0,
            expected_tenant_id="tenant-a",
        )
        second = self.store.confirm_topup_payment(
            provider="razorpay",
            provider_order_id="order_test_1",
            provider_payment_id="pay_test_1",
            amount_inr=250.0,
            expected_tenant_id="tenant-a",
        )

        account = self.store.get_billing_account("tenant-a")
        ledger = self.store.list_billing_ledger("tenant-a", limit=10)

        self.assertEqual(topup["status"], "created")
        self.assertEqual(first["topup"]["status"], "paid")
        self.assertEqual(second["topup"]["status"], "paid")
        self.assertAlmostEqual(account["available_credit_inr"], 250.0, places=6)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["entry_type"], "topup_credit")

    def test_require_api_key_blocks_chargeable_route_when_prepaid_is_enabled(self) -> None:
        _, raw_key = self.store.create_key(name="operator", tenant_id="tenant-a")
        chargeable_request = _build_request("POST", "/agent/action", "/agent/action")
        billing_request = _build_request("GET", "/billing/account", "/billing/account")

        with patch.dict(os.environ, {"EVOMIND_PREPAID_REQUIRED": "true"}, clear=False):
            with patch.object(auth, "api_key_store", self.store):
                with self.assertRaises(HTTPException) as exc_info:
                    asyncio.run(auth.require_api_key(chargeable_request, header_key=raw_key, bearer=None))
                principal = asyncio.run(auth.require_api_key(billing_request, header_key=raw_key, bearer=None))

        self.assertEqual(exc_info.exception.status_code, 402)
        self.assertEqual(principal.tenant_id, "tenant-a")

    def test_razorpay_signature_verification_and_webhook_parsing(self) -> None:
        secret = "rzp_secret_123"
        webhook_secret = "rzp_webhook_secret_456"
        order_id = "order_123"
        payment_id = "pay_123"
        checkout_signature = hmac.new(
            secret.encode("utf-8"),
            f"{order_id}|{payment_id}".encode("utf-8"),
            sha256,
        ).hexdigest()
        webhook_body = json.dumps(
            {
                "event": "order.paid",
                "payload": {
                    "payment": {
                        "entity": {
                            "id": payment_id,
                            "order_id": order_id,
                            "status": "captured",
                            "amount": 49900,
                        }
                    },
                    "order": {
                        "entity": {
                            "id": order_id,
                            "status": "paid",
                            "amount_paid": 49900,
                        }
                    },
                },
            }
        ).encode("utf-8")
        webhook_signature = hmac.new(
            webhook_secret.encode("utf-8"),
            webhook_body,
            sha256,
        ).hexdigest()

        with patch.dict(
            os.environ,
            {
                "EVOMIND_RAZORPAY_KEY_ID": "rzp_test_key",
                "EVOMIND_RAZORPAY_KEY_SECRET": secret,
                "EVOMIND_RAZORPAY_WEBHOOK_SECRET": webhook_secret,
            },
            clear=False,
        ):
            RazorpayClient.verify_checkout_signature(
                order_id=order_id,
                payment_id=payment_id,
                signature=checkout_signature,
            )
            RazorpayClient.verify_webhook_signature(body=webhook_body, signature=webhook_signature)
            parsed = RazorpayClient.parse_webhook(webhook_body)
            extracted = RazorpayClient.extract_captured_payment(parsed)

        self.assertIsNotNone(extracted)
        assert extracted is not None
        self.assertEqual(extracted["order_id"], order_id)
        self.assertEqual(extracted["payment_id"], payment_id)
        self.assertAlmostEqual(extracted["amount_inr"], 499.0, places=6)


if __name__ == "__main__":
    unittest.main()
