import base64
import hashlib
import hmac
import json
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from api.env_utils import read_env_value


DEFAULT_RAZORPAY_BASE_URL = "https://api.razorpay.com"


def _round_inr(value: float) -> float:
    return round(float(value), 6)


def inr_to_subunits(amount_inr: float) -> int:
    normalized = float(amount_inr)
    if normalized <= 0:
        raise ValueError("Amount must be greater than zero")
    return int(round(normalized * 100.0))


def inr_from_subunits(amount_subunits: Any) -> float:
    return _round_inr(float(amount_subunits or 0) / 100.0)


class RazorpayClient:
    @staticmethod
    def _config() -> Dict[str, str]:
        return {
            "key_id": str(read_env_value("EVOMIND_RAZORPAY_KEY_ID", "") or "").strip(),
            "key_secret": str(read_env_value("EVOMIND_RAZORPAY_KEY_SECRET", "") or "").strip(),
            "webhook_secret": str(read_env_value("EVOMIND_RAZORPAY_WEBHOOK_SECRET", "") or "").strip(),
            "base_url": str(
                read_env_value("EVOMIND_RAZORPAY_BASE_URL", DEFAULT_RAZORPAY_BASE_URL) or DEFAULT_RAZORPAY_BASE_URL
            ).strip().rstrip("/"),
        }

    @classmethod
    def require_checkout_config(cls) -> Dict[str, str]:
        config = cls._config()
        if not config["key_id"] or not config["key_secret"]:
            raise RuntimeError(
                "Razorpay checkout is not configured. Set EVOMIND_RAZORPAY_KEY_ID and EVOMIND_RAZORPAY_KEY_SECRET."
            )
        return config

    @classmethod
    def require_webhook_secret(cls) -> str:
        secret = cls._config()["webhook_secret"]
        if not secret:
            raise RuntimeError(
                "Razorpay webhooks are not configured. Set EVOMIND_RAZORPAY_WEBHOOK_SECRET."
            )
        return secret

    @classmethod
    def checkout_key_id(cls) -> str:
        return cls.require_checkout_config()["key_id"]

    @classmethod
    def create_order(
        cls,
        *,
        amount_inr: float,
        receipt: str,
        notes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        config = cls.require_checkout_config()
        payload = {
            "amount": inr_to_subunits(amount_inr),
            "currency": "INR",
            "receipt": str(receipt),
            "notes": {str(key): str(value) for key, value in (notes or {}).items()},
        }
        token = base64.b64encode(f"{config['key_id']}:{config['key_secret']}".encode("utf-8")).decode("ascii")
        request = urllib_request.Request(
            url=f"{config['base_url']}/v1/orders",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=15) as response:
                raw_body = response.read()
        except urllib_error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Razorpay order creation failed with HTTP {exc.code}: {raw_error}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Razorpay order creation failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Razorpay returned an invalid JSON response") from exc
        if not isinstance(parsed, dict) or not str(parsed.get("id") or "").strip():
            raise RuntimeError("Razorpay order response did not include an order id")
        return parsed

    @classmethod
    def verify_checkout_signature(
        cls,
        *,
        order_id: str,
        payment_id: str,
        signature: str,
    ) -> None:
        config = cls.require_checkout_config()
        message = f"{order_id}|{payment_id}".encode("utf-8")
        expected_signature = hmac.new(
            config["key_secret"].encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, str(signature or "").strip()):
            raise PermissionError("Invalid Razorpay checkout signature")

    @classmethod
    def verify_webhook_signature(cls, *, body: bytes, signature: str) -> None:
        expected_signature = hmac.new(
            cls.require_webhook_secret().encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, str(signature or "").strip()):
            raise PermissionError("Invalid Razorpay webhook signature")

    @staticmethod
    def parse_webhook(body: bytes) -> Dict[str, Any]:
        try:
            parsed = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid Razorpay webhook payload") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Invalid Razorpay webhook payload")
        return parsed

    @staticmethod
    def extract_captured_payment(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_name = str(payload.get("event") or "").strip()
        if event_name not in {"order.paid", "payment.captured"}:
            return None

        payload_root = payload.get("payload")
        if not isinstance(payload_root, dict):
            return None
        payment_wrapper = payload_root.get("payment")
        order_wrapper = payload_root.get("order")
        payment_entity = payment_wrapper.get("entity") if isinstance(payment_wrapper, dict) else {}
        order_entity = order_wrapper.get("entity") if isinstance(order_wrapper, dict) else {}
        if not isinstance(payment_entity, dict):
            payment_entity = {}
        if not isinstance(order_entity, dict):
            order_entity = {}

        order_id = str(payment_entity.get("order_id") or order_entity.get("id") or "").strip()
        payment_id = str(payment_entity.get("id") or "").strip()
        if not order_id or not payment_id:
            return None

        payment_status = str(payment_entity.get("status") or "").strip().lower()
        order_status = str(order_entity.get("status") or "").strip().lower()
        if event_name == "payment.captured" and payment_status and payment_status != "captured":
            return None
        if event_name == "order.paid" and order_status and order_status != "paid":
            return None

        amount_subunits = (
            payment_entity.get("amount")
            if payment_entity.get("amount") is not None
            else order_entity.get("amount_paid")
        )
        return {
            "event": event_name,
            "order_id": order_id,
            "payment_id": payment_id,
            "amount_inr": inr_from_subunits(amount_subunits),
        }
