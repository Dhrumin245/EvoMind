import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


logger = logging.getLogger(__name__)


DEFAULT_DELIVERY_TIMEOUT_SECONDS = 5
DEFAULT_MAX_DELIVERY_ATTEMPTS = 5
DEFAULT_DELIVERY_BATCH_SIZE = 20
DELIVERY_POLL_INTERVAL_SECONDS = 2
RETRY_DELAYS_SECONDS = [0, 60, 300, 1800, 7200]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class JobEventRecord:
    event_id: str
    tenant_id: str
    job_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at,
        }


@dataclass
class WebhookRecord:
    webhook_id: str
    tenant_id: str
    url: str
    description: str
    subscribed_events: List[str]
    status: str
    created_at: str
    updated_at: str
    last_delivery_at: Optional[str]
    last_delivery_status: Optional[str]
    last_delivery_error: Optional[str]
    secret: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "webhook_id": self.webhook_id,
            "tenant_id": self.tenant_id,
            "url": self.url,
            "description": self.description,
            "subscribed_events": self.subscribed_events,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_delivery_at": self.last_delivery_at,
            "last_delivery_status": self.last_delivery_status,
            "last_delivery_error": self.last_delivery_error,
        }


@dataclass
class WebhookDeliveryAttemptRecord:
    attempt_id: str
    delivery_id: str
    attempt_number: int
    status: str
    response_status_code: Optional[int]
    error_message: Optional[str]
    created_at: str
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "delivery_id": self.delivery_id,
            "attempt_number": self.attempt_number,
            "status": self.status,
            "response_status_code": self.response_status_code,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class WebhookDeliveryRecord:
    delivery_id: str
    webhook_id: str
    event_id: str
    tenant_id: str
    job_id: str
    event_type: str
    status: str
    attempt_count: int
    max_attempts: int
    next_retry_at: Optional[str]
    delivered_at: Optional[str]
    last_error: Optional[str]
    created_at: str
    updated_at: str
    attempts: Optional[List[WebhookDeliveryAttemptRecord]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "webhook_id": self.webhook_id,
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "event_type": self.event_type,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "next_retry_at": self.next_retry_at,
            "delivered_at": self.delivered_at,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "attempts": [item.to_dict() for item in (self.attempts or [])],
        }


class EventManager:
    def __init__(self, db_path: str = "data/api_events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._delivery_worker_task: Optional[asyncio.Task] = None
        self._delivery_lock = asyncio.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS webhooks (
                    webhook_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    secret TEXT,
                    subscribed_events_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_delivery_at TEXT,
                    last_delivery_status TEXT,
                    last_delivery_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    webhook_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 5,
                    next_retry_at TEXT,
                    delivered_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_delivery_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    delivery_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    response_status_code INTEGER,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_events_tenant_job_created
                ON job_events (tenant_id, job_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhooks_tenant_status
                ON webhooks (tenant_id, status)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_webhook_deliveries_webhook_event
                ON webhook_deliveries (webhook_id, event_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_due
                ON webhook_deliveries (status, next_retry_at, updated_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_delivery_attempts_delivery
                ON webhook_delivery_attempts (delivery_id, attempt_number)
                """
            )
            conn.commit()

    @staticmethod
    def _parse_subscribed_events(value: Optional[List[str]]) -> List[str]:
        if not value:
            return []
        normalized = sorted({str(item).strip() for item in value if str(item).strip()})
        return normalized

    @staticmethod
    def _validate_webhook_url(url: str) -> str:
        parsed = urllib_parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Webhook URL must be a valid http or https URL")
        return url

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> JobEventRecord:
        payload_raw = row["payload_json"] if row["payload_json"] else "{}"
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return JobEventRecord(
            event_id=str(row["event_id"]),
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            event_type=str(row["event_type"]),
            payload=payload,
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _row_to_webhook(row: sqlite3.Row) -> WebhookRecord:
        subscribed_events_raw = row["subscribed_events_json"] or "[]"
        try:
            subscribed_events = json.loads(subscribed_events_raw)
        except json.JSONDecodeError:
            subscribed_events = []
        if not isinstance(subscribed_events, list):
            subscribed_events = []
        return WebhookRecord(
            webhook_id=str(row["webhook_id"]),
            tenant_id=str(row["tenant_id"]),
            url=str(row["url"]),
            description=str(row["description"] or ""),
            subscribed_events=[str(item) for item in subscribed_events],
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_delivery_at=row["last_delivery_at"],
            last_delivery_status=row["last_delivery_status"],
            last_delivery_error=row["last_delivery_error"],
            secret=row["secret"],
        )

    @staticmethod
    def _row_to_delivery(row: sqlite3.Row) -> WebhookDeliveryRecord:
        return WebhookDeliveryRecord(
            delivery_id=str(row["delivery_id"]),
            webhook_id=str(row["webhook_id"]),
            event_id=str(row["event_id"]),
            tenant_id=str(row["tenant_id"]),
            job_id=str(row["job_id"]),
            event_type=str(row["event_type"]),
            status=str(row["status"]),
            attempt_count=int(row["attempt_count"] or 0),
            max_attempts=int(row["max_attempts"] or DEFAULT_MAX_DELIVERY_ATTEMPTS),
            next_retry_at=row["next_retry_at"],
            delivered_at=row["delivered_at"],
            last_error=row["last_error"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            attempts=[],
        )

    @staticmethod
    def _row_to_attempt(row: sqlite3.Row) -> WebhookDeliveryAttemptRecord:
        response_status_code = row["response_status_code"]
        return WebhookDeliveryAttemptRecord(
            attempt_id=str(row["attempt_id"]),
            delivery_id=str(row["delivery_id"]),
            attempt_number=int(row["attempt_number"]),
            status=str(row["status"]),
            response_status_code=int(response_status_code) if response_status_code is not None else None,
            error_message=row["error_message"],
            created_at=str(row["created_at"]),
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _retry_delay_seconds(attempt_number: int) -> int:
        safe_index = min(max(0, int(attempt_number)), len(RETRY_DELAYS_SECONDS) - 1)
        return int(RETRY_DELAYS_SECONDS[safe_index])

    @staticmethod
    def _add_seconds(iso_timestamp: str, seconds: int) -> str:
        base = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (base + timedelta(seconds=int(seconds))).strftime("%Y-%m-%dT%H:%M:%SZ")

    def create_webhook(
        self,
        tenant_id: str,
        url: str,
        description: Optional[str] = None,
        subscribed_events: Optional[List[str]] = None,
        secret: Optional[str] = None,
    ) -> WebhookRecord:
        webhook_id = secrets.token_hex(8)
        now = _utc_now()
        normalized_url = self._validate_webhook_url(url)
        normalized_events = self._parse_subscribed_events(subscribed_events)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO webhooks (
                    webhook_id,
                    tenant_id,
                    url,
                    description,
                    secret,
                    subscribed_events_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    webhook_id,
                    tenant_id,
                    normalized_url,
                    description or "",
                    secret,
                    json.dumps(normalized_events),
                    now,
                    now,
                ),
            )
            conn.commit()

        return WebhookRecord(
            webhook_id=webhook_id,
            tenant_id=tenant_id,
            url=normalized_url,
            description=description or "",
            subscribed_events=normalized_events,
            status="active",
            created_at=now,
            updated_at=now,
            last_delivery_at=None,
            last_delivery_status=None,
            last_delivery_error=None,
            secret=secret,
        )

    def list_webhooks(self, tenant_id: str) -> List[WebhookRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM webhooks
                WHERE tenant_id = ? AND status != 'deleted'
                ORDER BY created_at DESC
                """,
                (tenant_id,),
            ).fetchall()
        return [self._row_to_webhook(row) for row in rows]

    def delete_webhook(self, tenant_id: str, webhook_id: str) -> bool:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE webhooks
                SET status = 'deleted', updated_at = ?
                WHERE tenant_id = ? AND webhook_id = ? AND status != 'deleted'
                """,
                (now, tenant_id, webhook_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def start_worker(self) -> None:
        if self._delivery_worker_task is not None and not self._delivery_worker_task.done():
            return
        self._delivery_worker_task = asyncio.create_task(self._delivery_worker_loop())

    async def stop_worker(self) -> None:
        task = self._delivery_worker_task
        self._delivery_worker_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _matching_webhooks(self, tenant_id: str, event_type: str) -> List[WebhookRecord]:
        matches: List[WebhookRecord] = []
        for webhook in self.list_webhooks(tenant_id):
            if webhook.status != "active":
                continue
            if not webhook.subscribed_events or event_type in webhook.subscribed_events:
                matches.append(webhook)
        return matches

    def _get_webhook(self, webhook_id: str) -> Optional[WebhookRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM webhooks WHERE webhook_id = ?",
                (webhook_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_webhook(row)

    def _get_event(self, event_id: str) -> Optional[JobEventRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def _enqueue_deliveries(
        self,
        event: JobEventRecord,
        webhooks: List[WebhookRecord],
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            for webhook in webhooks:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO webhook_deliveries (
                        delivery_id,
                        webhook_id,
                        event_id,
                        tenant_id,
                        job_id,
                        event_type,
                        status,
                        attempt_count,
                        max_attempts,
                        next_retry_at,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                    """,
                    (
                        secrets.token_hex(12),
                        webhook.webhook_id,
                        event.event_id,
                        event.tenant_id,
                        event.job_id,
                        event.event_type,
                        DEFAULT_MAX_DELIVERY_ATTEMPTS,
                        now,
                        now,
                        now,
                    ),
                )
            conn.commit()

    def list_deliveries(
        self,
        tenant_id: str,
        webhook_id: str,
        limit: int = 50,
        status: Optional[str] = None,
    ) -> List[WebhookDeliveryRecord]:
        clauses = ["tenant_id = ?", "webhook_id = ?"]
        params: List[Any] = [tenant_id, webhook_id]
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, int(limit)))

        query = (
            "SELECT * FROM webhook_deliveries WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            delivery_rows = conn.execute(query, params).fetchall()
            deliveries = [self._row_to_delivery(row) for row in delivery_rows]
            if not deliveries:
                return []

            delivery_ids = [item.delivery_id for item in deliveries]
            placeholders = ",".join("?" for _ in delivery_ids)
            attempt_rows = conn.execute(
                f"""
                SELECT *
                FROM webhook_delivery_attempts
                WHERE delivery_id IN ({placeholders})
                ORDER BY attempt_number DESC, created_at DESC
                """,
                delivery_ids,
            ).fetchall()

        attempts_by_delivery: Dict[str, List[WebhookDeliveryAttemptRecord]] = {}
        for row in attempt_rows:
            attempt = self._row_to_attempt(row)
            attempts_by_delivery.setdefault(attempt.delivery_id, []).append(attempt)

        for delivery in deliveries:
            delivery.attempts = attempts_by_delivery.get(delivery.delivery_id, [])

        return deliveries

    def _record_event(
        self,
        tenant_id: str,
        job_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> JobEventRecord:
        event = JobEventRecord(
            event_id=secrets.token_hex(12),
            tenant_id=tenant_id,
            job_id=job_id,
            event_type=event_type,
            payload=payload or {},
            created_at=_utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events (
                    event_id,
                    tenant_id,
                    job_id,
                    event_type,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.tenant_id,
                    event.job_id,
                    event.event_type,
                    json.dumps(event.payload, default=str),
                    event.created_at,
                ),
            )
            conn.commit()
        return event

    async def emit_event(
        self,
        tenant_id: str,
        job_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> JobEventRecord:
        event = self._record_event(
            tenant_id=tenant_id,
            job_id=job_id,
            event_type=event_type,
            payload=payload,
        )
        matching_webhooks = self._matching_webhooks(tenant_id, event_type)
        if matching_webhooks:
            self._enqueue_deliveries(event, matching_webhooks)
            asyncio.create_task(self.process_pending_deliveries_once())
        return event

    def list_events(
        self,
        tenant_id: str,
        job_id: str,
        limit: int = 50,
        event_type: Optional[str] = None,
    ) -> List[JobEventRecord]:
        clauses = ["tenant_id = ?", "job_id = ?"]
        params: List[Any] = [tenant_id, job_id]
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        params.append(max(1, int(limit)))

        query = (
            "SELECT * FROM job_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    @staticmethod
    def _signature(secret: str, body: bytes) -> str:
        digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    def _update_delivery_result(
        self,
        webhook_id: str,
        status_value: str,
        error_message: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE webhooks
                SET
                    last_delivery_at = ?,
                    last_delivery_status = ?,
                    last_delivery_error = ?,
                    updated_at = ?
                WHERE webhook_id = ?
                """,
                (now, status_value, error_message, now, webhook_id),
            )
            conn.commit()

    def _record_attempt(
        self,
        delivery_id: str,
        attempt_number: int,
        status_value: str,
        response_status_code: Optional[int],
        error_message: Optional[str],
        created_at: str,
        completed_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO webhook_delivery_attempts (
                    attempt_id,
                    delivery_id,
                    attempt_number,
                    status,
                    response_status_code,
                    error_message,
                    created_at,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    secrets.token_hex(12),
                    delivery_id,
                    int(attempt_number),
                    status_value,
                    response_status_code,
                    error_message,
                    created_at,
                    completed_at,
                ),
            )
            conn.commit()

    def _post_webhook(self, webhook: WebhookRecord, event: JobEventRecord) -> int:
        payload = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "job_id": event.job_id,
            "created_at": event.created_at,
            "payload": event.payload,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Evomind-Webhooks/1.0",
            "X-Evomind-Event": event.event_type,
            "X-Evomind-Webhook": webhook.webhook_id,
        }
        if webhook.secret:
            headers["X-Evomind-Signature"] = self._signature(webhook.secret, body)

        request = urllib_request.Request(
            webhook.url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(
            request,
            timeout=DEFAULT_DELIVERY_TIMEOUT_SECONDS,
        ) as response:
            status_code = getattr(response, "status", 200)
            if int(status_code) >= 400:
                raise RuntimeError(f"Webhook returned status {status_code}")
            return int(status_code)

    def _set_delivery_state(
        self,
        delivery_id: str,
        status_value: str,
        attempt_count: int,
        next_retry_at: Optional[str],
        delivered_at: Optional[str],
        last_error: Optional[str],
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE webhook_deliveries
                SET
                    status = ?,
                    attempt_count = ?,
                    next_retry_at = ?,
                    delivered_at = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE delivery_id = ?
                """,
                (
                    status_value,
                    int(attempt_count),
                    next_retry_at,
                    delivered_at,
                    last_error,
                    now,
                    delivery_id,
                ),
            )
            conn.commit()

    async def _process_delivery(self, delivery: WebhookDeliveryRecord) -> None:
        webhook = self._get_webhook(delivery.webhook_id)
        event = self._get_event(delivery.event_id)
        attempt_number = int(delivery.attempt_count) + 1
        started_at = _utc_now()

        if webhook is None or webhook.status != "active":
            error_message = "Webhook is missing or inactive"
            self._record_attempt(
                delivery.delivery_id,
                attempt_number,
                "failed",
                None,
                error_message,
                started_at,
                _utc_now(),
            )
            self._set_delivery_state(
                delivery.delivery_id,
                "failed",
                attempt_number,
                None,
                None,
                error_message,
            )
            return

        if event is None:
            error_message = "Event payload not found"
            self._record_attempt(
                delivery.delivery_id,
                attempt_number,
                "failed",
                None,
                error_message,
                started_at,
                _utc_now(),
            )
            self._set_delivery_state(
                delivery.delivery_id,
                "failed",
                attempt_number,
                None,
                None,
                error_message,
            )
            self._update_delivery_result(webhook.webhook_id, "failed", error_message)
            return

        try:
            response_status_code = await asyncio.to_thread(self._post_webhook, webhook, event)
            completed_at = _utc_now()
            self._record_attempt(
                delivery.delivery_id,
                attempt_number,
                "delivered",
                response_status_code,
                None,
                started_at,
                completed_at,
            )
            self._set_delivery_state(
                delivery.delivery_id,
                "delivered",
                attempt_number,
                None,
                completed_at,
                None,
            )
            self._update_delivery_result(webhook.webhook_id, "delivered", None)
        except urllib_error.HTTPError as exc:
            await self._handle_delivery_failure(
                delivery,
                webhook,
                attempt_number,
                started_at,
                f"HTTP {exc.code}",
                exc.code,
            )
        except urllib_error.URLError as exc:
            await self._handle_delivery_failure(
                delivery,
                webhook,
                attempt_number,
                started_at,
                str(exc.reason),
                None,
            )
        except Exception as exc:
            await self._handle_delivery_failure(
                delivery,
                webhook,
                attempt_number,
                started_at,
                str(exc),
                None,
            )

    async def _handle_delivery_failure(
        self,
        delivery: WebhookDeliveryRecord,
        webhook: WebhookRecord,
        attempt_number: int,
        started_at: str,
        error_message: str,
        response_status_code: Optional[int],
    ) -> None:
        logger.warning(
            "Webhook delivery failed tenant=%s webhook=%s event=%s attempt=%s error=%s",
            webhook.tenant_id,
            webhook.webhook_id,
            delivery.event_type,
            attempt_number,
            error_message,
        )
        completed_at = _utc_now()
        self._record_attempt(
            delivery.delivery_id,
            attempt_number,
            "failed",
            response_status_code,
            error_message,
            started_at,
            completed_at,
        )

        if attempt_number >= int(delivery.max_attempts):
            self._set_delivery_state(
                delivery.delivery_id,
                "failed",
                attempt_number,
                None,
                None,
                error_message,
            )
        else:
            next_retry_at = self._add_seconds(
                completed_at,
                self._retry_delay_seconds(attempt_number),
            )
            self._set_delivery_state(
                delivery.delivery_id,
                "retrying",
                attempt_number,
                next_retry_at,
                None,
                error_message,
            )
        self._update_delivery_result(webhook.webhook_id, "failed", error_message)

    async def process_pending_deliveries_once(
        self,
        limit: int = DEFAULT_DELIVERY_BATCH_SIZE,
    ) -> int:
        async with self._delivery_lock:
            due_at = _utc_now()
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM webhook_deliveries
                    WHERE status IN ('pending', 'retrying')
                      AND (next_retry_at IS NULL OR next_retry_at <= ?)
                    ORDER BY next_retry_at ASC, created_at ASC
                    LIMIT ?
                    """,
                    (due_at, max(1, int(limit))),
                ).fetchall()

            deliveries = [self._row_to_delivery(row) for row in rows]
            if not deliveries:
                return 0

            await asyncio.gather(
                *(self._process_delivery(delivery) for delivery in deliveries),
                return_exceptions=True,
            )
            return len(deliveries)

    async def _delivery_worker_loop(self) -> None:
        try:
            while True:
                try:
                    await self.process_pending_deliveries_once()
                except Exception as exc:
                    logger.warning("Webhook delivery worker loop error: %s", exc)
                await asyncio.sleep(DELIVERY_POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
