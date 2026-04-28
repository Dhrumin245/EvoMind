import asyncio
import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from api.logging_utils import merge_log_context
from api.storage import (
    api_auth_db_path,
    auto_increment_primary_key_sql,
    column_names,
    connect_database,
    resolve_db_target,
)


DEFAULT_REQUESTS_PER_MINUTE = 120
DEFAULT_REQUESTS_PER_DAY = 10000
DEFAULT_MAX_JOBS = 5
DEFAULT_EXPORT_LIMIT = 5000
DEFAULT_BILLING_LEDGER_LIMIT = 100
API_KEY_STATUS_ACTIVE = "active"
API_KEY_STATUS_REVOKED = "revoked"
API_KEY_STATUS_EXPIRED = "expired"
API_KEY_STATUS_ROTATED = "rotated"
API_KEY_ROLE_ADMIN = "admin"
API_KEY_ROLE_OPERATOR = "operator"
API_KEY_ROLE_READER = "reader"
API_KEY_SCOPE_ALL = "*"
API_KEY_SCOPE_BILLING_READ = "billing:read"
API_KEY_SCOPE_BILLING_WRITE = "billing:write"
API_KEY_SCOPE_USAGE_READ = "usage:read"
API_KEY_SCOPE_WEBHOOKS_READ = "webhooks:read"
API_KEY_SCOPE_WEBHOOKS_WRITE = "webhooks:write"
API_KEY_SCOPE_JOBS_READ = "jobs:read"
API_KEY_SCOPE_JOBS_WRITE = "jobs:write"
API_KEY_SCOPE_JOB_EVENTS_READ = "job_events:read"
API_KEY_SCOPE_TRAINING_READ = "training:read"
API_KEY_SCOPE_TRAINING_WRITE = "training:write"
API_KEY_SCOPE_CHECKPOINTS_READ = "checkpoints:read"
API_KEY_SCOPE_CHECKPOINTS_WRITE = "checkpoints:write"
API_KEY_SCOPE_AGENT_READ = "agent:read"
API_KEY_SCOPE_AGENT_INVOKE = "agent:invoke"
API_KEY_SCOPE_GENOMES_READ = "genomes:read"

API_KEY_PERMISSION_RULES: Dict[tuple[str, str], tuple[str, ...]] = {}
for route_template in (
    "/billing/account",
    "/billing/ledger",
    "/usage/limits",
    "/usage/summary",
    "/usage/billing-tiers",
    "/usage/export",
):
    required_scope = API_KEY_SCOPE_BILLING_READ if route_template.startswith("/billing/") else API_KEY_SCOPE_USAGE_READ
    API_KEY_PERMISSION_RULES[("GET", route_template)] = (required_scope,)
for method, route_template, scope in (
    ("POST", "/billing/topups", API_KEY_SCOPE_BILLING_WRITE),
    ("POST", "/billing/topups/confirm", API_KEY_SCOPE_BILLING_WRITE),
    ("GET", "/webhooks", API_KEY_SCOPE_WEBHOOKS_READ),
    ("POST", "/webhooks", API_KEY_SCOPE_WEBHOOKS_WRITE),
    ("DELETE", "/webhooks/{webhook_id}", API_KEY_SCOPE_WEBHOOKS_WRITE),
    ("GET", "/webhooks/{webhook_id}/deliveries", API_KEY_SCOPE_WEBHOOKS_READ),
    ("POST", "/jobs", API_KEY_SCOPE_JOBS_WRITE),
    ("GET", "/jobs", API_KEY_SCOPE_JOBS_READ),
    ("GET", "/jobs/{job_id}", API_KEY_SCOPE_JOBS_READ),
    ("GET", "/jobs/{job_id}/events", API_KEY_SCOPE_JOB_EVENTS_READ),
    ("POST", "/jobs/{job_id}/train/start", API_KEY_SCOPE_TRAINING_WRITE),
    ("POST", "/jobs/{job_id}/train/stop", API_KEY_SCOPE_TRAINING_WRITE),
    ("POST", "/jobs/{job_id}/train/resume", API_KEY_SCOPE_TRAINING_WRITE),
    ("GET", "/jobs/{job_id}/train/status", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/jobs/{job_id}/train/insights", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/jobs/{job_id}/train/metrics", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/jobs/{job_id}/train/checkpoints", API_KEY_SCOPE_CHECKPOINTS_READ),
    ("POST", "/jobs/{job_id}/train/checkpoints", API_KEY_SCOPE_CHECKPOINTS_WRITE),
    ("POST", "/jobs/{job_id}/agent/action", API_KEY_SCOPE_AGENT_INVOKE),
    ("POST", "/jobs/{job_id}/agent/action/batch", API_KEY_SCOPE_AGENT_INVOKE),
    ("GET", "/jobs/{job_id}/agent/info", API_KEY_SCOPE_AGENT_READ),
    ("GET", "/jobs/{job_id}/genomes", API_KEY_SCOPE_GENOMES_READ),
    ("GET", "/jobs/{job_id}/genomes/{genome_id}", API_KEY_SCOPE_GENOMES_READ),
    ("POST", "/train/start", API_KEY_SCOPE_TRAINING_WRITE),
    ("POST", "/train/stop", API_KEY_SCOPE_TRAINING_WRITE),
    ("POST", "/train/resume", API_KEY_SCOPE_TRAINING_WRITE),
    ("GET", "/train/status", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/train/insights", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/train/metrics", API_KEY_SCOPE_TRAINING_READ),
    ("GET", "/train/checkpoints", API_KEY_SCOPE_CHECKPOINTS_READ),
    ("POST", "/train/checkpoints", API_KEY_SCOPE_CHECKPOINTS_WRITE),
    ("POST", "/agent/action", API_KEY_SCOPE_AGENT_INVOKE),
    ("POST", "/agent/action/batch", API_KEY_SCOPE_AGENT_INVOKE),
    ("GET", "/agent/info", API_KEY_SCOPE_AGENT_READ),
    ("GET", "/genomes", API_KEY_SCOPE_GENOMES_READ),
    ("GET", "/genomes/{genome_id}", API_KEY_SCOPE_GENOMES_READ),
):
    API_KEY_PERMISSION_RULES[(method, route_template)] = (scope,)

ALL_API_KEY_SCOPES = {
    API_KEY_SCOPE_BILLING_READ,
    API_KEY_SCOPE_BILLING_WRITE,
    API_KEY_SCOPE_USAGE_READ,
    API_KEY_SCOPE_WEBHOOKS_READ,
    API_KEY_SCOPE_WEBHOOKS_WRITE,
    API_KEY_SCOPE_JOBS_READ,
    API_KEY_SCOPE_JOBS_WRITE,
    API_KEY_SCOPE_JOB_EVENTS_READ,
    API_KEY_SCOPE_TRAINING_READ,
    API_KEY_SCOPE_TRAINING_WRITE,
    API_KEY_SCOPE_CHECKPOINTS_READ,
    API_KEY_SCOPE_CHECKPOINTS_WRITE,
    API_KEY_SCOPE_AGENT_READ,
    API_KEY_SCOPE_AGENT_INVOKE,
    API_KEY_SCOPE_GENOMES_READ,
}
ROLE_DEFAULT_SCOPES = {
    API_KEY_ROLE_ADMIN: [API_KEY_SCOPE_ALL],
    API_KEY_ROLE_OPERATOR: [
        API_KEY_SCOPE_BILLING_READ,
        API_KEY_SCOPE_BILLING_WRITE,
        API_KEY_SCOPE_USAGE_READ,
        API_KEY_SCOPE_WEBHOOKS_READ,
        API_KEY_SCOPE_WEBHOOKS_WRITE,
        API_KEY_SCOPE_JOBS_READ,
        API_KEY_SCOPE_JOBS_WRITE,
        API_KEY_SCOPE_JOB_EVENTS_READ,
        API_KEY_SCOPE_TRAINING_READ,
        API_KEY_SCOPE_TRAINING_WRITE,
        API_KEY_SCOPE_CHECKPOINTS_READ,
        API_KEY_SCOPE_CHECKPOINTS_WRITE,
        API_KEY_SCOPE_AGENT_READ,
        API_KEY_SCOPE_AGENT_INVOKE,
        API_KEY_SCOPE_GENOMES_READ,
    ],
    API_KEY_ROLE_READER: [
        API_KEY_SCOPE_BILLING_READ,
        API_KEY_SCOPE_USAGE_READ,
        API_KEY_SCOPE_WEBHOOKS_READ,
        API_KEY_SCOPE_JOBS_READ,
        API_KEY_SCOPE_JOB_EVENTS_READ,
        API_KEY_SCOPE_TRAINING_READ,
        API_KEY_SCOPE_CHECKPOINTS_READ,
        API_KEY_SCOPE_AGENT_READ,
        API_KEY_SCOPE_GENOMES_READ,
    ],
}

USD_TO_INR_REFERENCE_RATE = 93.0
# Preserve the previous relative price ladder by converting the old USD/request
# prices into INR/token rates using a 1k-token reference block.
TOKEN_PRICE_REFERENCE = 1000.0


def _inr_price_per_token(usd_price_per_request: float) -> float:
    return round(
        (float(usd_price_per_request) * USD_TO_INR_REFERENCE_RATE) / TOKEN_PRICE_REFERENCE,
        6,
    )


BILLING_TIERS: Dict[str, Dict[str, Any]] = {
    "admin_free": {
        "unit_name": "token",
        "unit_price_inr": 0.0,
    },
    "job_write": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.01),
    },
    "job_read": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.005),
    },
    "job_events": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.005),
    },
    "training_control": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.02),
    },
    "training_snapshot": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.01),
    },
    "training_metrics": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.015),
    },
    "artifact_read": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.01),
    },
    "artifact_write": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.02),
    },
    "inference_single": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.03),
    },
    "inference_batch": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.01),
    },
    "model_catalog": {
        "unit_name": "token",
        "unit_price_inr": _inr_price_per_token(0.005),
    },
}

ROUTE_BILLING_CATALOG = [
    {
        "method": "GET",
        "route_template": "/billing/account",
        "billing_tier": "admin_free",
        "description": "Read tenant prepaid balance and billing summary",
    },
    {
        "method": "GET",
        "route_template": "/billing/ledger",
        "billing_tier": "admin_free",
        "description": "Read tenant billing ledger history",
    },
    {
        "method": "POST",
        "route_template": "/billing/topups",
        "billing_tier": "admin_free",
        "description": "Create a Razorpay prepaid credit top-up order",
    },
    {
        "method": "POST",
        "route_template": "/billing/topups/confirm",
        "billing_tier": "admin_free",
        "description": "Confirm a successful Razorpay prepaid credit payment",
    },
    {
        "method": "GET",
        "route_template": "/usage/limits",
        "billing_tier": "admin_free",
        "description": "Read tenant quota configuration",
    },
    {
        "method": "GET",
        "route_template": "/usage/summary",
        "billing_tier": "admin_free",
        "description": "Read tenant usage summary",
    },
    {
        "method": "GET",
        "route_template": "/usage/billing-tiers",
        "billing_tier": "admin_free",
        "description": "Read billing tier catalog",
    },
    {
        "method": "GET",
        "route_template": "/usage/export",
        "billing_tier": "admin_free",
        "description": "Export tenant usage history",
    },
    {
        "method": "GET",
        "route_template": "/webhooks",
        "billing_tier": "admin_free",
        "description": "List tenant webhooks",
    },
    {
        "method": "POST",
        "route_template": "/webhooks",
        "billing_tier": "admin_free",
        "description": "Create tenant webhook",
    },
    {
        "method": "DELETE",
        "route_template": "/webhooks/{webhook_id}",
        "billing_tier": "admin_free",
        "description": "Delete tenant webhook",
    },
    {
        "method": "GET",
        "route_template": "/webhooks/{webhook_id}/deliveries",
        "billing_tier": "admin_free",
        "description": "Inspect webhook delivery history",
    },
    {
        "method": "POST",
        "route_template": "/jobs",
        "billing_tier": "job_write",
        "description": "Create a job",
    },
    {
        "method": "GET",
        "route_template": "/jobs",
        "billing_tier": "job_read",
        "description": "List jobs",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}",
        "billing_tier": "job_read",
        "description": "Get job metadata",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/events",
        "billing_tier": "job_events",
        "description": "List job lifecycle events",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/start",
        "billing_tier": "training_control",
        "description": "Start training",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/stop",
        "billing_tier": "training_control",
        "description": "Stop training",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/resume",
        "billing_tier": "training_control",
        "description": "Resume training",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/status",
        "billing_tier": "training_snapshot",
        "description": "Read training status snapshot",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/insights",
        "billing_tier": "training_snapshot",
        "description": "Read trend insights",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/metrics",
        "billing_tier": "training_metrics",
        "description": "Read raw metrics rows",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/checkpoints",
        "billing_tier": "artifact_read",
        "description": "List checkpoints",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/checkpoints",
        "billing_tier": "artifact_write",
        "description": "Create checkpoint",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/agent/action",
        "billing_tier": "inference_single",
        "description": "Single inference call",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/agent/action/batch",
        "billing_tier": "inference_batch",
        "description": "Batch inference charged by payload tokens",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/agent/info",
        "billing_tier": "model_catalog",
        "description": "Selected agent metadata",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/genomes",
        "billing_tier": "model_catalog",
        "description": "List genomes",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/genomes/{genome_id}",
        "billing_tier": "model_catalog",
        "description": "Genome metadata lookup",
    },
    {
        "method": "POST",
        "route_template": "/train/start",
        "billing_tier": "training_control",
        "description": "Start training on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/stop",
        "billing_tier": "training_control",
        "description": "Stop training on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/resume",
        "billing_tier": "training_control",
        "description": "Resume training on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/status",
        "billing_tier": "training_snapshot",
        "description": "Read training status snapshot on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/insights",
        "billing_tier": "training_snapshot",
        "description": "Read trend insights on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/metrics",
        "billing_tier": "training_metrics",
        "description": "Read raw metrics rows on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/checkpoints",
        "billing_tier": "artifact_read",
        "description": "List checkpoints on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/checkpoints",
        "billing_tier": "artifact_write",
        "description": "Create checkpoint on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/agent/action",
        "billing_tier": "inference_single",
        "description": "Single inference call on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/agent/action/batch",
        "billing_tier": "inference_batch",
        "description": "Batch inference on the tenant default job, charged by payload tokens",
    },
    {
        "method": "GET",
        "route_template": "/agent/info",
        "billing_tier": "model_catalog",
        "description": "Selected agent metadata on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/genomes",
        "billing_tier": "model_catalog",
        "description": "List genomes on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/genomes/{genome_id}",
        "billing_tier": "model_catalog",
        "description": "Genome metadata lookup on the tenant default job",
    },
]


def _build_billing_catalog() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for route_item in ROUTE_BILLING_CATALOG:
        tier = BILLING_TIERS[route_item["billing_tier"]]
        item = dict(route_item)
        item["unit_name"] = tier["unit_name"]
        item["unit_price_inr"] = tier["unit_price_inr"]
        items.append(item)
    return items


BILLING_CATALOG = _build_billing_catalog()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class APIKeyPrincipal:
    key_id: str
    name: str
    tenant_id: str
    status: str
    role: str = API_KEY_ROLE_ADMIN
    scopes: List[str] = field(default_factory=lambda: [API_KEY_SCOPE_ALL])
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    revoked_at: Optional[str] = None
    expired_at: Optional[str] = None
    rotated_at: Optional[str] = None
    rotated_from_key_id: Optional[str] = None
    replaced_by_key_id: Optional[str] = None


class APIKeyStore:
    def __init__(self, db_path: Optional[str] = None, db_url: Optional[str] = None):
        self.db_target = resolve_db_target(
            context="API auth",
            explicit_path=Path(db_path) if db_path is not None else None,
            explicit_url=db_url,
            env_url_names=("EVOMIND_API_AUTH_DB_URL",),
            default_path=api_auth_db_path(),
        )
        self.db_path = self.db_target.path
        self.db_url = self.db_target.url
        self.db_backend = self.db_target.backend
        self._init_db()

    def _connect(self):
        return connect_database(self.db_target, timeout=30.0)

    @staticmethod
    def _normalize_role(role: Optional[str]) -> str:
        normalized = str(role or API_KEY_ROLE_ADMIN).strip().lower()
        if normalized not in ROLE_DEFAULT_SCOPES:
            raise ValueError(
                "API key role must be one of: "
                + ", ".join(sorted(ROLE_DEFAULT_SCOPES))
            )
        return normalized

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        normalized = str(value or "").strip()
        if not normalized:
            return None
        try:
            return datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError("Timestamps must use ISO-8601 UTC format like 2026-04-15T00:00:00Z") from exc

    @classmethod
    def _normalize_expiration(cls, expires_at: Optional[str]) -> Optional[str]:
        if expires_at is None or str(expires_at).strip() == "":
            return None
        parsed = cls._parse_timestamp(expires_at)
        assert parsed is not None
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")

    @classmethod
    def _default_scopes_for_role(cls, role: str) -> List[str]:
        normalized_role = cls._normalize_role(role)
        return list(ROLE_DEFAULT_SCOPES[normalized_role])

    @classmethod
    def _normalize_scopes(cls, scopes: Optional[Sequence[str]], role: str) -> List[str]:
        normalized_role = cls._normalize_role(role)
        requested = [str(item).strip().lower() for item in (scopes or []) if str(item).strip()]
        if not requested:
            requested = cls._default_scopes_for_role(normalized_role)

        normalized = sorted(set(requested))
        if API_KEY_SCOPE_ALL in normalized:
            if normalized_role != API_KEY_ROLE_ADMIN:
                raise ValueError("Only admin API keys may use the '*' scope")
            return [API_KEY_SCOPE_ALL]

        unknown = sorted(scope for scope in normalized if scope not in ALL_API_KEY_SCOPES)
        if unknown:
            raise ValueError("Unknown API key scopes: " + ", ".join(unknown))

        if normalized_role != API_KEY_ROLE_ADMIN:
            allowed = set(cls._default_scopes_for_role(normalized_role))
            disallowed = sorted(scope for scope in normalized if scope not in allowed)
            if disallowed:
                raise ValueError(
                    f"Scopes exceed the permissions allowed for role '{normalized_role}': "
                    + ", ".join(disallowed)
                )

        return normalized

    @classmethod
    def _scopes_json(cls, scopes: Sequence[str]) -> str:
        return json.dumps(sorted({str(item).strip().lower() for item in scopes if str(item).strip()}))

    @classmethod
    def _load_scopes(cls, raw_value: Any, role: str) -> List[str]:
        if raw_value in (None, ""):
            return cls._default_scopes_for_role(role)
        try:
            items = json.loads(str(raw_value))
        except json.JSONDecodeError:
            items = []
        if not isinstance(items, list):
            items = []
        return cls._normalize_scopes(items, role)

    @staticmethod
    def _current_rate_limit_windows(now: Optional[datetime] = None) -> tuple[str, str]:
        current = now or datetime.now(timezone.utc)
        return (
            current.strftime("%Y-%m-%dT%H:%M"),
            current.strftime("%Y-%m-%d"),
        )

    def _ensure_tenant_limits_conn(self, conn: Any, tenant_id: str) -> None:
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO tenant_limits (
                tenant_id, requests_per_minute, requests_per_day, max_jobs, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id) DO NOTHING
            """,
            (
                tenant_id,
                DEFAULT_REQUESTS_PER_MINUTE,
                DEFAULT_REQUESTS_PER_DAY,
                DEFAULT_MAX_JOBS,
                now,
                now,
            ),
        )

    def _get_tenant_limits_conn(self, conn: Any, tenant_id: str) -> Dict[str, int]:
        self._ensure_tenant_limits_conn(conn, tenant_id)
        row = conn.execute(
            """
            SELECT requests_per_minute, requests_per_day, max_jobs
            FROM tenant_limits
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()

        if row is None:
            return {
                "requests_per_minute": DEFAULT_REQUESTS_PER_MINUTE,
                "requests_per_day": DEFAULT_REQUESTS_PER_DAY,
                "max_jobs": DEFAULT_MAX_JOBS,
            }

        return {
            "requests_per_minute": int(row["requests_per_minute"]),
            "requests_per_day": int(row["requests_per_day"]),
            "max_jobs": int(row["max_jobs"]),
        }

    def _ensure_tenant_billing_conn(self, conn: Any, tenant_id: str) -> None:
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO tenant_billing_accounts (
                tenant_id, currency, balance_inr, created_at, updated_at
            )
            VALUES (?, 'INR', 0, ?, ?)
            ON CONFLICT(tenant_id) DO NOTHING
            """,
            (tenant_id, now, now),
        )

    def is_available(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def _init_db(self) -> None:
        with self._connect() as conn:
            id_column_sql = auto_increment_primary_key_sql(self.db_backend)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_limits (
                    tenant_id TEXT PRIMARY KEY,
                    requests_per_minute INTEGER NOT NULL,
                    requests_per_day INTEGER NOT NULL,
                    max_jobs INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id {id_column_sql},
                    tenant_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    route_template TEXT,
                    status_code INTEGER NOT NULL,
                    duration_ms REAL NOT NULL DEFAULT 0,
                    job_id TEXT,
                    billing_tier TEXT NOT NULL DEFAULT 'unclassified',
                    billed_tokens INTEGER NOT NULL DEFAULT 0,
                    unit_price_inr REAL NOT NULL DEFAULT 0,
                    estimated_cost_inr REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rate_limit_counters (
                    tenant_id TEXT PRIMARY KEY,
                    minute_window TEXT NOT NULL,
                    minute_count INTEGER NOT NULL DEFAULT 0,
                    day_window TEXT NOT NULL,
                    day_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_billing_accounts (
                    tenant_id TEXT PRIMARY KEY,
                    currency TEXT NOT NULL DEFAULT 'INR',
                    balance_inr REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS billing_ledger (
                    id {id_column_sql},
                    tenant_id TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    amount_inr REAL NOT NULL,
                    balance_after_inr REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'INR',
                    description TEXT NOT NULL DEFAULT '',
                    reference_type TEXT,
                    reference_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{{}}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS billing_topups (
                    topup_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    amount_inr REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'INR',
                    receipt TEXT NOT NULL,
                    provider_order_id TEXT NOT NULL,
                    provider_payment_id TEXT,
                    description TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    paid_at TEXT
                )
                """
            )

            usage_log_columns = column_names(conn, "usage_logs")
            usage_log_migrations = {
                "route_template": "TEXT",
                "billing_tier": "TEXT NOT NULL DEFAULT 'unclassified'",
                "billed_tokens": "INTEGER NOT NULL DEFAULT 0",
                "unit_price_inr": "REAL NOT NULL DEFAULT 0",
                "estimated_cost_inr": "REAL NOT NULL DEFAULT 0",
            }
            for column_name, column_definition in usage_log_migrations.items():
                if column_name not in usage_log_columns:
                    try:
                        conn.execute(
                            f"ALTER TABLE usage_logs ADD COLUMN {column_name} {column_definition}"
                        )
                    except Exception as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_logs_tenant_created
                ON usage_logs (tenant_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_billing_ledger_tenant_created
                ON billing_ledger (tenant_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_topups_provider_order
                ON billing_topups (provider, provider_order_id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_topups_provider_payment
                ON billing_topups (provider, provider_payment_id)
                """
            )
            api_key_columns = column_names(conn, "api_keys")
            api_key_migrations = {
                "role": "TEXT NOT NULL DEFAULT 'admin'",
                "scopes_json": """TEXT NOT NULL DEFAULT '["*"]'""",
                "expires_at": "TEXT",
                "updated_at": "TEXT",
                "revoked_at": "TEXT",
                "expired_at": "TEXT",
                "rotated_at": "TEXT",
                "rotated_from_key_id": "TEXT",
                "replaced_by_key_id": "TEXT",
            }
            for column_name, column_definition in api_key_migrations.items():
                if column_name not in api_key_columns:
                    conn.execute(
                        f"ALTER TABLE api_keys ADD COLUMN {column_name} {column_definition}"
                    )

            conn.execute(
                """
                UPDATE api_keys
                SET
                    role = COALESCE(NULLIF(role, ''), 'admin'),
                    scopes_json = COALESCE(NULLIF(scopes_json, ''), '["*"]'),
                    updated_at = COALESCE(NULLIF(updated_at, ''), created_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_status
                ON api_keys (tenant_id, status)
                """
            )
            conn.commit()

    @classmethod
    def _row_to_principal(cls, row: Any) -> APIKeyPrincipal:
        role = cls._normalize_role(row["role"] if row["role"] is not None else API_KEY_ROLE_ADMIN)
        return APIKeyPrincipal(
            key_id=str(row["key_id"]),
            name=str(row["name"]),
            tenant_id=str(row["tenant_id"]),
            status=str(row["status"]),
            role=role,
            scopes=cls._load_scopes(row["scopes_json"], role),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_used_at=row["last_used_at"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
            expired_at=row["expired_at"],
            rotated_at=row["rotated_at"],
            rotated_from_key_id=row["rotated_from_key_id"],
            replaced_by_key_id=row["replaced_by_key_id"],
        )

    @staticmethod
    def _hash_key(raw_key: str, salt_hex: str) -> str:
        salt = bytes.fromhex(salt_hex)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            raw_key.encode("utf-8"),
            salt,
            200_000,
        )
        return digest.hex()

    @staticmethod
    def _generate_key_material() -> tuple[str, str]:
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        raw_key = f"evm_{key_id}_{secret}"
        return key_id, raw_key

    @staticmethod
    def _extract_key_id(raw_key: str) -> Optional[str]:
        if not raw_key.startswith("evm_"):
            return None
        parts = raw_key.split("_", 2)
        if len(parts) != 3 or not parts[1]:
            return None
        return parts[1]

    @staticmethod
    def _round_cost(value: float) -> float:
        return round(float(value), 6)

    @staticmethod
    def _coerce_billed_tokens(billed_tokens: Optional[int]) -> int:
        if billed_tokens is None:
            return 0
        return max(0, int(billed_tokens))

    @staticmethod
    def _is_chargeable_status(status_code: int) -> bool:
        return 200 <= int(status_code) < 400

    @staticmethod
    def _utc_now_datetime() -> datetime:
        return datetime.now(timezone.utc)

    @classmethod
    def _is_expired_timestamp(cls, expires_at: Optional[str], now: Optional[datetime] = None) -> bool:
        parsed = cls._parse_timestamp(expires_at)
        if parsed is None:
            return False
        current = now or cls._utc_now_datetime()
        return parsed <= current

    @staticmethod
    def prepaid_required() -> bool:
        return _is_truthy(os.getenv("EVOMIND_PREPAID_REQUIRED"))

    def ensure_billing_account(self, tenant_id: str) -> None:
        with self._connect() as conn:
            self._ensure_tenant_billing_conn(conn, tenant_id)
            conn.commit()

    @staticmethod
    def _coerce_amount_inr(amount_inr: float) -> float:
        normalized = round(float(amount_inr), 6)
        if normalized <= 0:
            raise ValueError("Amount must be greater than zero")
        return normalized

    @staticmethod
    def _json_dumps(value: Optional[Dict[str, Any]]) -> str:
        return json.dumps(value or {}, sort_keys=True)

    @staticmethod
    def _json_loads(raw_value: Any) -> Dict[str, Any]:
        if raw_value in (None, ""):
            return {}
        try:
            parsed = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    @staticmethod
    def _ledger_entry_to_dict(row: Any) -> Dict[str, Any]:
        return {
            "entry_id": int(row["id"]),
            "tenant_id": str(row["tenant_id"]),
            "entry_type": str(row["entry_type"]),
            "amount_inr": APIKeyStore._round_cost(float(row["amount_inr"] or 0.0)),
            "balance_after_inr": APIKeyStore._round_cost(float(row["balance_after_inr"] or 0.0)),
            "currency": str(row["currency"] or "INR"),
            "description": str(row["description"] or ""),
            "reference_type": row["reference_type"],
            "reference_id": row["reference_id"],
            "created_at": str(row["created_at"]),
        }

    @staticmethod
    def _topup_row_to_dict(row: Any) -> Dict[str, Any]:
        return {
            "topup_id": str(row["topup_id"]),
            "tenant_id": str(row["tenant_id"]),
            "provider": str(row["provider"]),
            "status": str(row["status"]),
            "amount_inr": APIKeyStore._round_cost(float(row["amount_inr"] or 0.0)),
            "currency": str(row["currency"] or "INR"),
            "receipt": str(row["receipt"]),
            "provider_order_id": str(row["provider_order_id"]),
            "provider_payment_id": row["provider_payment_id"],
            "description": row["description"],
            "metadata": APIKeyStore._json_loads(row["metadata_json"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "paid_at": row["paid_at"],
        }

    def _get_billing_account_conn(self, conn: Any, tenant_id: str) -> Dict[str, Any]:
        self._ensure_tenant_billing_conn(conn, tenant_id)
        account_row = conn.execute(
            """
            SELECT tenant_id, currency, balance_inr, created_at, updated_at
            FROM tenant_billing_accounts
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()
        totals_row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN amount_inr > 0 THEN amount_inr ELSE 0 END), 0) AS total_credited_inr,
                COALESCE(SUM(CASE WHEN amount_inr < 0 THEN -amount_inr ELSE 0 END), 0) AS total_debited_inr
            FROM billing_ledger
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()
        if account_row is None:
            raise ValueError(f"Billing account for tenant '{tenant_id}' was not found")
        balance_inr = self._round_cost(float(account_row["balance_inr"] or 0.0))
        return {
            "tenant_id": str(account_row["tenant_id"]),
            "currency": str(account_row["currency"] or "INR"),
            "available_credit_inr": balance_inr,
            "outstanding_amount_inr": self._round_cost(max(0.0, -balance_inr)),
            "total_credited_inr": self._round_cost(float(totals_row["total_credited_inr"] or 0.0)),
            "total_debited_inr": self._round_cost(float(totals_row["total_debited_inr"] or 0.0)),
            "prepaid_required": self.prepaid_required(),
        }

    def _append_billing_ledger_entry_conn(
        self,
        conn: Any,
        tenant_id: str,
        *,
        entry_type: str,
        amount_inr: float,
        description: str,
        reference_type: Optional[str] = None,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_tenant_billing_conn(conn, tenant_id)
        row = conn.execute(
            """
            SELECT balance_inr, currency
            FROM tenant_billing_accounts
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Billing account for tenant '{tenant_id}' was not found")
        current_balance = float(row["balance_inr"] or 0.0)
        next_balance = self._round_cost(current_balance + float(amount_inr))
        now = _utc_now()
        conn.execute(
            """
            UPDATE tenant_billing_accounts
            SET balance_inr = ?, updated_at = ?
            WHERE tenant_id = ?
            """,
            (next_balance, now, tenant_id),
        )
        conn.execute(
            """
            INSERT INTO billing_ledger (
                tenant_id,
                entry_type,
                amount_inr,
                balance_after_inr,
                currency,
                description,
                reference_type,
                reference_id,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                entry_type,
                self._round_cost(float(amount_inr)),
                next_balance,
                str(row["currency"] or "INR"),
                str(description or ""),
                reference_type,
                reference_id,
                self._json_dumps(metadata),
                now,
            ),
        )
        created_row = conn.execute(
            """
            SELECT
                id,
                tenant_id,
                entry_type,
                amount_inr,
                balance_after_inr,
                currency,
                description,
                reference_type,
                reference_id,
                created_at
            FROM billing_ledger
            WHERE tenant_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (tenant_id,),
        ).fetchone()
        if created_row is None:
            raise ValueError("Billing ledger entry was not created")
        return self._ledger_entry_to_dict(created_row)

    def add_credit(
        self,
        tenant_id: str,
        amount_inr: float,
        *,
        description: str,
        reference_type: Optional[str] = None,
        reference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_amount = self._coerce_amount_inr(amount_inr)
        with self._connect() as conn:
            entry = self._append_billing_ledger_entry_conn(
                conn,
                tenant_id,
                entry_type="credit",
                amount_inr=normalized_amount,
                description=description,
                reference_type=reference_type,
                reference_id=reference_id,
                metadata=metadata,
            )
            conn.commit()
        return entry

    def get_billing_account(self, tenant_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            account = self._get_billing_account_conn(conn, tenant_id)
        return account

    def list_billing_ledger(
        self,
        tenant_id: str,
        limit: int = DEFAULT_BILLING_LEDGER_LIMIT,
    ) -> List[Dict[str, Any]]:
        normalized_limit = max(1, int(limit))
        with self._connect() as conn:
            self._ensure_tenant_billing_conn(conn, tenant_id)
            rows = conn.execute(
                """
                SELECT
                    id,
                    tenant_id,
                    entry_type,
                    amount_inr,
                    balance_after_inr,
                    currency,
                    description,
                    reference_type,
                    reference_id,
                    created_at
                FROM billing_ledger
                WHERE tenant_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, normalized_limit),
            ).fetchall()
        return [self._ledger_entry_to_dict(row) for row in rows]

    def create_topup(
        self,
        tenant_id: str,
        *,
        provider: str,
        amount_inr: float,
        provider_order_id: str,
        receipt: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_amount = self._coerce_amount_inr(amount_inr)
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider:
            raise ValueError("Payment provider is required")
        now = _utc_now()
        topup_id = f"topup_{secrets.token_hex(8)}"
        with self._connect() as conn:
            self._ensure_tenant_billing_conn(conn, tenant_id)
            conn.execute(
                """
                INSERT INTO billing_topups (
                    topup_id,
                    tenant_id,
                    provider,
                    status,
                    amount_inr,
                    currency,
                    receipt,
                    provider_order_id,
                    provider_payment_id,
                    description,
                    metadata_json,
                    created_at,
                    updated_at,
                    paid_at
                )
                VALUES (?, ?, ?, 'created', ?, 'INR', ?, ?, NULL, ?, ?, ?, ?, NULL)
                """,
                (
                    topup_id,
                    tenant_id,
                    normalized_provider,
                    normalized_amount,
                    receipt,
                    provider_order_id,
                    description,
                    self._json_dumps(metadata),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM billing_topups
                WHERE topup_id = ?
                """,
                (topup_id,),
            ).fetchone()
            conn.commit()
        if row is None:
            raise ValueError("Billing top-up was not created")
        return self._topup_row_to_dict(row)

    def confirm_topup_payment(
        self,
        *,
        provider: str,
        provider_order_id: str,
        provider_payment_id: str,
        amount_inr: Optional[float] = None,
        expected_tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_provider = str(provider or "").strip().lower()
        normalized_order_id = str(provider_order_id or "").strip()
        normalized_payment_id = str(provider_payment_id or "").strip()
        if not normalized_provider or not normalized_order_id or not normalized_payment_id:
            raise ValueError("Provider, order id, and payment id are required")
        normalized_amount = None if amount_inr is None else self._coerce_amount_inr(amount_inr)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM billing_topups
                WHERE provider = ? AND provider_order_id = ?
                """,
                (normalized_provider, normalized_order_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"Billing top-up for order '{normalized_order_id}' was not found")
            if expected_tenant_id is not None and str(row["tenant_id"]) != str(expected_tenant_id):
                raise PermissionError("Top-up does not belong to the authenticated tenant")

            existing_payment_row = conn.execute(
                """
                SELECT topup_id
                FROM billing_topups
                WHERE provider = ? AND provider_payment_id = ? AND topup_id != ?
                """,
                (normalized_provider, normalized_payment_id, row["topup_id"]),
            ).fetchone()
            if existing_payment_row is not None:
                raise ValueError(f"Payment '{normalized_payment_id}' is already linked to another top-up")

            stored_amount = self._round_cost(float(row["amount_inr"] or 0.0))
            if normalized_amount is not None and normalized_amount != stored_amount:
                raise ValueError(
                    f"Payment amount mismatch for order '{normalized_order_id}': expected {stored_amount}, got {normalized_amount}"
                )

            if str(row["status"]) != "paid":
                ledger_metadata = dict(self._json_loads(row["metadata_json"]))
                ledger_metadata.update(metadata or {})
                ledger_metadata["provider_order_id"] = normalized_order_id
                ledger_metadata["provider_payment_id"] = normalized_payment_id
                self._append_billing_ledger_entry_conn(
                    conn,
                    str(row["tenant_id"]),
                    entry_type="topup_credit",
                    amount_inr=stored_amount,
                    description=f"{normalized_provider.capitalize()} prepaid top-up",
                    reference_type="payment",
                    reference_id=normalized_payment_id,
                    metadata=ledger_metadata,
                )
                now = _utc_now()
                conn.execute(
                    """
                    UPDATE billing_topups
                    SET
                        status = 'paid',
                        provider_payment_id = ?,
                        metadata_json = ?,
                        updated_at = ?,
                        paid_at = COALESCE(paid_at, ?)
                    WHERE topup_id = ?
                    """,
                    (
                        normalized_payment_id,
                        self._json_dumps(ledger_metadata),
                        now,
                        now,
                        row["topup_id"],
                    ),
                )
                row = conn.execute(
                    """
                    SELECT *
                    FROM billing_topups
                    WHERE topup_id = ?
                    """,
                    (row["topup_id"],),
                ).fetchone()
            if row is None:
                raise ValueError("Billing top-up payment confirmation failed")
            account = self._get_billing_account_conn(conn, str(row["tenant_id"]))
            conn.commit()
        if row is None:
            raise ValueError("Billing top-up payment confirmation failed")
        topup = self._topup_row_to_dict(row)
        return {
            "topup": topup,
            "account": account,
            "credited": topup["status"] == "paid" and topup["provider_payment_id"] == normalized_payment_id,
        }

    def require_prepaid_balance(
        self,
        tenant_id: str,
        *,
        method: str,
        route_template: Optional[str],
        path: str,
    ) -> None:
        if not self.prepaid_required():
            return
        billing_definition = self.resolve_billing_definition(
            method=method,
            route_template=route_template,
            path=path,
        )
        if float(billing_definition["unit_price_inr"]) <= 0.0:
            return
        account = self.get_billing_account(tenant_id)
        if float(account["available_credit_inr"]) > 0.0:
            return
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Prepaid balance exhausted. Top up credits before calling chargeable endpoints.",
        )

    def _fetch_key_row(self, conn: Any, key_id: str) -> Any:
        return conn.execute(
            """
            SELECT
                key_id,
                name,
                tenant_id,
                salt,
                key_hash,
                status,
                role,
                scopes_json,
                created_at,
                updated_at,
                last_used_at,
                expires_at,
                revoked_at,
                expired_at,
                rotated_at,
                rotated_from_key_id,
                replaced_by_key_id
            FROM api_keys
            WHERE key_id = ?
            """,
            (key_id,),
        ).fetchone()

    def _mark_key_expired(self, conn: Any, key_id: str, when: str) -> None:
        conn.execute(
            """
            UPDATE api_keys
            SET
                status = ?,
                expired_at = COALESCE(expired_at, ?),
                updated_at = ?,
                last_used_at = COALESCE(last_used_at, last_used_at)
            WHERE key_id = ? AND status = ?
            """,
            (
                API_KEY_STATUS_EXPIRED,
                when,
                when,
                key_id,
                API_KEY_STATUS_ACTIVE,
            ),
        )

    def get_key(self, key_id: str) -> Optional[APIKeyPrincipal]:
        with self._connect() as conn:
            row = self._fetch_key_row(conn, key_id)
            conn.commit()
        if row is None:
            return None
        return self._row_to_principal(row)

    def resolve_required_scopes(
        self,
        method: str,
        route_template: Optional[str],
    ) -> List[str]:
        normalized_method = str(method or "").strip().upper()
        normalized_route = str(route_template or "").strip()
        return list(API_KEY_PERMISSION_RULES.get((normalized_method, normalized_route), ()))

    def require_permission(
        self,
        principal: APIKeyPrincipal,
        method: str,
        route_template: Optional[str],
    ) -> None:
        required_scopes = self.resolve_required_scopes(method, route_template)
        if API_KEY_SCOPE_ALL in principal.scopes:
            return
        if required_scopes:
            if any(scope in principal.scopes for scope in required_scopes):
                return
            detail = (
                "API key is not permitted to access this endpoint. "
                f"Required scope: {', '.join(required_scopes)}"
            )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
        if principal.role == API_KEY_ROLE_ADMIN:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not permitted to access this endpoint",
        )

    def get_billing_catalog(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in BILLING_CATALOG]

    def resolve_billing_definition(
        self,
        method: str,
        route_template: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_method = method.upper()
        candidates = [route_template, path]

        for route_value in candidates:
            if not route_value:
                continue
            for item in BILLING_CATALOG:
                if item["method"] == normalized_method and item["route_template"] == route_value:
                    return dict(item)

        return {
            "method": normalized_method,
            "route_template": route_template or path or "",
            "billing_tier": "unclassified",
            "unit_name": "token",
            "unit_price_inr": 0.0,
            "description": "No billing rule registered for this endpoint",
        }

    def ensure_tenant_limits(self, tenant_id: str) -> None:
        with self._connect() as conn:
            self._ensure_tenant_limits_conn(conn, tenant_id)
            conn.commit()

    def get_tenant_limits(self, tenant_id: str) -> Dict[str, int]:
        with self._connect() as conn:
            limits = self._get_tenant_limits_conn(conn, tenant_id)
            conn.commit()
        return limits

    def get_rate_limit_snapshot(self, tenant_id: str) -> Dict[str, int]:
        limits = self.get_tenant_limits(tenant_id)
        minute_window, day_window = self._current_rate_limit_windows()

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT minute_window, minute_count, day_window, day_count
                FROM rate_limit_counters
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

        minute_count = 0
        day_count = 0
        if row is not None:
            if str(row["minute_window"]) == minute_window:
                minute_count = int(row["minute_count"] or 0)
            if str(row["day_window"]) == day_window:
                day_count = int(row["day_count"] or 0)

        return {
            "minute_count": minute_count,
            "day_count": day_count,
            **limits,
        }

    def consume_rate_limit(self, tenant_id: str, requests: int = 1) -> Dict[str, int]:
        normalized_requests = max(1, int(requests))
        now = datetime.now(timezone.utc)
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        minute_window, day_window = self._current_rate_limit_windows(now)

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            limits = self._get_tenant_limits_conn(conn, tenant_id)
            row = conn.execute(
                """
                SELECT minute_window, minute_count, day_window, day_count
                FROM rate_limit_counters
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

            minute_count = 0
            day_count = 0
            if row is not None:
                if str(row["minute_window"]) == minute_window:
                    minute_count = int(row["minute_count"] or 0)
                if str(row["day_window"]) == day_window:
                    day_count = int(row["day_count"] or 0)

            next_minute_count = minute_count + normalized_requests
            next_day_count = day_count + normalized_requests

            if next_minute_count > limits["requests_per_minute"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Per-minute request limit exceeded",
                )

            if next_day_count > limits["requests_per_day"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Per-day request limit exceeded",
                )

            conn.execute(
                """
                INSERT INTO rate_limit_counters (
                    tenant_id,
                    minute_window,
                    minute_count,
                    day_window,
                    day_count,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    minute_window = excluded.minute_window,
                    minute_count = excluded.minute_count,
                    day_window = excluded.day_window,
                    day_count = excluded.day_count,
                    updated_at = excluded.updated_at
                """,
                (
                    tenant_id,
                    minute_window,
                    next_minute_count,
                    day_window,
                    next_day_count,
                    now_iso,
                ),
            )
            conn.commit()

        return {
            "minute_count": next_minute_count,
            "day_count": next_day_count,
            **limits,
        }

    def set_tenant_limits(
        self,
        tenant_id: str,
        requests_per_minute: Optional[int] = None,
        requests_per_day: Optional[int] = None,
        max_jobs: Optional[int] = None,
    ) -> Dict[str, int]:
        current = self.get_tenant_limits(tenant_id)
        updated = {
            "requests_per_minute": int(
                requests_per_minute
                if requests_per_minute is not None
                else current["requests_per_minute"]
            ),
            "requests_per_day": int(
                requests_per_day
                if requests_per_day is not None
                else current["requests_per_day"]
            ),
            "max_jobs": int(max_jobs if max_jobs is not None else current["max_jobs"]),
        }

        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tenant_limits (
                    tenant_id, requests_per_minute, requests_per_day, max_jobs, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET
                    requests_per_minute = excluded.requests_per_minute,
                    requests_per_day = excluded.requests_per_day,
                    max_jobs = excluded.max_jobs,
                    updated_at = excluded.updated_at
                """,
                (
                    tenant_id,
                    updated["requests_per_minute"],
                    updated["requests_per_day"],
                    updated["max_jobs"],
                    now,
                    now,
                ),
            )
            conn.commit()

        return updated

    def create_key(
        self,
        name: str,
        tenant_id: str = "default",
        role: str = API_KEY_ROLE_ADMIN,
        scopes: Optional[Sequence[str]] = None,
        expires_at: Optional[str] = None,
        rotated_from_key_id: Optional[str] = None,
    ) -> tuple[APIKeyPrincipal, str]:
        normalized_role = self._normalize_role(role)
        normalized_scopes = self._normalize_scopes(scopes, normalized_role)
        normalized_expires_at = self._normalize_expiration(expires_at)
        if self._is_expired_timestamp(normalized_expires_at):
            raise ValueError("API key expiration must be in the future")
        key_id, raw_key = self._generate_key_material()
        salt_hex = secrets.token_hex(16)
        key_hash = self._hash_key(raw_key, salt_hex)
        created_at = _utc_now()
        self.ensure_tenant_limits(tenant_id)
        self.ensure_billing_account(tenant_id)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id,
                    name,
                    tenant_id,
                    salt,
                    key_hash,
                    status,
                    role,
                    scopes_json,
                    created_at,
                    updated_at,
                    expires_at,
                    rotated_from_key_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    name,
                    tenant_id,
                    salt_hex,
                    key_hash,
                    API_KEY_STATUS_ACTIVE,
                    normalized_role,
                    self._scopes_json(normalized_scopes),
                    created_at,
                    created_at,
                    normalized_expires_at,
                    rotated_from_key_id,
                ),
            )
            conn.commit()

        principal = APIKeyPrincipal(
            key_id=key_id,
            name=name,
            tenant_id=tenant_id,
            status=API_KEY_STATUS_ACTIVE,
            role=normalized_role,
            scopes=normalized_scopes,
            created_at=created_at,
            updated_at=created_at,
            expires_at=normalized_expires_at,
            rotated_from_key_id=rotated_from_key_id,
        )
        return principal, raw_key

    def list_keys(self, tenant_id: Optional[str] = None) -> List[APIKeyPrincipal]:
        with self._connect() as conn:
            if tenant_id is None:
                rows = conn.execute(
                    """
                    SELECT
                        key_id,
                        name,
                        tenant_id,
                        salt,
                        key_hash,
                        status,
                        role,
                        scopes_json,
                        created_at,
                        updated_at,
                        last_used_at,
                        expires_at,
                        revoked_at,
                        expired_at,
                        rotated_at,
                        rotated_from_key_id,
                        replaced_by_key_id
                    FROM api_keys
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        key_id,
                        name,
                        tenant_id,
                        salt,
                        key_hash,
                        status,
                        role,
                        scopes_json,
                        created_at,
                        updated_at,
                        last_used_at,
                        expires_at,
                        revoked_at,
                        expired_at,
                        rotated_at,
                        rotated_from_key_id,
                        replaced_by_key_id
                    FROM api_keys
                    WHERE tenant_id = ?
                    ORDER BY created_at DESC
                    """,
                    (tenant_id,),
                ).fetchall()
            conn.commit()

        return [self._row_to_principal(row) for row in rows]

    def update_key(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        role: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[APIKeyPrincipal]:
        with self._connect() as conn:
            row = self._fetch_key_row(conn, key_id)
            if row is None:
                conn.commit()
                return None

            current_role = self._normalize_role(row["role"])
            next_role = self._normalize_role(role if role is not None else current_role)
            next_scopes = self._normalize_scopes(
                scopes if scopes is not None else self._load_scopes(row["scopes_json"], current_role),
                next_role,
            )
            next_expires_at = self._normalize_expiration(
                expires_at if expires_at is not None else row["expires_at"]
            )
            if self._is_expired_timestamp(next_expires_at):
                raise ValueError("API key expiration must be in the future")

            updated_at = _utc_now()
            conn.execute(
                """
                UPDATE api_keys
                SET
                    name = ?,
                    role = ?,
                    scopes_json = ?,
                    expires_at = ?,
                    updated_at = ?
                WHERE key_id = ?
                """,
                (
                    name if name is not None else row["name"],
                    next_role,
                    self._scopes_json(next_scopes),
                    next_expires_at,
                    updated_at,
                    key_id,
                ),
            )
            updated_row = self._fetch_key_row(conn, key_id)
            conn.commit()

        if updated_row is None:
            return None
        return self._row_to_principal(updated_row)

    def revoke_key(self, key_id: str) -> bool:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE api_keys
                SET
                    status = ?,
                    revoked_at = COALESCE(revoked_at, ?),
                    updated_at = ?
                WHERE key_id = ? AND status = ?
                """,
                (
                    API_KEY_STATUS_REVOKED,
                    now,
                    now,
                    key_id,
                    API_KEY_STATUS_ACTIVE,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def rotate_key(
        self,
        key_id: str,
        *,
        name: Optional[str] = None,
        role: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[tuple[APIKeyPrincipal, str]]:
        with self._connect() as conn:
            existing_row = self._fetch_key_row(conn, key_id)
            if existing_row is None:
                conn.commit()
                return None
            if str(existing_row["status"]) != API_KEY_STATUS_ACTIVE:
                raise ValueError("Only active API keys can be rotated")
            if self._is_expired_timestamp(existing_row["expires_at"]):
                raise ValueError("Expired API keys cannot be rotated")

            next_role = self._normalize_role(role if role is not None else existing_row["role"])
            inherited_scopes = self._load_scopes(existing_row["scopes_json"], next_role)
            next_scopes = self._normalize_scopes(scopes if scopes is not None else inherited_scopes, next_role)
            next_expires_at = self._normalize_expiration(
                expires_at if expires_at is not None else existing_row["expires_at"]
            )
            if self._is_expired_timestamp(next_expires_at):
                raise ValueError("API key expiration must be in the future")

            new_key_id, raw_key = self._generate_key_material()
            salt_hex = secrets.token_hex(16)
            key_hash = self._hash_key(raw_key, salt_hex)
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id,
                    name,
                    tenant_id,
                    salt,
                    key_hash,
                    status,
                    role,
                    scopes_json,
                    created_at,
                    updated_at,
                    expires_at,
                    rotated_from_key_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_key_id,
                    name if name is not None else existing_row["name"],
                    existing_row["tenant_id"],
                    salt_hex,
                    key_hash,
                    API_KEY_STATUS_ACTIVE,
                    next_role,
                    self._scopes_json(next_scopes),
                    now,
                    now,
                    next_expires_at,
                    key_id,
                ),
            )
            conn.execute(
                """
                UPDATE api_keys
                SET
                    status = ?,
                    revoked_at = COALESCE(revoked_at, ?),
                    rotated_at = COALESCE(rotated_at, ?),
                    replaced_by_key_id = ?,
                    updated_at = ?
                WHERE key_id = ? AND status = ?
                """,
                (
                    API_KEY_STATUS_ROTATED,
                    now,
                    now,
                    new_key_id,
                    now,
                    key_id,
                    API_KEY_STATUS_ACTIVE,
                ),
            )
            rotated_row = self._fetch_key_row(conn, new_key_id)
            conn.commit()

        if rotated_row is None:
            return None
        return self._row_to_principal(rotated_row), raw_key

    def resolve_key(self, raw_key: str) -> Optional[APIKeyPrincipal]:
        key_id = self._extract_key_id(raw_key)
        if key_id is None:
            return None

        with self._connect() as conn:
            row = self._fetch_key_row(conn, key_id)
            if row is None:
                conn.commit()
                return None
            if str(row["status"]) != API_KEY_STATUS_ACTIVE:
                conn.commit()
                return None
            if self._is_expired_timestamp(row["expires_at"]):
                expired_at = _utc_now()
                self._mark_key_expired(conn, key_id, expired_at)
                conn.commit()
                return None

            candidate_hash = self._hash_key(raw_key, row["salt"])
            if not hmac.compare_digest(candidate_hash, row["key_hash"]):
                conn.commit()
                return None

            now = _utc_now()
            conn.execute(
                "UPDATE api_keys SET last_used_at = ?, updated_at = ? WHERE key_id = ?",
                (now, now, key_id),
            )
            row = self._fetch_key_row(conn, key_id)
            conn.commit()

        assert row is not None
        principal = self._row_to_principal(row)
        self.ensure_tenant_limits(principal.tenant_id)
        return principal

    def check_rate_limits(self, tenant_id: str) -> Dict[str, int]:
        snapshot = self.get_rate_limit_snapshot(tenant_id)

        if snapshot["minute_count"] >= snapshot["requests_per_minute"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Per-minute request limit exceeded",
            )

        if snapshot["day_count"] >= snapshot["requests_per_day"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Per-day request limit exceeded",
            )

        return snapshot

    def log_usage(
        self,
        tenant_id: str,
        key_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        route_template: Optional[str] = None,
        job_id: Optional[str] = None,
        billed_tokens: Optional[int] = None,
    ) -> None:
        billing_definition = self.resolve_billing_definition(
            method=method,
            route_template=route_template,
            path=path,
        )
        normalized_tokens = self._coerce_billed_tokens(billed_tokens)
        unit_price_inr = float(billing_definition["unit_price_inr"])
        estimated_cost_inr = 0.0
        if self._is_chargeable_status(status_code):
            estimated_cost_inr = self._round_cost(normalized_tokens * unit_price_inr)

        with self._connect() as conn:
            self._ensure_tenant_billing_conn(conn, tenant_id)
            conn.execute(
                """
                INSERT INTO usage_logs (
                    tenant_id,
                    key_id,
                    method,
                    path,
                    route_template,
                    status_code,
                    duration_ms,
                    job_id,
                    billing_tier,
                    billed_tokens,
                    unit_price_inr,
                    estimated_cost_inr,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    key_id,
                    method.upper(),
                    path,
                    billing_definition["route_template"] or route_template or path,
                    int(status_code),
                    float(duration_ms),
                    job_id,
                    billing_definition["billing_tier"],
                    normalized_tokens,
                    unit_price_inr,
                    estimated_cost_inr,
                    _utc_now(),
                ),
            )
            if estimated_cost_inr > 0.0:
                reference_target = route_template or path
                self._append_billing_ledger_entry_conn(
                    conn,
                    tenant_id,
                    entry_type="usage_charge",
                    amount_inr=-estimated_cost_inr,
                    description=f"Usage charge for {method.upper()} {reference_target}",
                    reference_type="usage_path",
                    reference_id=reference_target,
                    metadata={
                        "key_id": key_id,
                        "job_id": job_id,
                        "billing_tier": billing_definition["billing_tier"],
                        "billed_tokens": normalized_tokens,
                        "status_code": int(status_code),
                    },
                )
            conn.commit()

    def get_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        rate_snapshot = self.get_rate_limit_snapshot(tenant_id)
        now = datetime.now(timezone.utc)
        day_threshold = now.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        with self._connect() as conn:
            day_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(estimated_cost_inr), 0) AS estimated_cost_inr
                FROM usage_logs
                WHERE tenant_id = ? AND created_at >= ?
                """,
                (tenant_id, day_threshold),
            ).fetchone()
            total_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(estimated_cost_inr), 0) AS estimated_cost_inr
                FROM usage_logs
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

        minute_count = int(rate_snapshot["minute_count"])
        day_count = int(rate_snapshot["day_count"])
        total_count = int(total_row["request_count"]) if total_row is not None else 0
        day_cost = self._round_cost(float(day_row["estimated_cost_inr"] or 0.0)) if day_row is not None else 0.0
        total_cost = self._round_cost(float(total_row["estimated_cost_inr"] or 0.0)) if total_row is not None else 0.0

        return {
            "requests_last_minute": minute_count,
            "requests_last_day": day_count,
            "requests_total": total_count,
            "requests_per_minute_limit": rate_snapshot["requests_per_minute"],
            "requests_per_day_limit": rate_snapshot["requests_per_day"],
            "max_jobs": rate_snapshot["max_jobs"],
            "remaining_this_minute": max(0, rate_snapshot["requests_per_minute"] - minute_count),
            "remaining_today": max(0, rate_snapshot["requests_per_day"] - day_count),
            "estimated_cost_last_day_inr": day_cost,
            "estimated_cost_total_inr": total_cost,
        }

    def export_usage(
        self,
        tenant_id: str,
        days: int = 7,
        limit: int = DEFAULT_EXPORT_LIMIT,
    ) -> List[Dict[str, Any]]:
        since_threshold = (
            datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        normalized_limit = max(1, int(limit))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    created_at,
                    tenant_id,
                    key_id,
                    method,
                    path,
                    route_template,
                    status_code,
                    duration_ms,
                    job_id,
                    billing_tier,
                    billed_tokens,
                    unit_price_inr,
                    estimated_cost_inr
                FROM usage_logs
                WHERE tenant_id = ? AND created_at >= ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, since_threshold, normalized_limit),
            ).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "created_at": str(row["created_at"]),
                    "tenant_id": str(row["tenant_id"]),
                    "key_id": str(row["key_id"]),
                    "method": str(row["method"]),
                    "path": str(row["path"]),
                    "route_template": row["route_template"],
                    "status_code": int(row["status_code"]),
                    "duration_ms": float(row["duration_ms"]),
                    "job_id": row["job_id"],
                    "billing_tier": str(row["billing_tier"]),
                    "billed_tokens": int(row["billed_tokens"] or 0),
                    "unit_price_inr": self._round_cost(float(row["unit_price_inr"] or 0.0)),
                    "estimated_cost_inr": self._round_cost(
                        float(row["estimated_cost_inr"] or 0.0)
                    ),
                }
            )
        return items


api_key_store = APIKeyStore()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> APIKeyPrincipal:
    raw_key = header_key
    if raw_key is None and bearer is not None:
        raw_key = bearer.credentials

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    principal = await asyncio.to_thread(api_key_store.resolve_key, raw_key)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, rotated, or revoked API key",
        )

    route = request.scope.get("route")
    route_template = getattr(route, "path", request.url.path)
    await asyncio.to_thread(
        api_key_store.require_permission,
        principal,
        request.method,
        route_template,
    )
    await asyncio.to_thread(
        api_key_store.require_prepaid_balance,
        principal.tenant_id,
        method=request.method,
        route_template=route_template,
        path=request.url.path,
    )
    request.state.principal = principal
    merge_log_context(
        tenant_id=principal.tenant_id,
        key_id=principal.key_id,
        api_key_role=principal.role,
    )
    request.state.rate_limits = await asyncio.to_thread(
        api_key_store.consume_rate_limit,
        principal.tenant_id,
    )
    return principal


def _compute_cli_expiration(
    *,
    expires_at: Optional[str],
    expires_in_days: Optional[int],
    no_expiry: bool = False,
) -> Optional[str]:
    if no_expiry and (expires_at or expires_in_days is not None):
        raise ValueError("Do not combine --no-expiry with --expires-at or --expires-in-days")
    if no_expiry:
        return ""
    if expires_at and expires_in_days is not None:
        raise ValueError("Specify either --expires-at or --expires-in-days, not both")
    if expires_at:
        return APIKeyStore._normalize_expiration(expires_at)
    if expires_in_days is None:
        return None
    normalized_days = int(expires_in_days)
    if normalized_days <= 0:
        raise ValueError("--expires-in-days must be greater than zero")
    return (datetime.now(timezone.utc) + timedelta(days=normalized_days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_cli_scopes(raw_scopes: Optional[str]) -> Optional[List[str]]:
    if raw_scopes is None:
        return None
    return [item.strip().lower() for item in raw_scopes.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage EvoMind API keys, limits, billing tiers, and usage exports"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--name", required=True, help="Friendly key name")
    create_parser.add_argument("--tenant", default="default", help="Tenant identifier")
    create_parser.add_argument(
        "--role",
        choices=sorted(ROLE_DEFAULT_SCOPES),
        default=API_KEY_ROLE_ADMIN,
        help="Role baseline for the API key",
    )
    create_parser.add_argument(
        "--scopes",
        help="Comma-separated scope override; must stay within the selected role",
    )
    create_parser.add_argument(
        "--expires-at",
        help="Absolute expiry timestamp in UTC, e.g. 2026-04-15T00:00:00Z",
    )
    create_parser.add_argument(
        "--expires-in-days",
        type=int,
        help="Relative expiry window in days",
    )

    subparsers.add_parser("list", help="List API keys")
    subparsers.add_parser("billing-tiers", help="List endpoint billing tiers")

    update_parser = subparsers.add_parser("update", help="Update API key metadata")
    update_parser.add_argument("--key-id", required=True, help="API key id to update")
    update_parser.add_argument("--name", help="Updated friendly name")
    update_parser.add_argument(
        "--role",
        choices=sorted(ROLE_DEFAULT_SCOPES),
        help="Updated role baseline",
    )
    update_parser.add_argument(
        "--scopes",
        help="Comma-separated scope override; must stay within the selected role",
    )
    update_parser.add_argument(
        "--expires-at",
        help="Absolute expiry timestamp in UTC, e.g. 2026-04-15T00:00:00Z",
    )
    update_parser.add_argument(
        "--expires-in-days",
        type=int,
        help="Relative expiry window in days",
    )
    update_parser.add_argument(
        "--no-expiry",
        action="store_true",
        help="Clear any existing expiration on the API key",
    )

    rotate_parser = subparsers.add_parser("rotate", help="Rotate an API key and issue a replacement")
    rotate_parser.add_argument("--key-id", required=True, help="API key id to rotate")
    rotate_parser.add_argument("--name", help="Optional name override for the replacement key")
    rotate_parser.add_argument(
        "--role",
        choices=sorted(ROLE_DEFAULT_SCOPES),
        help="Optional role override for the replacement key",
    )
    rotate_parser.add_argument(
        "--scopes",
        help="Comma-separated scope override for the replacement key",
    )
    rotate_parser.add_argument(
        "--expires-at",
        help="Absolute expiry timestamp in UTC, e.g. 2026-04-15T00:00:00Z",
    )
    rotate_parser.add_argument(
        "--expires-in-days",
        type=int,
        help="Relative expiry window in days for the replacement key",
    )
    rotate_parser.add_argument(
        "--no-expiry",
        action="store_true",
        help="Create the replacement key without an expiration",
    )

    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("--key-id", required=True, help="API key id to revoke")

    limits_parser = subparsers.add_parser("set-limits", help="Set tenant request/job limits")
    limits_parser.add_argument("--tenant", required=True, help="Tenant identifier")
    limits_parser.add_argument("--rpm", type=int, help="Requests per minute")
    limits_parser.add_argument("--rpd", type=int, help="Requests per day")
    limits_parser.add_argument("--max-jobs", type=int, help="Maximum jobs per tenant")

    usage_parser = subparsers.add_parser("usage", help="Show tenant usage summary")
    usage_parser.add_argument("--tenant", required=True, help="Tenant identifier")

    export_parser = subparsers.add_parser("export", help="Export tenant usage rows")
    export_parser.add_argument("--tenant", required=True, help="Tenant identifier")
    export_parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    export_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_EXPORT_LIMIT,
        help="Maximum rows to export",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export output format",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "create":
        expires_at = _compute_cli_expiration(
            expires_at=args.expires_at,
            expires_in_days=args.expires_in_days,
            no_expiry=False,
        )
        principal, raw_key = api_key_store.create_key(
            name=args.name,
            tenant_id=args.tenant,
            role=args.role,
            scopes=_parse_cli_scopes(args.scopes),
            expires_at=expires_at,
        )
        print(f"key_id: {principal.key_id}")
        print(f"name: {principal.name}")
        print(f"tenant_id: {principal.tenant_id}")
        print(f"role: {principal.role}")
        print(f"scopes: {','.join(principal.scopes)}")
        print(f"expires_at: {principal.expires_at or ''}")
        print(f"api_key: {raw_key}")
        return 0

    if args.command == "list":
        for item in api_key_store.list_keys():
            print(
                "key_id={key_id} name={name} tenant_id={tenant_id} status={status} "
                "role={role} scopes={scopes} expires_at={expires_at} last_used_at={last_used_at} "
                "rotated_from={rotated_from} replaced_by={replaced_by}".format(
                    key_id=item.key_id,
                    name=item.name,
                    tenant_id=item.tenant_id,
                    status=item.status,
                    role=item.role,
                    scopes=",".join(item.scopes),
                    expires_at=item.expires_at or "",
                    last_used_at=item.last_used_at or "",
                    rotated_from=item.rotated_from_key_id or "",
                    replaced_by=item.replaced_by_key_id or "",
                )
            )
        return 0

    if args.command == "billing-tiers":
        for item in api_key_store.get_billing_catalog():
            print(
                "{method} {route_template} tier={billing_tier} unit={unit_name} "
                "price_inr={unit_price_inr:.6f} description={description}".format(**item)
            )
        return 0

    if args.command == "update":
        expires_at = _compute_cli_expiration(
            expires_at=args.expires_at,
            expires_in_days=args.expires_in_days,
            no_expiry=args.no_expiry,
        )
        updated = api_key_store.update_key(
            args.key_id,
            name=args.name,
            role=args.role,
            scopes=_parse_cli_scopes(args.scopes),
            expires_at=expires_at
            if (args.no_expiry or args.expires_at or args.expires_in_days is not None)
            else None,
        )
        if updated is None:
            print("not_found")
            return 1
        print(f"key_id: {updated.key_id}")
        print(f"name: {updated.name}")
        print(f"tenant_id: {updated.tenant_id}")
        print(f"role: {updated.role}")
        print(f"scopes: {','.join(updated.scopes)}")
        print(f"expires_at: {updated.expires_at or ''}")
        return 0

    if args.command == "rotate":
        expires_at = _compute_cli_expiration(
            expires_at=args.expires_at,
            expires_in_days=args.expires_in_days,
            no_expiry=args.no_expiry,
        )
        rotated = api_key_store.rotate_key(
            args.key_id,
            name=args.name,
            role=args.role,
            scopes=_parse_cli_scopes(args.scopes),
            expires_at=expires_at
            if (args.no_expiry or args.expires_at or args.expires_in_days is not None)
            else None,
        )
        if rotated is None:
            print("not_found")
            return 1
        principal, raw_key = rotated
        print(f"key_id: {principal.key_id}")
        print(f"name: {principal.name}")
        print(f"tenant_id: {principal.tenant_id}")
        print(f"role: {principal.role}")
        print(f"scopes: {','.join(principal.scopes)}")
        print(f"expires_at: {principal.expires_at or ''}")
        print(f"api_key: {raw_key}")
        return 0

    if args.command == "revoke":
        revoked = api_key_store.revoke_key(args.key_id)
        print("revoked" if revoked else "not_found")
        return 0 if revoked else 1

    if args.command == "set-limits":
        updated = api_key_store.set_tenant_limits(
            tenant_id=args.tenant,
            requests_per_minute=args.rpm,
            requests_per_day=args.rpd,
            max_jobs=args.max_jobs,
        )
        print(
            "tenant={tenant} rpm={rpm} rpd={rpd} max_jobs={max_jobs}".format(
                tenant=args.tenant,
                rpm=updated["requests_per_minute"],
                rpd=updated["requests_per_day"],
                max_jobs=updated["max_jobs"],
            )
        )
        return 0

    if args.command == "usage":
        summary = api_key_store.get_usage_summary(args.tenant)
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "export":
        rows = api_key_store.export_usage(
            tenant_id=args.tenant,
            days=args.days,
            limit=args.limit,
        )
        if args.format == "json":
            print(json.dumps(rows, indent=2))
            return 0

        fieldnames = [
            "created_at",
            "tenant_id",
            "key_id",
            "method",
            "path",
            "route_template",
            "status_code",
            "duration_ms",
            "job_id",
            "billing_tier",
            "billed_tokens",
            "unit_price_inr",
            "estimated_cost_inr",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
