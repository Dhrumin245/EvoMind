import asyncio
import argparse
import csv
import hashlib
import hmac
import json
import secrets
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

BILLING_CATALOG = [
    {
        "method": "GET",
        "route_template": "/usage/limits",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Read tenant quota configuration",
    },
    {
        "method": "GET",
        "route_template": "/usage/summary",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Read tenant usage summary",
    },
    {
        "method": "GET",
        "route_template": "/usage/billing-tiers",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Read billing tier catalog",
    },
    {
        "method": "GET",
        "route_template": "/usage/export",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Export tenant usage history",
    },
    {
        "method": "GET",
        "route_template": "/webhooks",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "List tenant webhooks",
    },
    {
        "method": "POST",
        "route_template": "/webhooks",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Create tenant webhook",
    },
    {
        "method": "DELETE",
        "route_template": "/webhooks/{webhook_id}",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Delete tenant webhook",
    },
    {
        "method": "GET",
        "route_template": "/webhooks/{webhook_id}/deliveries",
        "billing_tier": "admin_free",
        "unit_name": "request",
        "unit_price_usd": 0.0,
        "description": "Inspect webhook delivery history",
    },
    {
        "method": "POST",
        "route_template": "/jobs",
        "billing_tier": "job_management",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "Create a job",
    },
    {
        "method": "GET",
        "route_template": "/jobs",
        "billing_tier": "job_management",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "List jobs",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}",
        "billing_tier": "job_management",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "Get job metadata",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/events",
        "billing_tier": "job_events",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "List job lifecycle events",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/start",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Start training",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/stop",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Stop training",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/resume",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Resume training",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/status",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "Read training status snapshot",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/insights",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "Read trend insights",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/metrics",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.015,
        "description": "Read raw metrics rows",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/train/checkpoints",
        "billing_tier": "training_artifacts",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "List checkpoints",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/train/checkpoints",
        "billing_tier": "training_artifacts",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Create checkpoint",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/agent/action",
        "billing_tier": "inference_single",
        "unit_name": "request",
        "unit_price_usd": 0.03,
        "description": "Single inference call",
    },
    {
        "method": "POST",
        "route_template": "/jobs/{job_id}/agent/action/batch",
        "billing_tier": "inference_batch",
        "unit_name": "observation",
        "unit_price_usd": 0.01,
        "description": "Batch inference charged per observation",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/agent/info",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "Selected agent metadata",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/genomes",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "List genomes",
    },
    {
        "method": "GET",
        "route_template": "/jobs/{job_id}/genomes/{genome_id}",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "Genome metadata lookup",
    },
    {
        "method": "POST",
        "route_template": "/train/start",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Start training on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/stop",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Stop training on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/resume",
        "billing_tier": "training_control",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Resume training on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/status",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "Read training status snapshot on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/insights",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "Read trend insights on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/metrics",
        "billing_tier": "training_metrics",
        "unit_name": "request",
        "unit_price_usd": 0.015,
        "description": "Read raw metrics rows on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/train/checkpoints",
        "billing_tier": "training_artifacts",
        "unit_name": "request",
        "unit_price_usd": 0.01,
        "description": "List checkpoints on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/train/checkpoints",
        "billing_tier": "training_artifacts",
        "unit_name": "request",
        "unit_price_usd": 0.02,
        "description": "Create checkpoint on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/agent/action",
        "billing_tier": "inference_single",
        "unit_name": "request",
        "unit_price_usd": 0.03,
        "description": "Single inference call on the tenant default job",
    },
    {
        "method": "POST",
        "route_template": "/agent/action/batch",
        "billing_tier": "inference_batch",
        "unit_name": "observation",
        "unit_price_usd": 0.01,
        "description": "Batch inference on the tenant default job, charged per observation",
    },
    {
        "method": "GET",
        "route_template": "/agent/info",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "Selected agent metadata on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/genomes",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "List genomes on the tenant default job",
    },
    {
        "method": "GET",
        "route_template": "/genomes/{genome_id}",
        "billing_tier": "model_catalog",
        "unit_name": "request",
        "unit_price_usd": 0.005,
        "description": "Genome metadata lookup on the tenant default job",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class APIKeyPrincipal:
    key_id: str
    name: str
    tenant_id: str
    status: str


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
                    billed_units INTEGER NOT NULL DEFAULT 1,
                    unit_price_usd REAL NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0,
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

            usage_log_columns = column_names(conn, "usage_logs")
            usage_log_migrations = {
                "route_template": "TEXT",
                "billing_tier": "TEXT NOT NULL DEFAULT 'unclassified'",
                "billed_units": "INTEGER NOT NULL DEFAULT 1",
                "unit_price_usd": "REAL NOT NULL DEFAULT 0",
                "estimated_cost_usd": "REAL NOT NULL DEFAULT 0",
            }
            for column_name, column_definition in usage_log_migrations.items():
                if column_name not in usage_log_columns:
                    conn.execute(
                        f"ALTER TABLE usage_logs ADD COLUMN {column_name} {column_definition}"
                    )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_usage_logs_tenant_created
                ON usage_logs (tenant_id, created_at)
                """
            )
            conn.commit()

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
    def _coerce_billed_units(billed_units: Optional[int]) -> int:
        if billed_units is None:
            return 1
        return max(1, int(billed_units))

    @staticmethod
    def _is_chargeable_status(status_code: int) -> bool:
        return 200 <= int(status_code) < 400

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
            "unit_name": "request",
            "unit_price_usd": 0.0,
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

    def create_key(self, name: str, tenant_id: str = "default") -> tuple[APIKeyPrincipal, str]:
        key_id, raw_key = self._generate_key_material()
        salt_hex = secrets.token_hex(16)
        key_hash = self._hash_key(raw_key, salt_hex)
        created_at = _utc_now()
        self.ensure_tenant_limits(tenant_id)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_id, name, tenant_id, salt, key_hash, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'active', ?)
                """,
                (key_id, name, tenant_id, salt_hex, key_hash, created_at),
            )
            conn.commit()

        principal = APIKeyPrincipal(
            key_id=key_id,
            name=name,
            tenant_id=tenant_id,
            status="active",
        )
        return principal, raw_key

    def list_keys(self) -> List[APIKeyPrincipal]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key_id, name, tenant_id, status
                FROM api_keys
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [
            APIKeyPrincipal(
                key_id=row["key_id"],
                name=row["name"],
                tenant_id=row["tenant_id"],
                status=row["status"],
            )
            for row in rows
        ]

    def revoke_key(self, key_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE key_id = ? AND status != 'revoked'",
                (key_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def resolve_key(self, raw_key: str) -> Optional[APIKeyPrincipal]:
        key_id = self._extract_key_id(raw_key)
        if key_id is None:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT key_id, name, tenant_id, salt, key_hash, status
                FROM api_keys
                WHERE key_id = ?
                """,
                (key_id,),
            ).fetchone()

            if row is None or row["status"] != "active":
                return None

            candidate_hash = self._hash_key(raw_key, row["salt"])
            if not hmac.compare_digest(candidate_hash, row["key_hash"]):
                return None

            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (_utc_now(), key_id),
            )
            conn.commit()

        self.ensure_tenant_limits(str(row["tenant_id"]))
        return APIKeyPrincipal(
            key_id=row["key_id"],
            name=row["name"],
            tenant_id=row["tenant_id"],
            status=row["status"],
        )

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
        billed_units: Optional[int] = None,
    ) -> None:
        billing_definition = self.resolve_billing_definition(
            method=method,
            route_template=route_template,
            path=path,
        )
        normalized_units = self._coerce_billed_units(billed_units)
        unit_price_usd = float(billing_definition["unit_price_usd"])
        estimated_cost_usd = 0.0
        if self._is_chargeable_status(status_code):
            estimated_cost_usd = self._round_cost(normalized_units * unit_price_usd)

        with self._connect() as conn:
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
                    billed_units,
                    unit_price_usd,
                    estimated_cost_usd,
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
                    normalized_units,
                    unit_price_usd,
                    estimated_cost_usd,
                    _utc_now(),
                ),
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
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_logs
                WHERE tenant_id = ? AND created_at >= ?
                """,
                (tenant_id, day_threshold),
            ).fetchone()
            total_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM usage_logs
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()

        minute_count = int(rate_snapshot["minute_count"])
        day_count = int(rate_snapshot["day_count"])
        total_count = int(total_row["request_count"])
        day_cost = self._round_cost(float(day_row["estimated_cost_usd"] or 0.0))
        total_cost = self._round_cost(float(total_row["estimated_cost_usd"] or 0.0))

        return {
            "requests_last_minute": minute_count,
            "requests_last_day": day_count,
            "requests_total": total_count,
            "requests_per_minute_limit": rate_snapshot["requests_per_minute"],
            "requests_per_day_limit": rate_snapshot["requests_per_day"],
            "max_jobs": rate_snapshot["max_jobs"],
            "remaining_this_minute": max(0, rate_snapshot["requests_per_minute"] - minute_count),
            "remaining_today": max(0, rate_snapshot["requests_per_day"] - day_count),
            "estimated_cost_last_day_usd": day_cost,
            "estimated_cost_total_usd": total_cost,
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
                    billed_units,
                    unit_price_usd,
                    estimated_cost_usd
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
                    "billed_units": int(row["billed_units"] or 1),
                    "unit_price_usd": self._round_cost(float(row["unit_price_usd"] or 0.0)),
                    "estimated_cost_usd": self._round_cost(
                        float(row["estimated_cost_usd"] or 0.0)
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
            detail="Invalid or revoked API key",
        )

    request.state.principal = principal
    merge_log_context(tenant_id=principal.tenant_id, key_id=principal.key_id)
    request.state.rate_limits = await asyncio.to_thread(
        api_key_store.consume_rate_limit,
        principal.tenant_id,
    )
    return principal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage EvoMind API keys, limits, billing tiers, and usage exports"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--name", required=True, help="Friendly key name")
    create_parser.add_argument("--tenant", default="default", help="Tenant identifier")

    subparsers.add_parser("list", help="List API keys")
    subparsers.add_parser("billing-tiers", help="List endpoint billing tiers")

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
        principal, raw_key = api_key_store.create_key(name=args.name, tenant_id=args.tenant)
        print(f"key_id: {principal.key_id}")
        print(f"name: {principal.name}")
        print(f"tenant_id: {principal.tenant_id}")
        print(f"api_key: {raw_key}")
        return 0

    if args.command == "list":
        for item in api_key_store.list_keys():
            print(
                f"key_id={item.key_id} name={item.name} tenant_id={item.tenant_id} status={item.status}"
            )
        return 0

    if args.command == "billing-tiers":
        for item in api_key_store.get_billing_catalog():
            print(
                "{method} {route_template} tier={billing_tier} unit={unit_name} "
                "price_usd={unit_price_usd:.6f} description={description}".format(**item)
            )
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
            "billed_units",
            "unit_price_usd",
            "estimated_cost_usd",
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
