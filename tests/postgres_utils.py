import os
import unittest
from typing import Iterable

from api.storage import DatabaseTarget, connect_database


CONTROL_PLANE_TABLES = (
    "webhook_delivery_attempts",
    "webhook_deliveries",
    "webhooks",
    "job_events",
    "runtime_workers",
    "job_runtime_status",
    "job_commands",
    "job_runtime_claims",
    "jobs",
    "billing_topups",
    "billing_ledger",
    "tenant_billing_accounts",
    "usage_logs",
    "rate_limit_counters",
    "tenant_limits",
    "user_sessions",
    "user_accounts",
    "api_keys",
)


def postgres_url() -> str:
    url = os.getenv("EVOMIND_TEST_CONTROL_PLANE_DB_URL") or os.getenv("EVOMIND_CONTROL_PLANE_DB_URL")
    if not url:
        raise unittest.SkipTest("PostgreSQL integration tests require EVOMIND_TEST_CONTROL_PLANE_DB_URL")
    return url


def reset_tables(url: str, tables: Iterable[str] = CONTROL_PLANE_TABLES) -> None:
    target = DatabaseTarget(url=url)
    quoted_tables = ", ".join(f'public."{table}"' for table in tables)
    if not quoted_tables:
        return
    with connect_database(target, timeout=30.0) as conn:
        existing_rows = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """
        ).fetchall()
        existing = {str(row["table_name"]) for row in existing_rows}
        selected = [table for table in tables if table in existing]
        if selected:
            conn.execute(
                "TRUNCATE TABLE "
                + ", ".join(f'public."{table}"' for table in selected)
                + " RESTART IDENTITY CASCADE"
            )
        conn.commit()
