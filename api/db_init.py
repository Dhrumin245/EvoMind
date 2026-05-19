from pathlib import Path
from typing import Iterable, List, Optional

from api.storage import DatabaseTarget, connect_database, resolve_db_target


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

REQUIRED_TABLES = {
    "api_keys",
    "billing_ledger",
    "billing_topups",
    "job_commands",
    "job_events",
    "job_runtime_claims",
    "job_runtime_status",
    "jobs",
    "rate_limit_counters",
    "runtime_workers",
    "tenant_billing_accounts",
    "tenant_limits",
    "usage_logs",
    "user_accounts",
    "user_sessions",
    "webhook_deliveries",
    "webhook_delivery_attempts",
    "webhooks",
}


def _split_sql_statements(script: str) -> List[str]:
    statements: List[str] = []
    current: List[str] = []
    in_single_quote = False
    in_double_quote = False
    index = 0

    while index < len(script):
        char = script[index]
        next_char = script[index + 1] if index + 1 < len(script) else ""

        if not in_single_quote and not in_double_quote and char == "-" and next_char == "-":
            while index < len(script) and script[index] not in "\r\n":
                index += 1
            continue

        if char == "'" and not in_double_quote:
            current.append(char)
            if in_single_quote and next_char == "'":
                current.append(next_char)
                index += 2
                continue
            in_single_quote = not in_single_quote
            index += 1
            continue

        if char == '"' and not in_single_quote:
            current.append(char)
            in_double_quote = not in_double_quote
            index += 1
            continue

        if char == ";" and not in_single_quote and not in_double_quote:
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


def _migration_files() -> Iterable[Path]:
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _execute_migrations(connection) -> None:
    for migration_path in _migration_files():
        script = migration_path.read_text(encoding="utf-8")
        for statement in _split_sql_statements(script):
            connection.execute(statement)


def _existing_tables(connection) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """
    ).fetchall()
    return {str(row["table_name"]) for row in rows}


def verify_required_tables(connection) -> None:
    missing = sorted(REQUIRED_TABLES - _existing_tables(connection))
    if missing:
        raise RuntimeError(
            "PostgreSQL schema initialization failed; missing table(s): "
            + ", ".join(missing)
        )


def initialize_database(target: DatabaseTarget) -> None:
    with connect_database(target, timeout=30.0) as conn:
        _execute_migrations(conn)
        verify_required_tables(conn)
        conn.commit()


def initialize_control_plane_database() -> DatabaseTarget:
    target = resolve_db_target(
        context="API control plane",
        explicit_path=None,
        explicit_url=None,
        env_url_names=(
            "EVOMIND_API_AUTH_DB_URL",
            "EVOMIND_API_EVENTS_DB_URL",
            "EVOMIND_API_JOBS_DB_URL",
        ),
        default_path=None,
    )
    initialize_database(target)
    return target
