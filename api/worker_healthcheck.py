import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode, urlparse, urlunparse

from api.env_utils import read_env_value


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_postgres_url(value: str) -> bool:
    return urlparse(str(value).strip()).scheme in {"postgres", "postgresql"}


def _control_plane_url() -> Optional[str]:
    for name in ("EVOMIND_API_JOBS_DB_URL", "EVOMIND_CONTROL_PLANE_DB_URL"):
        raw_value = read_env_value(name)
        if raw_value:
            return raw_value

    host = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_HOST", "") or "").strip()
    database = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_NAME", "") or "").strip()
    user = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_USER", "") or "").strip()
    password = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_PASSWORD", "") or "").strip()
    port = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_PORT", "") or "").strip()
    sslmode = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_SSLMODE", "") or "").strip()

    if not any((host, database, user, password, port, sslmode)):
        return None
    if not host or not database or not user:
        return None

    auth = quote(user, safe="")
    if password:
        auth += ":" + quote(password, safe="")
    netloc = f"{auth}@{host}"
    if port:
        netloc += f":{port}"
    query = urlencode({"sslmode": sslmode}) if sslmode else ""
    return urlunparse(("postgresql", netloc, "/" + quote(database, safe=""), "", query, ""))


def _sqlite_path() -> Path:
    explicit = os.getenv("EVOMIND_API_JOBS_DB")
    if explicit:
        return Path(explicit)
    data_dir = Path(os.getenv("EVOMIND_DATA_DIR", "data"))
    return data_dir / "api_jobs.db"


def _has_active_worker_postgres(url: str) -> bool:
    import psycopg

    with psycopg.connect(str(url), connect_timeout=3) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM runtime_workers
                WHERE worker_type = %s AND lease_expires_at > %s
                LIMIT 1
                """,
                ("training", _utc_now()),
            )
            return cursor.fetchone() is not None


def _has_active_worker_sqlite(path: Path) -> bool:
    with sqlite3.connect(path, timeout=3.0) as conn:
        cursor = conn.execute(
            """
            SELECT 1
            FROM runtime_workers
            WHERE worker_type = ? AND lease_expires_at > ?
            LIMIT 1
            """,
            ("training", _utc_now()),
        )
        return cursor.fetchone() is not None


def main() -> int:
    try:
        database_url = _control_plane_url()
        if database_url:
            if not _is_postgres_url(database_url):
                return 1
            return 0 if _has_active_worker_postgres(database_url) else 1
        return 0 if _has_active_worker_sqlite(_sqlite_path()) else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
