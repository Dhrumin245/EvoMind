import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote, urlencode, urlparse, urlunparse

from api.env_utils import read_env_value

PathLike = Union[str, Path]


class ManagedSqliteConnection(sqlite3.Connection):
    backend = "sqlite"

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class ManagedPostgresConnection:
    backend = "postgres"

    class _NoOpCursor:
        @property
        def rowcount(self) -> int:
            return -1

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    def __init__(self, connection: Any):
        self._connection = connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def __enter__(self) -> "ManagedPostgresConnection":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self._connection.commit()
            else:
                self._connection.rollback()
        finally:
            self._connection.close()

    @staticmethod
    def _adapt_query(query: str) -> Optional[str]:
        normalized = str(query)
        if normalized.strip().upper() in {"BEGIN", "BEGIN IMMEDIATE"}:
            return None
        return normalized.replace("?", "%s")

    def execute(self, query: str, params: Optional[Any] = None):
        adapted_query = self._adapt_query(query)
        if adapted_query is None:
            return self._NoOpCursor()
        if params is None:
            return self._connection.execute(adapted_query)
        return self._connection.execute(adapted_query, params)


@dataclass(frozen=True)
class DatabaseTarget:
    backend: str
    path: Optional[Path] = None
    url: Optional[str] = None

    @property
    def is_sqlite(self) -> bool:
        return self.backend == "sqlite"

    @property
    def is_postgres(self) -> bool:
        return self.backend == "postgres"

    @property
    def display_value(self) -> str:
        if self.path is not None:
            return str(self.path)
        return str(self.url or "")


def _resolve_path(value: PathLike) -> Path:
    path = Path(value)
    if str(path).strip() == "":
        raise ValueError("Path value must not be empty")
    return path


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: PathLike, create_parent: bool = False, create_dir: bool = False) -> Path:
    raw_value = os.getenv(name)
    path = _resolve_path(raw_value.strip() if raw_value is not None else default)
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    elif create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_postgres_url(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"postgres", "postgresql"}


def _require_managed_storage_for_production(target: DatabaseTarget, context: str) -> None:
    app_env = str(os.getenv("EVOMIND_ENV", "development")).strip().lower()
    allow_sqlite = _is_truthy(os.getenv("EVOMIND_ALLOW_SQLITE_IN_PRODUCTION"))
    if app_env == "production" and target.is_sqlite and not allow_sqlite:
        raise ValueError(
            f"{context} must use PostgreSQL in production. "
            "Set the appropriate *_DB_URL environment variable or explicitly opt in with "
            "EVOMIND_ALLOW_SQLITE_IN_PRODUCTION=true."
        )


def data_dir() -> Path:
    return _env_path("EVOMIND_DATA_DIR", Path("data"), create_dir=True)


def backup_dir() -> Path:
    return _env_path("EVOMIND_BACKUP_DIR", Path("backups"), create_dir=True)


def tenant_root_dir() -> Path:
    return _env_path("EVOMIND_TENANT_ROOT_DIR", data_dir() / "tenants", create_dir=True)


def api_auth_db_path() -> Path:
    return _env_path("EVOMIND_API_AUTH_DB", data_dir() / "api_auth.db", create_parent=True)


def api_events_db_path() -> Path:
    return _env_path("EVOMIND_API_EVENTS_DB", data_dir() / "api_events.db", create_parent=True)


def api_jobs_db_path() -> Path:
    return _env_path("EVOMIND_API_JOBS_DB", data_dir() / "api_jobs.db", create_parent=True)


def _control_plane_url(*specific_names: str) -> Optional[str]:
    for name in specific_names:
        raw_value = read_env_value(name)
        if raw_value:
            return raw_value
    generic = read_env_value("EVOMIND_CONTROL_PLANE_DB_URL")
    if generic:
        return generic
    return _build_control_plane_url_from_components()


def _build_control_plane_url_from_components() -> Optional[str]:
    host = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_HOST", "") or "").strip()
    port = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_PORT", "") or "").strip()
    database = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_NAME", "") or "").strip()
    user = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_USER", "") or "").strip()
    password = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_PASSWORD", "") or "").strip()
    sslmode = str(read_env_value("EVOMIND_CONTROL_PLANE_DB_SSLMODE", "") or "").strip()

    if not any((host, port, database, user, password, sslmode)):
        return None

    missing = [
        name
        for name, value in (
            ("EVOMIND_CONTROL_PLANE_DB_HOST", host),
            ("EVOMIND_CONTROL_PLANE_DB_NAME", database),
            ("EVOMIND_CONTROL_PLANE_DB_USER", user),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Incomplete control-plane database configuration. Missing: " + ", ".join(missing)
        )

    auth = quote(user, safe="")
    if password:
        auth += ":" + quote(password, safe="")

    netloc = f"{auth}@{host}"
    if port:
        netloc += f":{port}"

    query = urlencode({"sslmode": sslmode}) if sslmode else ""
    return urlunparse(
        (
            "postgresql",
            netloc,
            "/" + quote(database, safe=""),
            "",
            query,
            "",
        )
    )


def resolve_db_target(
    *,
    context: str,
    explicit_path: Optional[PathLike],
    explicit_url: Optional[str],
    env_url_names: tuple[str, ...],
    default_path: PathLike,
) -> DatabaseTarget:
    raw_url = str(explicit_url or _control_plane_url(*env_url_names) or "").strip()
    if raw_url:
        if not _is_postgres_url(raw_url):
            raise ValueError(f"{context} database URL must use the postgres:// or postgresql:// scheme")
        target = DatabaseTarget(backend="postgres", url=raw_url)
        _require_managed_storage_for_production(target, context)
        return target

    resolved_path = _resolve_path(explicit_path) if explicit_path is not None else _resolve_path(default_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    target = DatabaseTarget(backend="sqlite", path=resolved_path)
    _require_managed_storage_for_production(target, context)
    return target


def sqlite_connect(path: PathLike, timeout: float = 30.0) -> sqlite3.Connection:
    db_path = _resolve_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=float(timeout),
        factory=ManagedSqliteConnection,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={max(1000, int(float(timeout) * 1000.0))}")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def postgres_connect(url: str, timeout: float = 30.0) -> ManagedPostgresConnection:
    try:
        import psycopg  # type: ignore
        from psycopg.rows import dict_row  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL support requires psycopg. Install dependencies from requirements.txt."
        ) from exc

    connection = psycopg.connect(
        str(url),
        connect_timeout=max(1, int(timeout)),
        row_factory=dict_row,
    )
    return ManagedPostgresConnection(connection)


def connect_database(target: DatabaseTarget, timeout: float = 30.0):
    if target.is_postgres:
        assert target.url is not None
        return postgres_connect(target.url, timeout=timeout)
    assert target.path is not None
    return sqlite_connect(target.path, timeout=timeout)


def connection_backend(connection: Any) -> str:
    return str(getattr(connection, "backend", "sqlite"))


def column_names(connection: Any, table_name: str) -> set[str]:
    backend = connection_backend(connection)
    if backend == "postgres":
        row = connection.execute(
            """
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_name = %s
            ORDER BY CASE WHEN table_schema = 'public' THEN 0 ELSE 1 END, table_schema
            LIMIT 1
            """,
            (table_name,),
        ).fetchone()
        if row is None:
            return set()
        schema_name = str(row["table_schema"])
        rows = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, table_name),
        ).fetchall()
        return {str(row["column_name"]) for row in rows}

    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row["name"]) for row in rows}


def auto_increment_primary_key_sql(backend: str) -> str:
    if backend == "postgres":
        return "BIGSERIAL PRIMARY KEY"
    return "INTEGER PRIMARY KEY AUTOINCREMENT"


def snapshot_sqlite_database(source_path: Path, destination_path: Path) -> Path:
    source = _resolve_path(source_path)
    destination = _resolve_path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source_conn = sqlite3.connect(source)
    destination_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(destination_conn)
    finally:
        destination_conn.close()
        source_conn.close()
    return destination


def copytree_overwrite(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
