import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote, urlencode, urlparse, urlunparse

from api.env_utils import read_env_value

PathLike = Union[str, Path]


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
    backend: str = "postgres"
    path: Optional[Path] = None
    url: Optional[str] = None

    @property
    def is_postgres(self) -> bool:
        return self.backend == "postgres"

    @property
    def display_value(self) -> str:
        return str(self.url or "")


def _resolve_path(value: PathLike) -> Path:
    path = Path(value)
    if str(path).strip() == "":
        raise ValueError("Path value must not be empty")
    return path


def _env_path(name: str, default: PathLike, create_dir: bool = False) -> Path:
    raw_value = os.getenv(name)
    path = _resolve_path(raw_value.strip() if raw_value is not None else default)
    if create_dir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _is_postgres_url(value: str) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme in {"postgres", "postgresql"}


def data_dir() -> Path:
    return _env_path("EVOMIND_DATA_DIR", Path("data"), create_dir=True)


def backup_dir() -> Path:
    return _env_path("EVOMIND_BACKUP_DIR", Path("backups"), create_dir=True)


def tenant_root_dir() -> Path:
    return _env_path("EVOMIND_TENANT_ROOT_DIR", data_dir() / "tenants", create_dir=True)


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
    if sslmode and not any((host, port, database, user, password)):
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
    return urlunparse(("postgresql", netloc, "/" + quote(database, safe=""), "", query, ""))


def resolve_db_target(
    *,
    context: str,
    explicit_path: Optional[PathLike],
    explicit_url: Optional[str],
    env_url_names: tuple[str, ...],
    default_path: Optional[PathLike] = None,
    allow_unconfigured: bool = False,
) -> DatabaseTarget:
    if explicit_path is not None:
        raise ValueError(
            f"{context} no longer supports filesystem database paths. "
            "Configure PostgreSQL with EVOMIND_CONTROL_PLANE_DB_URL or the appropriate *_DB_URL variable."
        )

    raw_url = str(explicit_url or _control_plane_url(*env_url_names) or "").strip()
    if not raw_url:
        if allow_unconfigured:
            return DatabaseTarget(backend="postgres", url=None)
        raise ValueError(
            f"{context} requires PostgreSQL. "
            "Set EVOMIND_CONTROL_PLANE_DB_URL, a service-specific *_DB_URL, or the "
            "EVOMIND_CONTROL_PLANE_DB_HOST/NAME/USER component variables."
        )
    if not _is_postgres_url(raw_url):
        raise ValueError(f"{context} database URL must use the postgres:// or postgresql:// scheme")
    return DatabaseTarget(backend="postgres", url=raw_url)


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


def connect_database(target: DatabaseTarget, timeout: float = 30.0) -> ManagedPostgresConnection:
    if not target.url:
        raise ValueError(
            "PostgreSQL database URL is not configured. Set EVOMIND_CONTROL_PLANE_DB_URL "
            "or a service-specific *_DB_URL variable."
        )
    return postgres_connect(target.url, timeout=timeout)


def connection_backend(connection: Any) -> str:
    return str(getattr(connection, "backend", "postgres"))


def column_names(connection: Any, table_name: str) -> set[str]:
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


def copytree_overwrite(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
