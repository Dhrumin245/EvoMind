import argparse
import hashlib
import json
import shutil
import tarfile
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit
import uuid

from api.storage import (
    DatabaseTarget,
    backup_dir,
    connect_database,
    copytree_overwrite,
    resolve_db_target,
    tenant_root_dir,
)


BACKUP_FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlsplit(url)
    if parsed.password is None:
        return url
    username = parsed.username or ""
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{username}:***@{hostname}{port}" if username else f"***@{hostname}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _resolve_control_plane_targets() -> Dict[str, DatabaseTarget]:
    return {
        "auth": resolve_db_target(
            context="API auth",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_AUTH_DB_URL",),
            default_path=None,
        ),
        "events": resolve_db_target(
            context="API events",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_EVENTS_DB_URL",),
            default_path=None,
        ),
        "jobs": resolve_db_target(
            context="API jobs",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_JOBS_DB_URL",),
            default_path=None,
        ),
    }


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, set):
        return [_json_safe_value(item) for item in sorted(value, key=str)]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _postgres_table_names(connection: Any) -> List[str]:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name ASC
        """
    ).fetchall()
    return [str(row["table_name"]) for row in rows]


def _postgres_column_names(connection: Any, table_name: str) -> List[str]:
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ?
        ORDER BY ordinal_position ASC
        """,
        (table_name,),
    ).fetchall()
    return [str(row["column_name"]) for row in rows]


def _stage_postgres_snapshot(target: DatabaseTarget, staged_root: Path, archive_path: str) -> Dict[str, Any]:
    staged_path = staged_root / archive_path
    staged_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "backend": "postgres",
        "captured_at": _utc_now(),
        "tables": [],
    }

    with connect_database(target, timeout=30.0) as conn:
        for table_name in _postgres_table_names(conn):
            columns = _postgres_column_names(conn, table_name)
            if columns:
                quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
                rows = conn.execute(
                    f"SELECT {quoted_columns} FROM public.{_quote_identifier(table_name)}"
                ).fetchall()
            else:
                rows = []
            payload["tables"].append(
                {
                    "name": table_name,
                    "columns": columns,
                    "rows": [
                        {
                            column: _json_safe_value(row[column])
                            for column in columns
                        }
                        for row in rows
                    ],
                }
            )

    staged_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "archive_path": archive_path,
        "size": staged_path.stat().st_size,
        "sha256": _sha256(staged_path),
    }


def _stage_directory(source_dir: Path, staged_root: Path, archive_path: str) -> Optional[Dict[str, Any]]:
    if not source_dir.exists():
        return None
    staged_path = staged_root / archive_path
    if staged_path.exists():
        shutil.rmtree(staged_path)
    shutil.copytree(source_dir, staged_path)
    file_entries: List[Dict[str, Any]] = []
    for file_path in sorted(path for path in staged_path.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(staged_root).as_posix()
        file_entries.append(
            {
                "archive_path": relative_path,
                "size": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
    return {
        "archive_path": archive_path,
        "files": file_entries,
    }


def _build_control_plane_manifest() -> Dict[str, Any]:
    targets = _resolve_control_plane_targets()
    manifest: Dict[str, Any] = {}
    for name, target in targets.items():
        manifest[name] = {
            "backend": target.backend,
            "path": str(target.path) if target.path is not None else None,
            "url": _redact_url(target.url),
        }
    return manifest


def create_backup(output_path: Optional[str] = None) -> Path:
    backups_root = backup_dir()
    archive_path = Path(output_path) if output_path is not None else backups_root / f"evomind-backup-{_timestamp_slug()}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    control_plane_targets = _build_control_plane_manifest()
    staging_root = backups_root / f".staging-backup-{uuid.uuid4().hex}"
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        entries: List[Dict[str, Any]] = []

        resolved_targets = _resolve_control_plane_targets()
        for name, target in resolved_targets.items():
            target_manifest = control_plane_targets[name]
            entry: Optional[Dict[str, Any]] = None
            if target.backend == "postgres":
                entry = _stage_postgres_snapshot(
                    target=target,
                    staged_root=staging_root,
                    archive_path=f"control_plane/{name}.postgres.json",
                )
            if entry is not None:
                entries.append(entry)
                target_manifest["snapshot"] = entry["archive_path"]

        artifact_entry = _stage_directory(
            source_dir=tenant_root_dir(),
            staged_root=staging_root,
            archive_path="artifacts/tenants",
        )
        if artifact_entry is not None:
            entries.extend(artifact_entry["files"])

        manifest = {
            "format_version": BACKUP_FORMAT_VERSION,
            "created_at": _utc_now(),
            "control_plane": control_plane_targets,
            "artifact_root": str(tenant_root_dir()),
            "entries": entries,
        }

        manifest_path = staging_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            for path in sorted(staging_root.rglob("*")):
                if path == manifest_path or path.is_dir():
                    continue
                archive.add(path, arcname=path.relative_to(staging_root).as_posix())
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    return archive_path


def _validate_backup_contents(extracted_root: Path, manifest: Dict[str, Any]) -> None:
    for entry in manifest.get("entries", []):
        archive_path = entry.get("archive_path")
        if not archive_path:
            continue
        target_path = extracted_root / str(archive_path)
        if not target_path.exists():
            raise ValueError(f"Backup entry is missing: {archive_path}")
        if target_path.is_file():
            expected_hash = str(entry.get("sha256", ""))
            if expected_hash and _sha256(target_path) != expected_hash:
                raise ValueError(f"Backup entry failed integrity verification: {archive_path}")


def _postgres_table_has_rows(connection: Any, table_name: str) -> bool:
    row = connection.execute(
        f"SELECT EXISTS (SELECT 1 FROM public.{_quote_identifier(table_name)} LIMIT 1) AS has_rows"
    ).fetchone()
    return bool(row and row.get("has_rows"))


def _reset_postgres_sequences(connection: Any, table_name: str, columns: Iterable[str]) -> None:
    qualified_table = f"public.{table_name}"
    quoted_table = f"public.{_quote_identifier(table_name)}"
    for column_name in columns:
        row = connection.execute(
            "SELECT pg_get_serial_sequence(?, ?) AS sequence_name",
            (qualified_table, column_name),
        ).fetchone()
        sequence_name = row["sequence_name"] if row is not None else None
        if not sequence_name:
            continue
        max_row = connection.execute(
            f"SELECT MAX({_quote_identifier(column_name)}) AS max_value FROM {quoted_table}"
        ).fetchone()
        max_value = max_row["max_value"] if max_row is not None else None
        if max_value in (None, 0):
            connection.execute("SELECT setval(?, ?, ?)", (sequence_name, 1, False)).fetchone()
            continue
        connection.execute("SELECT setval(?, ?, ?)", (sequence_name, max_value, True)).fetchone()


def _restore_postgres_snapshot(target: DatabaseTarget, snapshot_path: Path, force: bool) -> None:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tables = payload.get("tables", [])
    table_names = [str(item.get("name", "")).strip() for item in tables if str(item.get("name", "")).strip()]

    with connect_database(target, timeout=30.0) as conn:
        if not force:
            non_empty_tables = [table_name for table_name in table_names if _postgres_table_has_rows(conn, table_name)]
            if non_empty_tables:
                raise ValueError(
                    "Refusing to overwrite existing PostgreSQL data without force=True: "
                    + ", ".join(sorted(non_empty_tables))
                )

        if table_names:
            quoted_tables = ", ".join(f"public.{_quote_identifier(table_name)}" for table_name in table_names)
            conn.execute(f"TRUNCATE TABLE {quoted_tables} RESTART IDENTITY CASCADE")

        for table in tables:
            table_name = str(table.get("name", "")).strip()
            columns = [str(item) for item in table.get("columns", [])]
            rows = list(table.get("rows", []))
            if not table_name or not columns or not rows:
                if table_name and columns:
                    _reset_postgres_sequences(conn, table_name, columns)
                continue

            column_list = ", ".join(_quote_identifier(column) for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            insert_query = (
                f"INSERT INTO public.{_quote_identifier(table_name)} ({column_list}) VALUES ({placeholders})"
            )
            for row in rows:
                conn.execute(insert_query, tuple(row.get(column) for column in columns))
            _reset_postgres_sequences(conn, table_name, columns)


def restore_backup(archive_path: str, force: bool = False) -> None:
    source_archive = Path(archive_path)
    if not source_archive.exists():
        raise FileNotFoundError(f"Backup archive not found: {source_archive}")

    extracted_root = backup_dir() / f".staging-restore-{uuid.uuid4().hex}"
    if extracted_root.exists():
        shutil.rmtree(extracted_root, ignore_errors=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(source_archive, "r:gz") as archive:
            archive.extractall(extracted_root, filter="data")

        manifest_path = extracted_root / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Backup archive does not contain manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", 0)) != BACKUP_FORMAT_VERSION:
            raise ValueError("Unsupported backup format version")

        _validate_backup_contents(extracted_root, manifest)

        current_targets = _build_control_plane_manifest()
        resolved_targets = _resolve_control_plane_targets()
        for name, current_target in current_targets.items():
            snapshot_name = manifest.get("control_plane", {}).get(name, {}).get("snapshot")
            if not snapshot_name:
                continue
            snapshot_manifest = manifest.get("control_plane", {}).get(name, {})
            snapshot_backend = str(snapshot_manifest.get("backend", "")).strip().lower()
            current_backend = str(current_target["backend"]).strip().lower()
            if snapshot_backend != current_backend:
                raise ValueError(
                    f"Cannot restore {snapshot_backend or 'unknown'} backup for {name} while current storage backend is {current_backend or 'unknown'}"
                )

            if current_backend == "postgres":
                _restore_postgres_snapshot(
                    target=resolved_targets[name],
                    snapshot_path=extracted_root / snapshot_name,
                    force=force,
                )
                continue

            raise ValueError(f"Unsupported restore backend for {name}: {current_backend}")

        staged_tenants = extracted_root / "artifacts" / "tenants"
        if staged_tenants.exists():
            destination_root = tenant_root_dir()
            if destination_root.exists() and any(destination_root.iterdir()) and not force:
                raise ValueError(
                    f"Refusing to overwrite existing tenant artifact root without force=True: {destination_root}"
                )
            copytree_overwrite(staged_tenants, destination_root)
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and restore EvoMind storage backups")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a backup archive")
    create_parser.add_argument("--output", help="Optional backup archive path")

    restore_parser = subparsers.add_parser("restore", help="Restore a backup archive")
    restore_parser.add_argument("--input", required=True, help="Backup archive path")
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PostgreSQL data and tenant artifacts",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "create":
        archive = create_backup(output_path=args.output)
        print(archive)
        return 0

    if args.command == "restore":
        restore_backup(args.input, force=bool(args.force))
        print(f"restored: {args.input}")
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
