import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
import uuid

from api.storage import (
    api_auth_db_path,
    api_events_db_path,
    api_jobs_db_path,
    backup_dir,
    copytree_overwrite,
    resolve_db_target,
    snapshot_sqlite_database,
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


def _stage_sqlite_snapshot(source_path: Path, staged_root: Path, archive_path: str) -> Optional[Dict[str, Any]]:
    if not source_path.exists():
        return None
    staged_path = staged_root / archive_path
    snapshot_sqlite_database(source_path, staged_path)
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
    targets = {
        "auth": resolve_db_target(
            context="API auth",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_AUTH_DB_URL",),
            default_path=api_auth_db_path(),
        ),
        "events": resolve_db_target(
            context="API events",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_EVENTS_DB_URL",),
            default_path=api_events_db_path(),
        ),
        "jobs": resolve_db_target(
            context="API jobs",
            explicit_path=None,
            explicit_url=None,
            env_url_names=("EVOMIND_API_JOBS_DB_URL",),
            default_path=api_jobs_db_path(),
        ),
    }
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

        sqlite_sources = {
            "auth": api_auth_db_path(),
            "events": api_events_db_path(),
            "jobs": api_jobs_db_path(),
        }
        for name, source_path in sqlite_sources.items():
            target_manifest = control_plane_targets[name]
            if target_manifest["backend"] != "sqlite":
                continue
            entry = _stage_sqlite_snapshot(
                source_path=source_path,
                staged_root=staging_root,
                archive_path=f"control_plane/{name}.sqlite3",
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
        for name, current_target in current_targets.items():
            snapshot_name = manifest.get("control_plane", {}).get(name, {}).get("snapshot")
            if not snapshot_name:
                continue
            if current_target["backend"] != "sqlite":
                raise ValueError(
                    f"Cannot restore SQLite backup for {name} while current storage backend is {current_target['backend']}"
                )
            destination = {
                "auth": api_auth_db_path(),
                "events": api_events_db_path(),
                "jobs": api_jobs_db_path(),
            }[name]
            if destination.exists() and not force:
                raise ValueError(f"Refusing to overwrite existing database without force=True: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if force:
                for sidecar in (destination, destination.with_name(f"{destination.name}-wal"), destination.with_name(f"{destination.name}-shm")):
                    if sidecar.exists():
                        sidecar.unlink()
            shutil.copy2(extracted_root / snapshot_name, destination)

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
        help="Overwrite existing SQLite databases and tenant artifacts",
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
