import argparse
import sys
import tarfile
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.backup import create_backup
from api.storage import backup_dir


def _sorted_backup_archives(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.glob("evomind-backup-*.tar.gz") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _verify_archive(archive_path: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
    if "manifest.json" not in members:
        raise ValueError(f"Backup archive is missing manifest.json: {archive_path}")


def _prune_archives(archives: Iterable[Path], keep_last: int, max_age_days: int | None) -> list[Path]:
    import time

    now = time.time()
    removed: list[Path] = []
    for index, archive_path in enumerate(archives):
        age_days = max(0.0, (now - archive_path.stat().st_mtime) / 86400.0)
        should_remove = index >= max(0, int(keep_last))
        if max_age_days is not None and age_days > max(0, int(max_age_days)):
            should_remove = True
        if not should_remove:
            continue
        archive_path.unlink(missing_ok=True)
        removed.append(archive_path)
    return removed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a production backup and prune old archives")
    parser.add_argument(
        "--output-dir",
        default=str(backup_dir()),
        help="Directory where backup archives are stored",
    )
    parser.add_argument(
        "--keep-last",
        type=int,
        default=14,
        help="Keep at least this many recent archives",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete archives older than this age in days; use a negative value to disable age pruning",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = create_backup(
        output_path=str(output_dir / f"evomind-backup-{__import__('datetime').datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.tar.gz")
    )
    _verify_archive(archive_path)

    removed_archives = _prune_archives(
        _sorted_backup_archives(output_dir),
        keep_last=max(0, int(args.keep_last)),
        max_age_days=None if int(args.max_age_days) < 0 else int(args.max_age_days),
    )

    print(f"created: {archive_path}")
    for removed in removed_archives:
        print(f"pruned: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
