import gc
import shutil
import stat
import time
from pathlib import Path


def cleanup_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return

    gc.collect()

    def _on_error(func, failed_path, exc_info):
        failed_target = Path(failed_path)
        try:
            failed_target.chmod(stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            return
        try:
            func(failed_path)
        except OSError:
            return

    for attempt in range(3):
        try:
            if target.is_dir():
                shutil.rmtree(target, onerror=_on_error)
            else:
                target.unlink()
            return
        except OSError:
            if attempt == 2:
                raise
            gc.collect()
            time.sleep(0.1)
