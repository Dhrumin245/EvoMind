import os
from pathlib import Path
from typing import Optional


def read_env_value(name: str, default: Optional[str] = None, strip: bool = True) -> Optional[str]:
    direct_value = os.getenv(name)
    file_value = os.getenv(f"{name}_FILE")

    if direct_value not in (None, "") and file_value not in (None, ""):
        raise ValueError(f"Set either {name} or {name}_FILE, not both")

    if direct_value not in (None, ""):
        return direct_value.strip() if strip else direct_value

    if file_value is not None and file_value.strip():
        secret_path = Path(file_value.strip())
        try:
            loaded_value = secret_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Unable to read secret file for {name}: {secret_path}") from exc
        return loaded_value.strip() if strip else loaded_value

    return default
