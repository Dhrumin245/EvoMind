import os
from pathlib import Path
from typing import Dict, Optional


_ENV_FILE_CACHE: Dict[str, Dict[str, str]] = {}


def _read_env_file(path: Path) -> Dict[str, str]:
    cache_key = str(path.resolve())
    if cache_key in _ENV_FILE_CACHE:
        return _ENV_FILE_CACHE[cache_key]

    values: Dict[str, str] = {}
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        _ENV_FILE_CACHE[cache_key] = values
        return values

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value

    _ENV_FILE_CACHE[cache_key] = values
    return values


def _read_local_env_value(name: str) -> Optional[str]:
    for filename in (".env", ".env.production"):
        value = _read_env_file(Path.cwd() / filename).get(name)
        if value not in (None, ""):
            return value
    return None


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

    local_value = _read_local_env_value(name)
    if local_value not in (None, ""):
        return local_value.strip() if strip else local_value

    return default
