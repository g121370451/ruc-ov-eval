"""Environment-backed configuration helpers for benchmark entry points."""

from __future__ import annotations

import atexit
import json
import os
import re
from pathlib import Path
from typing import Any


_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_RUNTIME_CONFIGS: set[Path] = set()


def _cleanup_runtime_configs() -> None:
    for path in tuple(_RUNTIME_CONFIGS):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_runtime_configs)


def required_env(name: str) -> str:
    """Return one required, non-empty environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"环境变量 {name} 未设置或为空，请检查 ov_test/.env")
    return value


def resolve_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references and reject empty values."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return required_env(match.group(1))

        return _ENV_REFERENCE.sub(replace, value)
    if isinstance(value, dict):
        return {key: resolve_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def prepare_openviking_config(template_path: str, runtime_path: str) -> str:
    """Expand an env-only ov.conf template into an ignored runtime file."""
    source = Path(template_path)
    destination = Path(runtime_path)
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    resolved = resolve_env_vars(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, ensure_ascii=False, indent=2)
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    _RUNTIME_CONFIGS.add(destination)
    return str(destination)
