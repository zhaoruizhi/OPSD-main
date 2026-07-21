"""Write a reproducible manifest for an OPSD training run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


RUNTIME_ENVIRONMENT_KEYS = (
    "ACCELERATE_CONFIG_FILE",
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    return str(value)


def _config_dict(config: Any) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    elif is_dataclass(config):
        value = asdict(config)
    else:
        value = vars(config)
    return _json_safe(value)


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def write_training_experiment_manifest(
    output_file: str | Path,
    *,
    script_args: Any,
    training_args: Any,
    model_args: Any,
    skeleton_file: str | Path | None,
    repo_root: str | Path,
    argv: Sequence[str],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    repo_path = Path(repo_root).resolve()
    input_files: dict[str, dict[str, str]] = {}
    if skeleton_file not in (None, ""):
        resolved_skeleton = Path(skeleton_file).resolve()
        input_files["skeletons"] = {
            "path": str(resolved_skeleton),
            "sha256": sha256_file(resolved_skeleton),
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "effective_config": {
            "script_args": _config_dict(script_args),
            "training_args": _config_dict(training_args),
            "model_args": _config_dict(model_args),
        },
        "command": {"argv": [str(value) for value in argv]},
        "runtime": {
            "environment": {
                key: environ[key]
                for key in RUNTIME_ENVIRONMENT_KEYS
                if key in environ
            }
        },
        "input_files": input_files,
        "git": _git_state(repo_path),
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return payload
