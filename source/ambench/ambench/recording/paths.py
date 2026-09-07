# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for recording output paths and dataset metadata files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ENV_CFG_FILENAME = "env_cfg.yaml"
VIDEO_DIRNAME = "videos"
LEROBOT_DIRNAME = "lerobot"


def task_id_to_output_name(task_id: str) -> str:
    """Convert a task ID into a compact dataset or video directory name.

    Example:
        ``CabinetPickPlace-Am-EE-Abs-PID-Direct-v0`` -> ``CabinetPickPlaceEEAbsPID``

    The conversion removes the ``Isaac`` namespace token when present, drops the
    ``Am`` token, and stops at the first ``Direct`` or version token.

    Args:
        task_id: Registered Gym task ID, optionally with a namespace prefix.

    Returns:
        Compact task name suitable for dataset or video directory names.

    Raises:
        ValueError: If no usable name tokens can be derived from ``task_id``.
    """
    task_name = task_id.split(":")[-1]
    output_parts: list[str] = []

    for part in task_name.split("-"):
        if not part:
            continue
        if part in {"Isaac", "Am"}:
            continue
        if part == "Direct" or (part.startswith("v") and part[1:].isdigit()):
            break
        output_parts.append(part)

    if not output_parts:
        raise ValueError(f"Could not derive output name from task ID: {task_id}")

    return "".join(output_parts)


def resolve_dataset_dir(
    task_output_name: str,
    *,
    root_dir: str | Path,
) -> Path:
    """Resolve the output directory for a dataset collection run."""
    return _resolve_output_dir(
        task_output_name,
        root_dir=root_dir,
        prefix="demo",
    )


def resolve_video_dir(
    task_output_name: str,
    *,
    root_dir: str | Path,
) -> Path:
    """Resolve the output directory for a rollout video run."""
    return _resolve_output_dir(
        task_output_name,
        root_dir=root_dir,
        prefix="video",
    )


def write_env_cfg(output_dir: str | Path, env_cfg: Any) -> Path:
    """Write ``env_cfg.yaml`` into a dataset directory."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    env_cfg_path = output_path / ENV_CFG_FILENAME

    try:
        from isaaclab.utils.io import dump_yaml

        dump_yaml(str(env_cfg_path), env_cfg)
    except Exception:
        with env_cfg_path.open("w", encoding="utf-8") as cfg_file:
            yaml.safe_dump(env_cfg, cfg_file, sort_keys=False)

    return env_cfg_path


def load_env_cfg(dataset_dir: str | Path) -> dict[str, Any] | None:
    """Load ``env_cfg.yaml`` from a dataset directory when present."""
    env_cfg_path = Path(dataset_dir) / ENV_CFG_FILENAME
    if not env_cfg_path.is_file():
        return None

    with env_cfg_path.open("r", encoding="utf-8") as cfg_file:
        cfg_data = yaml.safe_load(cfg_file) or {}

    if not isinstance(cfg_data, dict):
        raise ValueError(f"Environment config must contain a mapping: {env_cfg_path}")

    return cfg_data


def _resolve_output_dir(
    task_output_name: str,
    *,
    root_dir: str | Path,
    prefix: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(root_dir) / task_output_name / f"{prefix}-{timestamp}").expanduser().resolve()
