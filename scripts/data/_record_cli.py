# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI helpers for dataset recording entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from ambench.recording.dataset import DEFAULT_STATE_KEYS
from ambench.recording.paths import resolve_dataset_dir, task_id_to_output_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets"


def add_dataset_export_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared LeRobot dataset export options to a recorder parser."""
    parser.add_argument(
        "--repo_id",
        type=str,
        default="",
        help="Optional LeRobot repo_id metadata to store in the dataset root.",
    )
    parser.add_argument(
        "--state_keys",
        type=str,
        nargs="+",
        default=list(DEFAULT_STATE_KEYS),
        help="Observation keys concatenated into canonical observation.state for LeRobot export.",
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default="",
        help="Optional task prompt stored with each recorded LeRobot frame.",
    )


def resolve_dataset_output_dir(task: str, dataset_root: str) -> Path:
    """Resolve a recording session directory under the requested dataset root."""
    return resolve_dataset_dir(
        task_id_to_output_name(task),
        root_dir=dataset_root or DEFAULT_DATASET_ROOT,
    )
