# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_public_policy_commands_reference_tracked_entrypoints() -> None:
    expected = (
        "scripts/data/record_demos_scripted.py",
        "scripts/data/validate_lerobotdataset.py",
        "scripts/data/dp/lerobot_to_zarr.py",
        "scripts/data/dp/validate_zarr.py",
        "scripts/data/export_lerobot_to_openpi.py",
        "source/ambench_learn/ambench_learn/policies/act/train.py",
        "source/ambench_learn/ambench_learn/policies/act/eval.py",
        "source/ambench_learn/ambench_learn/policies/dp/eval.py",
        "source/ambench_learn/ambench_learn/policies/pi/eval.py",
    )
    assert all((REPO_ROOT / path).is_file() for path in expected)
