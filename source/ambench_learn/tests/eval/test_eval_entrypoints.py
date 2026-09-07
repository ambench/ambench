# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("policy", ["act", "dp", "pi"])
def test_evaluator_help_does_not_launch_isaac(policy: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"ambench_learn.policies.{policy}.eval", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--num-envs" in result.stdout
    assert "--num_envs" not in result.stdout
    assert "[AppLauncher]" not in result.stdout + result.stderr


def test_dp_rejects_vectorized_eval_before_launch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ambench_learn.policies.dp.eval",
            "--task",
            "Task-v0",
            "--checkpoint",
            "missing",
            "--num-envs",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "not vectorized" in result.stderr
    assert "[AppLauncher]" not in result.stdout + result.stderr
