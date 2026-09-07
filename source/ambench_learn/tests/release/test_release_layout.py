# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_root_pyproject_is_tooling_only() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert "project" not in pyproject
    assert "build-system" not in pyproject


def test_public_scripts_have_no_bash_entrypoints() -> None:
    result = subprocess.run(
        ["git", "ls-files", "*.sh", ":!ext/**"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_diffusers_pin_is_defined_for_learning_install() -> None:
    # Installs go through `uv pip install -e source/...`; the root
    # pyproject.toml contains tooling configuration only.
    expected = '"diffusers==0.36.0"'

    assert expected in (REPO_ROOT / "source" / "ambench_learn" / "setup.py").read_text(encoding="utf-8")
