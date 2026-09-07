# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_public_scripts_have_no_bash_entrypoints() -> None:
    result = subprocess.run(
        ["git", "ls-files", "*.sh", ":!ext/**"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_diffusers_pin_is_consistent_across_install_surfaces() -> None:
    # Only the two surfaces that govern a real install are checked. Installs go
    # through `uv pip install -e source/...`, which resolves these files; there
    # is no lockfile in the install path.
    expected = '"diffusers==0.36.0"'

    assert expected in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert expected in (REPO_ROOT / "source" / "ambench_learn" / "setup.py").read_text(encoding="utf-8")
