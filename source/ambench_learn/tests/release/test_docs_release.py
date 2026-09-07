# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC_DOCS = tuple((REPO_ROOT / "docs").rglob("*.md"))


def test_public_docs_do_not_reference_removed_or_internal_workflows() -> None:
    forbidden = (
        ".sh",
        "scripts/experiments",
        "verify_zarr.py",
        "act_dp_pi_policy_alignment_audit.md",
        "cleanup_pr_151.md",
        "control_representation_alignment_notes.md",
        "PressButton-Am-EE-Delta",
        "/home/",
        "/datasets/",
    )
    violations = {
        str(path.relative_to(REPO_ROOT)): token
        for path in PUBLIC_DOCS
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }
    assert violations == {}


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


def test_mkdocs_exposes_data_and_all_policy_guides() -> None:
    """Every data and policy guide must be reachable from the site navigation.

    This asserts on the guide pages rather than on section titles. The previous
    version keyed off "Data" and "Policies" sections, and silently started
    failing when the navigation was reorganised around task-oriented sections.
    """

    config = yaml.safe_load((REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    def nav_pages(node) -> set[str]:
        if isinstance(node, str):
            return {node} if node.endswith(".md") else set()
        if isinstance(node, dict):
            return set().union(*(nav_pages(v) for v in node.values())) if node else set()
        if isinstance(node, list):
            return set().union(*(nav_pages(v) for v in node)) if node else set()
        return set()

    pages = nav_pages(config["nav"])

    required = {
        # Data workflow
        "workflows/collect-demos.md",
        "workflows/validate-data.md",
        "reference/dataset-format.md",
        # Every supported policy family
        "policies/index.md",
        "policies/scripted-policies.md",
        "policies/act.md",
        "policies/diffusion-policy.md",
        "policies/openpi.md",
    }
    assert required <= pages, f"missing from mkdocs nav: {sorted(required - pages)}"

    # Anything reachable from the navigation must actually exist.
    missing = sorted(page for page in pages if not (REPO_ROOT / "docs" / page).is_file())
    assert missing == [], f"nav references missing pages: {missing}"
