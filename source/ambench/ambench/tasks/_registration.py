# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for consistent task registration."""

from __future__ import annotations

import gymnasium as gym


def register_env(
    task_id: str,
    entry_point: str,
    env_cfg_entry_point: str,
    scripted_policy_entry_point: str,
) -> None:
    """Register a single public task ID."""
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "scripted_policy_entry_point": scripted_policy_entry_point,
        },
    )
