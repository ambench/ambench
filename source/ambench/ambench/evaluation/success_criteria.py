# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for aggregating binary task success criteria during evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class SuccessCriteriaTracker:
    """Track whether each binary success criterion has ever been met in one rollout."""

    def __init__(self, env_index: int = 0) -> None:
        self.env_index = env_index
        self._completed: dict[str, bool] = {}

    def update(self, info: Mapping[str, Any] | None) -> None:
        """Merge criteria from one env step into the rollout-level state."""
        if not info:
            return

        criteria = info.get("success_criteria", {})
        if not isinstance(criteria, Mapping):
            return

        for name, value in criteria.items():
            criterion = torch.as_tensor(value)
            if criterion.ndim > 0:
                criterion = criterion[self.env_index]
            criterion_name = str(name)
            self._completed[criterion_name] = self._completed.get(criterion_name, False) or bool(criterion.item())

    @property
    def completed_count(self) -> int:
        return sum(self._completed.values())

    @property
    def total_count(self) -> int:
        return len(self._completed)

    @property
    def completion_fraction(self) -> float | None:
        if self.total_count == 0:
            return None
        return self.completed_count / self.total_count

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe criteria summary."""
        return {
            "criteria": dict(sorted(self._completed.items())),
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "subtask_completion": self.completion_fraction,
        }
