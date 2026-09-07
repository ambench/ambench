# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ambench_learn.policies.act.eval_utils import resolve_act_action_representation
from ambench_learn.policies.act.se_relative_processor import (
    BaseJointAbsoluteTrajectoryProcessorStep,
    EELocalAbsoluteTrajectoryProcessorStep,
)


@pytest.mark.parametrize(
    ("semantics", "step", "expected"),
    [
        ("ee_absolute", EELocalAbsoluteTrajectoryProcessorStep(enabled=True), "ee_local_relative"),
        (
            "base_joint_absolute",
            BaseJointAbsoluteTrajectoryProcessorStep(enabled=True),
            "base_joint_relative",
        ),
    ],
)
def test_act_representation_resolution(semantics: str, step, expected: str) -> None:
    result = resolve_act_action_representation(
        semantics,
        SimpleNamespace(action_representation=None),
        SimpleNamespace(steps=[step]),
    )
    assert result == expected


def test_act_rejects_legacy_representation() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        resolve_act_action_representation(
            "ee_absolute",
            SimpleNamespace(action_representation="ee_relative"),
            SimpleNamespace(steps=[]),
        )
