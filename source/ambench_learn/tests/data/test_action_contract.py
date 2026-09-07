# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from ambench_learn.data.action_resampling import (
    compute_stride,
    interpolate_absolute_action_for_execution,
    interpolate_base_joint_action_for_execution,
)
from ambench_learn.data.action_semantics import (
    BASE_JOINT_RELATIVE,
    EE_ABSOLUTE,
    EE_LOCAL_RELATIVE,
    dataset_metadata,
    resolve_eval_action_semantics,
    resolve_policy_action_representation_for_dataset,
    to_base_joint_absolute_trajectory,
    to_base_joint_relative_trajectory,
    to_ee_local_absolute_trajectory,
    to_ee_local_relative_trajectory,
)


def test_action_representation_resolution_and_legacy_rejection() -> None:
    assert resolve_policy_action_representation_for_dataset(EE_ABSOLUTE) == EE_LOCAL_RELATIVE
    assert resolve_policy_action_representation_for_dataset("base_joint_absolute") == BASE_JOINT_RELATIVE
    with pytest.raises(ValueError, match="must be one of"):
        resolve_policy_action_representation_for_dataset(EE_ABSOLUTE, "ee_relative")


def test_eval_action_semantics_accepts_environment_config(monkeypatch: pytest.MonkeyPatch) -> None:
    action_mode = SimpleNamespace(
        ABSOLUTE_EE_POSE="absolute_ee_pose",
        ABSOLUTE_BASE_JOINTS="absolute_base_joints",
    )
    monkeypatch.setitem(
        sys.modules,
        "ambench.controllers.control_pipeline",
        SimpleNamespace(ActionMode=action_mode),
    )
    env_cfg = SimpleNamespace(
        robot_profile=SimpleNamespace(
            control=SimpleNamespace(action_mode=action_mode.ABSOLUTE_EE_POSE),
        ),
    )

    assert resolve_eval_action_semantics(env_cfg) == EE_ABSOLUTE


def test_relative_transform_round_trips() -> None:
    ee_state = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.25]])
    ee_action = torch.tensor([[[2.0, 4.0, 6.0, -1.0, 0.0, 0.0, 0.0, -0.5]]])
    ee_round_trip = to_ee_local_absolute_trajectory(
        to_ee_local_relative_trajectory(ee_action, ee_state),
        ee_state,
    )
    torch.testing.assert_close(ee_round_trip[..., :3], ee_action[..., :3])
    torch.testing.assert_close(ee_round_trip[..., 7:], ee_action[..., 7:])
    assert torch.abs(torch.sum(ee_round_trip[..., 3:7] * ee_action[..., 3:7], dim=-1)).item() == pytest.approx(1.0)

    base_state = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    base_action = torch.tensor([[[2.0, 4.0, 6.0, -1.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, -0.5]]])
    base_round_trip = to_base_joint_absolute_trajectory(
        to_base_joint_relative_trajectory(base_action, base_state),
        base_state,
    )
    torch.testing.assert_close(base_round_trip[..., :3], base_action[..., :3])
    torch.testing.assert_close(base_round_trip[..., 7:], base_action[..., 7:])


def test_execution_resampling_preserves_endpoint_and_quaternion_continuity() -> None:
    assert compute_stride(120, 20) == 6
    previous_ee = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
    target_ee = torch.tensor([3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    interpolated_ee = interpolate_absolute_action_for_execution(previous_ee, target_ee, 3)
    torch.testing.assert_close(interpolated_ee[-1, :3], target_ee[:3])
    torch.testing.assert_close(interpolated_ee[-1, 3:7], -target_ee[3:7])
    torch.testing.assert_close(interpolated_ee[-1, 7:], target_ee[7:])

    previous_base = torch.cat((previous_ee[:7], torch.zeros(4), previous_ee[7:]))
    target_base = torch.cat((target_ee[:7], torch.ones(4), target_ee[7:]))
    interpolated_base = interpolate_base_joint_action_for_execution(previous_base, target_base, 2)
    torch.testing.assert_close(interpolated_base[-1, 7:11], target_base[7:11])


def test_dataset_metadata_reads_current_and_legacy_namespaces() -> None:
    current = {"ambench": {"action_semantics": EE_ABSOLUTE}}
    # Datasets recorded before the package rename carry the old key.
    legacy = {"am_isaac": {"action_semantics": EE_ABSOLUTE}}

    assert dataset_metadata(current)["action_semantics"] == EE_ABSOLUTE
    assert dataset_metadata(legacy)["action_semantics"] == EE_ABSOLUTE

    # The current key wins when both are somehow present.
    both = {"ambench": {"action_semantics": EE_ABSOLUTE}, "am_isaac": {"action_semantics": "stale"}}
    assert dataset_metadata(both)["action_semantics"] == EE_ABSOLUTE

    assert dataset_metadata({}) == {}
    assert dataset_metadata(None) == {}
