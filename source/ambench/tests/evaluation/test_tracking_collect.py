# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_empty_joint_vectors_are_valid_for_fixed_base_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.normalize = lambda value: torch.nn.functional.normalize(value, dim=-1)
    math_utils.quat_error_magnitude = lambda first, second: torch.zeros(first.shape[0], dtype=first.dtype)
    math_utils.euler_xyz_from_quat = lambda quat: tuple(torch.zeros(quat.shape[0], dtype=quat.dtype) for _ in range(3))
    isaaclab = ModuleType("isaaclab")
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.math = math_utils
    isaaclab.utils = isaaclab_utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    from ambench.evaluation.tracking.collect import TrackingCollector

    robot = SimpleNamespace(max_arm_reach=1.0, multirotor=None)
    env_unwrapped = SimpleNamespace(
        cfg=SimpleNamespace(robot_profile=SimpleNamespace(robot=robot)),
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
        ee_cmd_pos_w=torch.zeros((1, 3)),
        ee_cmd_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        arm_targets=torch.empty((1, 0)),
        gripper_targets=torch.tensor([[0.02, 0.02]]),
        control_output=None,
        dt=1.0 / 120.0,
    )
    collector = TrackingCollector(SimpleNamespace(unwrapped=env_unwrapped))
    raw_obs = {
        "policy": [{
            "ee_pos": torch.zeros(3),
            "ee_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "base_pos": torch.zeros(3),
            "base_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "arm_joint_pos": torch.empty(0),
            "gripper_width": torch.tensor([0.02, 0.02]),
        }]
    }

    first_record = collector.collect(raw_obs, timestep=1)
    second_record = collector.collect(raw_obs, timestep=2)

    assert first_record["actual"]["joint_pos"] == []
    assert first_record["desired"]["joint_pos"] == []
    assert first_record["error"]["joint"] == []
    assert math.isnan(first_record["error"]["joint_l2_rad"])
    assert math.isnan(first_record["error"]["joint_abs_max_rad"])
    assert math.isnan(first_record["error"]["joint_mean_abs_rad"])
    assert math.isnan(second_record["control"]["joint_cmd_step_l2_rad"])
    assert math.isnan(second_record["control"]["joint_cmd_step_max_abs_rad"])
