# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

UMI_ROOT = (
    Path(__file__).resolve().parents[3] / "ambench_learn" / "policies" / "dp" / "universal_manipulation_interface"
)
sys.path.insert(0, str(UMI_ROOT))

from umi.common.pose_util import mat_to_pose10d  # noqa: E402
from umi.real_world.real_inference_util import get_real_umi_action  # noqa: E402

from ambench_learn.policies.dp.eval_utils import (  # noqa: E402
    ObservationBufferManager,
    get_real_base_joint_action,
    resolve_dp_eval_action_semantics,
    resolve_dp_execution_schedule,
)


def test_dp_observation_history_padding_and_horizons() -> None:
    manager = ObservationBufferManager(2, 3, image_keys=["camera0_rgb"])
    manager.add_observation(
        {"camera0_rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
        np.array([1.0, 2.0, 3.0]),
        np.zeros(3),
        np.array([0.25]),
        robot_joint_pos=np.arange(4, dtype=np.float32),
        robot_base_pos=np.zeros(3),
        robot_base_rot=np.zeros(3),
    )

    observation = manager.get_stacked_observation()
    assert observation["camera0_rgb"].shape == (2, 4, 4, 3)
    assert observation["robot0_eef_pos"].shape == (3, 3)
    assert observation["robot0_joint_pos"].shape == (3, 4)
    np.testing.assert_array_equal(observation["robot0_eef_pos"], np.tile([1.0, 2.0, 3.0], (3, 1)))


def test_dp_base_joint_relative_layout_decodes_joint_deltas() -> None:
    relative_base_pose = mat_to_pose10d(np.eye(4, dtype=np.float32)[None])[0]
    action = np.concatenate([relative_base_pose, np.array([0.1, 0.2, 0.3, 0.4, -0.5], dtype=np.float32)])[None]
    env_obs = {
        "robot0_base_pos": np.zeros((1, 3), dtype=np.float32),
        "robot0_base_rot_axis_angle": np.zeros((1, 3), dtype=np.float32),
        "robot0_joint_pos": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    }

    decoded = get_real_base_joint_action(action, env_obs, action_pose_repr="rel")

    assert decoded.shape == (1, 12)
    np.testing.assert_allclose(decoded[0, :3], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(decoded[0, 7:11], [1.1, 2.2, 3.3, 4.4], atol=1.0e-6)
    assert decoded[0, 11] == -0.5


def test_dp_ee_relative_pose_conversion_uses_latest_measured_anchor() -> None:
    relative_pose = mat_to_pose10d(np.eye(4, dtype=np.float32)[None])[0]
    action = np.concatenate([relative_pose, np.array([-0.5], dtype=np.float32)])[None]
    env_obs = {
        "robot0_eef_pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros((1, 3), dtype=np.float32),
    }

    decoded = get_real_umi_action(action, env_obs, action_pose_repr="rel")

    assert decoded.shape == (1, 7)
    np.testing.assert_allclose(decoded[0, :3], [1.0, 2.0, 3.0], atol=1.0e-6)
    np.testing.assert_allclose(decoded[0, 3:6], 0.0, atol=1.0e-6)
    assert decoded[0, 6] == -0.5


def test_dp_execution_schedule_uses_half_horizon_replanning() -> None:
    assert resolve_dp_execution_schedule(action_horizon=16, obs_down_sample_steps=6) == (8, 48, 96)


def test_dp_eval_requires_exact_environment_action_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert resolve_dp_eval_action_semantics("ee_pose", env_cfg) == "ee_absolute"

    env_cfg.robot_profile.control.action_mode = "unsupported"
    with pytest.raises(ValueError, match="requires an env"):
        resolve_dp_eval_action_semantics("ee_pose", env_cfg)
