# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ambench_learn.data.action_semantics import BASE_JOINT_ABSOLUTE, EE_ABSOLUTE
from ambench_learn.policies.pi.eval_utils import (
    build_openpi_example,
    expand_policy_actions,
)


def _env(action_dim: int) -> SimpleNamespace:
    sensors = {
        "ee_camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.full((1, 2, 3, 3), 7)})),
    }
    unwrapped = SimpleNamespace(
        device="cpu",
        scene=SimpleNamespace(
            env_origins=torch.zeros((1, 3)),
            sensors=sensors,
        ),
        ee_cmd_pos_w=torch.zeros((1, 3)),
        ee_cmd_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        actions=torch.tensor([[0.0] * (action_dim - 1) + [-0.25]]),
    )
    return SimpleNamespace(unwrapped=unwrapped, action_space=SimpleNamespace(shape=(action_dim,)))


def test_openpi_examples_use_measured_ee_and_base_joint_state() -> None:
    env = _env(12)
    ee_obs = {
        "policy": [{
            "ee_pos": torch.tensor([1.0, 2.0, 3.0]),
            "ee_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "gripper_width": torch.tensor([0.04]),
        }]
    }
    ee_example = build_openpi_example(
        ee_obs,
        env,
        env_index=0,
        action_semantics=EE_ABSOLUTE,
        prompt="press the button",
    )
    assert set(ee_example) == {
        "am_bench/ee_pos",
        "am_bench/ee_quat",
        "am_bench/gripper_width",
        "am_bench/ee_image",
        "prompt",
    }
    np.testing.assert_array_equal(ee_example["am_bench/ee_pos"], [1.0, 2.0, 3.0])
    assert ee_example["am_bench/ee_image"].shape == (2, 3, 3)

    base_joint_obs = {
        "policy": [{
            "base_pos": torch.tensor([4.0, 5.0, 6.0]),
            "base_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "arm_joint_pos": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "gripper_width": torch.tensor([0.05]),
        }]
    }
    base_joint_example = build_openpi_example(
        base_joint_obs,
        env,
        env_index=0,
        action_semantics=BASE_JOINT_ABSOLUTE,
        prompt="press the button",
    )
    assert "am_bench/ee_pos" not in base_joint_example
    np.testing.assert_array_equal(base_joint_example["am_bench/base_pos"], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(base_joint_example["am_bench/arm_joint_pos"], [0.1, 0.2, 0.3, 0.4])


def test_openpi_action_expansion_preserves_ee_and_base_joint_endpoints() -> None:
    ee_env = _env(8)
    ee_actions = np.array([[3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
    ee_plan = expand_policy_actions(
        ee_actions,
        ee_env,
        action_semantics=EE_ABSOLUTE,
        n_action_steps=1,
        action_execution_stride=3,
    )
    assert len(ee_plan) == 3
    torch.testing.assert_close(ee_plan[-1][:3], torch.tensor([3.0, 0.0, 0.0]))
    torch.testing.assert_close(ee_plan[-1][3:7], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    base_joint_env = _env(12)
    base_joint_actions = np.array(
        [[3.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]],
        dtype=np.float32,
    )
    base_joint_plan = expand_policy_actions(
        base_joint_actions,
        base_joint_env,
        action_semantics=BASE_JOINT_ABSOLUTE,
        n_action_steps=1,
        action_execution_stride=2,
    )
    assert len(base_joint_plan) == 2
    torch.testing.assert_close(base_joint_plan[-1][:3], torch.tensor([3.0, 0.0, 0.0]))
    torch.testing.assert_close(base_joint_plan[-1][7:], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))


def test_openpi_action_expansion_rejects_environment_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected action dim 10 to match env action space, got 8"):
        expand_policy_actions(
            np.zeros((1, 8), dtype=np.float32),
            _env(10),
            action_semantics=EE_ABSOLUTE,
            n_action_steps=1,
            action_execution_stride=1,
        )
