# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _install_isaaclab_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the single isaaclab math helper the integrator uses.

    ``apply_delta_pose`` is stubbed as a plain translation with the orientation
    left untouched. That keeps these tests focused on the parts this module is
    responsible for -- seeding, environment-relative conversion, gripper
    passthrough, and broadcasting -- rather than re-testing Isaac Lab's
    rotation math.
    """

    math_utils = ModuleType("isaaclab.utils.math")

    def apply_delta_pose(pos, quat, delta):
        return pos + delta[:, :3], quat

    math_utils.apply_delta_pose = apply_delta_pose
    isaaclab = ModuleType("isaaclab")
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.math = math_utils
    isaaclab.utils = isaaclab_utils
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)


def _fake_env(num_envs: int = 1, origin: float = 0.0, ee_pos: float = 0.0):
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=SimpleNamespace(env_origins=torch.full((num_envs, 3), origin)),
        ee_cmd_pos_w=torch.full((num_envs, 3), ee_pos),
        ee_cmd_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
    )


def _reading(dx: float = 0.0, gripper: float = 0.0) -> torch.Tensor:
    return torch.tensor([[dx, 0.0, 0.0, 0.0, 0.0, 0.0, gripper]])


def test_action_is_absolute_and_environment_relative(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    # End-effector sits at world 5.0 in an environment whose origin is 5.0, so
    # the environment-relative target starts at zero.
    env = _fake_env(origin=5.0, ee_pos=5.0)
    integrator = AbsoluteEETargetIntegrator(env)

    action = integrator.advance(_reading(dx=0.25))

    assert action.shape == (1, 8)
    # Position is reported relative to the environment origin, not in world frame.
    assert torch.allclose(action[:, :3], torch.tensor([[0.25, 0.0, 0.0]]))
    assert torch.allclose(action[:, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_increments_accumulate_into_a_running_target(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    integrator = AbsoluteEETargetIntegrator(_fake_env())

    integrator.advance(_reading(dx=0.1))
    action = integrator.advance(_reading(dx=0.1))

    # A second identical increment must move the target further, not repeat it.
    assert torch.allclose(action[:, :3], torch.tensor([[0.2, 0.0, 0.0]]))


def test_gripper_passes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    integrator = AbsoluteEETargetIntegrator(_fake_env())

    action = integrator.advance(_reading(gripper=-0.6))

    assert action[0, 7] == pytest.approx(-0.6)


def test_sync_from_env_reseeds_after_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    env = _fake_env()
    integrator = AbsoluteEETargetIntegrator(env)
    integrator.advance(_reading(dx=1.0))

    # A reset moves the end-effector; without a re-sync the stale target would
    # carry the previous episode's accumulated offset.
    env.ee_cmd_pos_w = torch.full((1, 3), 3.0)
    integrator.sync_from_env()
    action = integrator.advance(_reading(dx=0.5))

    # Target restarts from the post-reset pose (3.0) plus this increment only,
    # with no trace of the 1.0 accumulated before the reset.
    assert torch.allclose(action[:, :3], torch.tensor([[3.5, 3.0, 3.0]]))


def test_single_reading_broadcasts_to_all_environments(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    integrator = AbsoluteEETargetIntegrator(_fake_env(num_envs=4))

    action = integrator.advance(_reading(dx=0.3))

    assert action.shape == (4, 8)
    assert torch.allclose(action[:, 0], torch.full((4,), 0.3))


def test_unexpected_device_width_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_isaaclab_stub(monkeypatch)
    from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator

    integrator = AbsoluteEETargetIntegrator(_fake_env())

    with pytest.raises(ValueError, match="expected 7"):
        integrator.advance(torch.zeros((1, 6)))
