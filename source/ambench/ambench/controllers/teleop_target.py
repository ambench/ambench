# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert incremental teleoperation input into absolute end-effector actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from ambench.tasks.base.base_env import BaseEnv

# SE(3) device readings are [position delta (3), rotation vector (3), gripper (1)].
DEVICE_ACTION_DIM = 7
_DELTA_SLICE = slice(0, 6)
_GRIPPER_SLICE = slice(6, 7)


class AbsoluteEETargetIntegrator:
    """Hold a running EE target and emit absolute end-effector actions.

    SE(3) teleoperation devices report incremental motion, while the public
    action interface is an absolute end-effector pose. This class keeps the
    running world-frame target that the environment used to maintain
    internally, applies each device increment to it, and returns the target in
    the absolute action layout: environment-relative position (3), WXYZ
    quaternion (4), gripper (1).

    The increment is applied with the same ``apply_delta_pose`` call the
    environment previously used, so teleoperation behaves as it did before,
    while the recorded actions now share their semantics with the scripted
    recorder and the learned-policy adapters.
    """

    def __init__(self, env: BaseEnv) -> None:
        self._env = env
        self._target_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        self._target_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
        self._target_quat_w[:, 0] = 1.0
        self.sync_from_env()

    def sync_from_env(self) -> None:
        """Re-seed the running target from the environment's EE command.

        Call this after every reset. The environment reseeds its own EE command
        from the measured end-effector pose there, so without a re-sync the
        target would keep drifting from wherever the previous episode ended.
        """
        self._target_pos_w = self._env.ee_cmd_pos_w.clone()
        self._target_quat_w = self._env.ee_cmd_quat_w.clone()

    def advance(self, device_action: torch.Tensor) -> torch.Tensor:
        """Apply one device reading and return the absolute EE-pose action."""
        reading = torch.as_tensor(device_action, device=self._env.device)
        reading = reading.reshape(-1, reading.shape[-1])
        if reading.shape[-1] != DEVICE_ACTION_DIM:
            raise ValueError(
                f"Teleoperation device returned width {reading.shape[-1]}, expected {DEVICE_ACTION_DIM} "
                "([position delta (3), rotation vector (3), gripper (1)])."
            )
        if reading.shape[0] == 1 and self._env.num_envs > 1:
            reading = reading.expand(self._env.num_envs, -1)

        self._target_pos_w, self._target_quat_w = math_utils.apply_delta_pose(
            self._target_pos_w,
            self._target_quat_w,
            reading[:, _DELTA_SLICE],
        )
        return torch.cat(
            [
                self._target_pos_w - self._env.scene.env_origins,
                self._target_quat_w,
                reading[:, _GRIPPER_SLICE],
            ],
            dim=-1,
        )
