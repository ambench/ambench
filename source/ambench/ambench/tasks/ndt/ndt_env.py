# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .ndt_env_cfg import NDTEnvDefaultCfg


class NDT(BaseEnv):
    cfg: NDTEnvDefaultCfg

    def __init__(self, cfg: NDTEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.success_hold_steps = int(self.cfg.success_hold_steps)
        self.success_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._goal_pos = torch.as_tensor(
            self.cfg.goal_position,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

    def _setup_scene(self):
        # These raw USDs must exist before BaseEnv clones the environment.
        self.cfg.scene_usd_cfg.func(
            "/World/envs/env_.*/Scene",
            self.cfg.scene_usd_cfg,
            translation=self.cfg.scene_translation,
        )
        self.cfg.marker_usd_cfg.func(
            "/World/envs/env_.*/Marker",
            self.cfg.marker_usd_cfg,
            translation=self.cfg.marker_translation,
            orientation=self.cfg.marker_orientation,
        )

        super()._setup_scene()

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

    def check_success(self) -> torch.Tensor:
        """Check whether the inspection target has been held long enough."""
        ee_pos = self.robot.data.body_link_state_w[:, self.ee_link_idx, 0:3] - self.scene.env_origins
        goal_pos = self._goal_pos.expand(self.num_envs, -1)

        x_dist = torch.abs(ee_pos[:, 0] - goal_pos[:, 0])
        yz_dist = torch.norm(ee_pos[:, 1:] - goal_pos[:, 1:], dim=1)
        instant_success = (x_dist < self.cfg.success_x_tolerance) & (yz_dist < self.cfg.success_yz_tolerance)

        self.success_counter.masked_fill_(~instant_success, 0)
        self.success_counter.add_(instant_success)

        return self.success_counter >= self.success_hold_steps

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return NDT task success for DirectRLEnv termination."""
        success = self.check_success()
        return success, {"inspection_target_held": success}

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""

        env_origins = self.scene.env_origins
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        goal_pos = self._goal_pos.expand(self.num_envs, -1)

        # Convert positions to environment-local frame.
        ee_pos = ee_state[:, 0:3] - env_origins
        ee_quat = ee_state[:, 3:7]
        ee_lin_vel = ee_state[:, 7:10]
        ee_ang_vel = ee_state[:, 10:13]
        base_pos = base_state[:, 0:3] - env_origins
        base_quat = base_state[:, 3:7]
        base_lin_vel = base_state[:, 7:10]
        base_ang_vel = base_state[:, 10:13]
        gripper_widths = torch.sum(gripper_state, dim=1)

        if self.robot_io.has_arm:
            arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
            arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        else:
            arm_joint_pos = None
            arm_joint_vel = None

        obs_list = []
        for env_idx in range(self.num_envs):
            env_obs = {
                "ee_pos": ee_pos[env_idx],
                "ee_quat": ee_quat[env_idx],
                "ee_lin_vel": ee_lin_vel[env_idx],
                "ee_ang_vel": ee_ang_vel[env_idx],
                "base_pos": base_pos[env_idx],
                "base_quat": base_quat[env_idx],
                "base_lin_vel": base_lin_vel[env_idx],
                "base_ang_vel": base_ang_vel[env_idx],
                "gripper_width": gripper_widths[env_idx].unsqueeze(0),
                "goal_pos": goal_pos[env_idx],
            }
            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the NDT task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.success_counter[env_ids] = 0
