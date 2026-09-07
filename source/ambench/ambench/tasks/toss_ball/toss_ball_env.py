# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .toss_ball_env_cfg import TossBallEnvDefaultCfg


class TossBall(BaseEnv):
    """Direct RL toss-ball task."""

    cfg: TossBallEnvDefaultCfg

    def __init__(self, cfg: TossBallEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

        self.container = RigidObject(self.cfg.container_object_cfg)
        self.scene.rigid_objects["container"] = self.container

        self.ball = RigidObject(self.cfg.ball_cfg)
        self.scene.rigid_objects["ball"] = self.ball

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ball-delivery success and subtask criteria."""
        ball_released, ball_entered_bin = self._compute_ball_criteria()
        robot_base_pos = self.robot.data.body_link_state_w[:, self.base_link_idx, 0:3]
        container_pos = self.container.data.root_pos_w

        robot_behind_bin = robot_base_pos[:, 0] <= (container_pos[:, 0] - self.cfg.success_robot_behind_bin_threshold)
        ball_delivered = ball_released & ball_entered_bin & robot_behind_bin
        return ball_delivered, {
            "ball_released": ball_released,
            "ball_delivered": ball_delivered,
        }

    def _compute_ball_criteria(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_pos = self.ball.data.root_pos_w
        container_pos = self.container.data.root_pos_w
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]

        gripper_tip_pos = self.robot_io.compute_tool_tip_pos_w(ee_state)
        released = torch.norm(ball_pos - gripper_tip_pos, dim=-1) >= self.cfg.release_distance_threshold

        container_size = torch.tensor(self.cfg.container_size, device=self.device, dtype=ball_pos.dtype)
        goal_region_min = container_pos - torch.tensor(
            [container_size[0] / 2, container_size[1] / 2, 0.0], device=self.device
        )
        goal_region_max = container_pos + torch.tensor(
            [container_size[0] / 2, container_size[1] / 2, container_size[2]], device=self.device
        )
        in_x = (ball_pos[:, 0] >= goal_region_min[:, 0]) & (ball_pos[:, 0] <= goal_region_max[:, 0])
        in_y = (ball_pos[:, 1] >= goal_region_min[:, 1]) & (ball_pos[:, 1] <= goal_region_max[:, 1])
        in_z = (ball_pos[:, 2] >= goal_region_min[:, 2]) & (ball_pos[:, 2] <= goal_region_max[:, 2])
        return released, in_x & in_y & in_z

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""

        # End-effector and base state
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]

        ee_pos = ee_state[:, 0:3]
        ee_quat = ee_state[:, 3:7]
        ee_lin_vel = ee_state[:, 7:10]
        ee_ang_vel = ee_state[:, 10:13]
        base_pos = base_state[:, 0:3]
        base_quat = base_state[:, 3:7]
        base_lin_vel = base_state[:, 7:10]
        base_ang_vel = base_state[:, 10:13]
        gripper_widths = torch.sum(gripper_state, dim=1)

        # Ball and target state
        ball_pos = self.ball.data.root_pos_w
        ball_lin_vel = self.ball.data.root_lin_vel_w
        target_pos = self.container.data.root_pos_w

        if self.robot_io.has_arm:
            arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
            arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        else:
            arm_joint_pos = None
            arm_joint_vel = None

        # Convert positions to environment-local frame
        env_origins = self.scene.env_origins
        ee_pos = ee_pos - env_origins
        base_pos = base_pos - env_origins
        ball_pos = ball_pos - env_origins
        target_pos = target_pos - env_origins

        # Build per-environment observations
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
                "ball_pos": ball_pos[env_idx],
                "ball_lin_vel": ball_lin_vel[env_idx],
                "target_pos": target_pos[env_idx],
            }

            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for toss-ball task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._grasp_rigid_object_at_ee(
            self.ball,
            env_ids=env_ids,
            grasp_width=2.0 * self.cfg.ball_radius,
        )
