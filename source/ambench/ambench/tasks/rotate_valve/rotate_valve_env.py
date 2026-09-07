# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .rotate_valve_env_cfg import RotateValveEnvDefaultCfg


class RotateValve(BaseEnv):
    cfg: RotateValveEnvDefaultCfg

    def _init_ids(self):
        self.valve_handle_idx = self.valve_object.find_bodies("handle_link")[0][0]
        self.valve_joint_idx = self.valve_object.find_joints("base_to_shaft")[0][0]

    def _setup_scene(self):
        super()._setup_scene()

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

        self.valve_object = Articulation(self.cfg.valve_object_cfg)
        self.scene.articulations["valve_object"] = self.valve_object

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return valve-rotation success and subtask criteria."""
        valve_joint_angle = self.valve_object.data.joint_pos[:, self.valve_joint_idx]
        valve_rotated = valve_joint_angle >= self.cfg.valve_target_angle
        return valve_rotated, {
            "valve_engaged": valve_joint_angle >= self.cfg.valve_engagement_ratio * self.cfg.valve_target_angle,
            "valve_rotated": valve_rotated,
        }

    def _get_observations(self) -> dict:
        """Compute per-environment policy observations."""
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]
        goal_pos = self.valve_object.data.body_pos_w[:, self.valve_handle_idx]

        ee_pos = ee_state[:, 0:3]
        ee_quat = ee_state[:, 3:7]
        ee_lin_vel = ee_state[:, 7:10]
        ee_ang_vel = ee_state[:, 10:13]
        base_pos = base_state[:, 0:3]
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
        # Convert positions to the environment-local frame.
        env_origins = self.scene.env_origins
        ee_pos = ee_pos - env_origins
        base_pos = base_pos - env_origins
        goal_pos = goal_pos - env_origins

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
        """Compute rewards for rotate valve task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset the base environment and restore the valve joint state."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        valve_joint_pos, valve_joint_vel = (
            self.valve_object.data.default_joint_pos[env_ids].clone(),
            self.valve_object.data.default_joint_vel[env_ids].clone(),
        )
        self.valve_object.write_joint_state_to_sim(valve_joint_pos, valve_joint_vel, None, env_ids)
