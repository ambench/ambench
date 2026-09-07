# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation

from ambench.tasks.base.base_env import BaseEnv

from .open_door_env_cfg import OpenDoorEnvDefaultCfg

ENGAGEMENT_THRESHOLD_RATIO = 0.05


class OpenDoor(BaseEnv):
    cfg: OpenDoorEnvDefaultCfg

    def __init__(self, cfg: OpenDoorEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _init_ids(self):
        self.door_joint_idx = self.door.find_joints("Hinge")[0][0]

    def _setup_scene(self):
        super()._setup_scene()

        self.door = Articulation(self.cfg.door_cfg)
        self.scene.articulations["door"] = self.door

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return door-open success and subtask criteria."""
        door_joint_pos = self.door.data.joint_pos[:, self.door_joint_idx]
        door_open_rad = torch.deg2rad(torch.tensor(self.cfg.door_opened_threshold_deg, device=self.device))
        door_opened = door_joint_pos > door_open_rad
        return door_opened, {
            "door_engaged": door_joint_pos >= ENGAGEMENT_THRESHOLD_RATIO * door_open_rad,
            "door_opened": door_opened,
        }

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""

        # Get base and ee state for all environments
        # Shape: (num_envs, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]

        # Door state
        door_joint_pos = self.door.data.joint_pos[:, self.door_joint_idx].unsqueeze(1)
        door_pos = self.door.data.root_pos_w  # (num_envs, 3)

        # Precompute batched observation components
        ee_pos = ee_state[:, 0:3]
        ee_quat = ee_state[:, 3:7]
        ee_lin_vel = ee_state[:, 7:10]
        ee_ang_vel = ee_state[:, 10:13]
        base_pos = base_state[:, 0:3]
        base_quat = base_state[:, 3:7]
        base_lin_vel = base_state[:, 7:10]
        base_ang_vel = base_state[:, 10:13]
        gripper_widths = torch.sum(gripper_state, dim=1)
        # Optional conversion to local frame in a vectorized manner
        # Arm joint states for manipulator robots (if applicable)
        if self.robot_io.has_arm:
            arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
            arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        else:
            arm_joint_pos = None
            arm_joint_vel = None
        env_origins = self.scene.env_origins  # (num_envs, 3)
        ee_pos = ee_pos - env_origins
        base_pos = base_pos - env_origins
        door_pos = door_pos - env_origins

        # Build per-environment observations
        obs_list = []
        for env_idx in range(self.num_envs):
            env_obs = {
                # End-effector states
                "ee_pos": ee_pos[env_idx],
                "ee_quat": ee_quat[env_idx],
                "ee_lin_vel": ee_lin_vel[env_idx],
                "ee_ang_vel": ee_ang_vel[env_idx],
                # Base states
                "base_pos": base_pos[env_idx],
                "base_quat": base_quat[env_idx],
                "base_lin_vel": base_lin_vel[env_idx],
                "base_ang_vel": base_ang_vel[env_idx],
                # Gripper state
                "gripper_width": gripper_widths[env_idx].unsqueeze(0),
                # Environment-specific states
                "door_joint_pos": door_joint_pos[env_idx],
                "door_pos": door_pos[env_idx],
            }
            # Add arm joint states for manipulator robots
            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for open door task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset door joint position
        door_joint_pos, door_joint_vel = (
            self.door.data.default_joint_pos[env_ids].clone(),
            self.door.data.default_joint_vel[env_ids].clone(),
        )
        # Write to simulation
        self.door.write_joint_state_to_sim(door_joint_pos, door_joint_vel, None, env_ids)
