# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .cabinet_pick_place_env_cfg import CabinetPickPlaceEnvDefaultCfg

CAN_LIFT_HEIGHT_THRESHOLD = 0.05
CABINET_PULLING_DOOR_OPEN_THRESHOLD = 0.2


class CabinetPickPlace(BaseEnv):
    cfg: CabinetPickPlaceEnvDefaultCfg

    def __init__(self, cfg: CabinetPickPlaceEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._can_reset_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _init_ids(self):
        self.pulling_door_joint_idx = self.pulling_door.find_joints("pull_door_slide")[0][0]

    def _setup_scene(self):
        super()._setup_scene()

        # Instantiate assets
        self.pulling_door = Articulation(self.cfg.pulling_door_cfg)
        self.scene.articulations["pulling_door"] = self.pulling_door
        self.can = RigidObject(self.cfg.can_cfg)
        self.scene.rigid_objects["can"] = self.can

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return can-placement success and subtask criteria."""
        can_pos = self.can.data.root_pos_w

        drawer_top_z = self.scene.env_origins[:, 2] + self.cfg.drawer_top_z

        can_height_above_drawer = can_pos[:, 2] - drawer_top_z
        can_on_drawer_z = (can_height_above_drawer >= self.cfg.can_place_z_min) & (
            can_height_above_drawer < self.cfg.can_place_z_max
        )

        can_vel = self.can.data.root_lin_vel_w
        can_velocity_magnitude = torch.norm(can_vel, dim=-1)
        can_stable = can_velocity_magnitude < self.cfg.can_place_velocity_threshold

        can_placed_on_drawer = can_on_drawer_z & can_stable
        pulling_door_joint_pos = self.pulling_door.data.joint_pos[:, self.pulling_door_joint_idx]
        pulling_door_closed_pos = self.pulling_door.data.default_joint_pos[:, self.pulling_door_joint_idx]
        pulling_door_open_distance = torch.abs(pulling_door_joint_pos - pulling_door_closed_pos)
        return can_placed_on_drawer, {
            "pulling_door_opened": pulling_door_open_distance >= CABINET_PULLING_DOOR_OPEN_THRESHOLD,
            "can_lifted": can_pos[:, 2] > self._can_reset_z + CAN_LIFT_HEIGHT_THRESHOLD,
            "can_placed_on_drawer": can_placed_on_drawer,
        }

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        # End-effector and base state.
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]

        # Task-specific state.
        pulling_door_pos = self.pulling_door.data.root_pos_w
        can_pos = self.can.data.root_pos_w

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

        # Convert positions to environment-local frame.
        env_origins = self.scene.env_origins
        ee_pos = ee_pos - env_origins
        base_pos = base_pos - env_origins
        pulling_door_pos = pulling_door_pos - env_origins
        can_pos = can_pos - env_origins

        # Build per-environment observations.
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
                "gripper_width": gripper_widths[env_idx : env_idx + 1],
                "pulling_door_pos": pulling_door_pos[env_idx],
                "can_pos": can_pos[env_idx],
            }
            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute task rewards."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Clear cabinet articulation motion and restore nominal joint states.
        zero_lin_vel = torch.zeros((len(env_ids), 3), device=self.device)
        zero_ang_vel = torch.zeros((len(env_ids), 3), device=self.device)
        zero_root_vel = torch.cat([zero_lin_vel, zero_ang_vel], dim=-1)
        pulling_door_joint_pos = self.pulling_door.data.default_joint_pos[env_ids].clone()
        zero_pulling_door_joint_vel = torch.zeros_like(self.pulling_door.data.default_joint_vel[env_ids])

        self.pulling_door.write_root_velocity_to_sim(zero_root_vel, env_ids)
        self.pulling_door.write_joint_state_to_sim(pulling_door_joint_pos, zero_pulling_door_joint_vel, None, env_ids)

        can_state = self.can.data.root_state_w[env_ids].clone()
        self._can_reset_z[env_ids] = can_state[:, 2]
