# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .frame_assembly_env_cfg import FrameAssemblyEnvDefaultCfg

FRAME_LIFT_HEIGHT_THRESHOLD = 0.05


class FrameAssembly(BaseEnv):
    cfg: FrameAssemblyEnvDefaultCfg

    def __init__(self, cfg: FrameAssemblyEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._frame_reset_z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _setup_scene(self):
        super()._setup_scene()

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

        # Add frame object to the scene
        self.frame_object = RigidObject(self.cfg.frame_object_cfg)
        self.scene.rigid_objects["frame_object"] = self.frame_object

        # Add pegs to the scene (kinematic, attached to wall)
        self.pegs = []
        for i, peg_cfg in enumerate(self.cfg.peg_cfgs, start=1):
            peg = RigidObject(peg_cfg)
            self.scene.rigid_objects[f"peg_{i}"] = peg
            self.pegs.append(peg)

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return frame-placement success and subtask criteria."""
        frame_pos = self.frame_object.data.root_pos_w
        peg_center = self._compute_peg_center()

        distance_x = torch.abs(frame_pos[:, 0] - peg_center[:, 0])
        distance_yz = torch.sqrt((frame_pos[:, 1] - peg_center[:, 1]) ** 2 + (frame_pos[:, 2] - peg_center[:, 2]) ** 2)
        frame_placed_on_pegs = (distance_x < self.cfg.frame_placement_tolerance_x) & (
            distance_yz < self.cfg.frame_placement_tolerance_yz
        )
        frame_lifted = frame_pos[:, 2] > self._frame_reset_z + FRAME_LIFT_HEIGHT_THRESHOLD
        return frame_placed_on_pegs, {
            "frame_lifted": frame_lifted,
            "frame_placed_on_pegs": frame_placed_on_pegs,
        }

    def _compute_peg_center(self) -> torch.Tensor:
        """Compute peg-center position for each environment."""
        peg_positions = torch.stack([peg.data.root_pos_w for peg in self.pegs], dim=0)
        return peg_positions.mean(dim=0)

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

        # Frame and peg state
        frame_pos = self.frame_object.data.root_pos_w
        peg_center_pos = self._compute_peg_center()

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
        frame_pos = frame_pos - env_origins
        peg_center_pos = peg_center_pos - env_origins

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
                "frame_pos": frame_pos[env_idx],
                "peg_center_pos": peg_center_pos[env_idx],
            }

            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for frame assembly task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Preserve event-randomized pose and only clear velocity.
        frame_state = self.frame_object.data.root_state_w[env_ids].clone()
        self._frame_reset_z[env_ids] = frame_state[:, 2]
        frame_state[:, 7:] = 0.0
        self.frame_object.write_root_state_to_sim(frame_state, env_ids=env_ids)
