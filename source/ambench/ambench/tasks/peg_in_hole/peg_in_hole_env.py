# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject

from ambench.tasks.base.base_env import BaseEnv

from .peg_in_hole_env_cfg import PegInHoleEnvDefaultCfg


class PegInHole(BaseEnv):
    cfg: PegInHoleEnvDefaultCfg

    def __init__(self, cfg: PegInHoleEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        self.hole_side_length_buf = torch.full(
            (cfg.scene.num_envs,),
            float(cfg.hole_side_length),
            device=cfg.sim.device,
            dtype=torch.float32,
        )
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

        self.hole_object = RigidObject(self.cfg.hole_object_cfg)
        self.scene.rigid_objects["hole_object"] = self.hole_object

        self.peg_object = RigidObject(self.cfg.peg_object_cfg)
        self.scene.rigid_objects["peg_object"] = self.peg_object

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return peg-in-hole success."""
        peg_pos = self.peg_object.data.root_pos_w
        peg_quat = self.peg_object.data.root_quat_w

        peg_half_length = self.cfg.peg_length * 0.5
        peg_tip_offset_local = torch.tensor([peg_half_length, 0.0, 0.0], device=self.device)
        peg_tip_offset_world = math_utils.quat_apply(
            peg_quat, peg_tip_offset_local.unsqueeze(0).expand(self.num_envs, -1)
        )
        peg_tip_pos = peg_pos + peg_tip_offset_world

        # Get hole pose in world frame
        hole_pos = self.hole_object.data.root_pos_w  # (num_envs, 3)
        hole_quat = self.hole_object.data.root_quat_w  # (num_envs, 4)

        # Transform peg tip to hole's local frame
        relative_pos = peg_tip_pos - hole_pos  # Vector from hole center to peg tip

        # Rotate relative position to hole's local frame
        peg_tip_local = math_utils.quat_apply_inverse(hole_quat, relative_pos)

        # Check 1: Depth (X-axis in hole's local frame)
        # Peg tip must be past the hole surface by at least insertion_depth_min
        depth_check = peg_tip_local[:, 0] > self.cfg.peg_insertion_depth_min

        # Check 2: Lateral boundaries (Y-Z plane in hole's local frame)
        half_side_length = 0.5 * self.hole_side_length_buf.to(dtype=peg_tip_local.dtype)
        y_check = torch.abs(peg_tip_local[:, 1]) < half_side_length
        z_check = torch.abs(peg_tip_local[:, 2]) < half_side_length
        lateral_check = y_check & z_check

        return depth_check & lateral_check, {}

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        # Get base and ee state for all environments
        # Shape: (num_envs, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]

        # Hole state
        goal_pos = self.hole_object.data.root_pos_w  # (num_envs, 3)

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
        goal_pos = goal_pos - env_origins

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
                "goal_pos": goal_pos[env_idx],
            }
            # Add arm joint states for manipulator robots
            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for peg insertion task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self._grasp_rigid_object_at_ee(
            self.peg_object,
            env_ids=env_ids,
            grasp_width=2.0 * self.cfg.peg_radius,
        )
