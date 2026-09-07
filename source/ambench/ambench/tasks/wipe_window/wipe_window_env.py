# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor

from ambench.tasks.base.base_env import BaseEnv

from .wipe_window_env_cfg import WipeWindowEnvDefaultCfg


class WipeWindow(BaseEnv):
    cfg: WipeWindowEnvDefaultCfg

    def __init__(self, cfg: WipeWindowEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()

        self.window_object = RigidObject(self.cfg.window_object_cfg)
        self.scene.rigid_objects["window_object"] = self.window_object

        # Create all dot objects dynamically
        self.dot_objects = []
        num_dots = self.cfg.num_dots
        for i in range(num_dots):
            dot_cfg = getattr(self.cfg, f"dot_{i}_object_cfg")
            dot_object = RigidObject(dot_cfg)
            dot_name = f"dot_{i}_object"
            self.dot_objects.append(dot_object)
            self.scene.rigid_objects[dot_name] = dot_object

        self.sponge_object = RigidObject(self.cfg.sponge_object_cfg)
        self.scene.rigid_objects["sponge_object"] = self.sponge_object

        # Add single contact sensor for detecting sponge-window contact
        self.contact_sensor = ContactSensor(cfg=self.cfg.contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # Contact tracking for stain removal
        # Track previous contact state for rising edge detection (per dot)
        num_dots = self.cfg.num_dots
        self._dot_was_in_contact = torch.zeros(self.num_envs, num_dots, dtype=torch.bool, device=self.device)
        # Count of contact events per dot
        self._dot_contact_count = torch.zeros(self.num_envs, num_dots, dtype=torch.int32, device=self.device)
        self._stain_visible_buf = torch.ones(self.num_envs, num_dots, dtype=torch.bool, device=self.device)

        # Cache batched visibility views for each dot family.
        self._dot_visibility_views = [
            sim_utils.XformPrimView(dot_object.cfg.prim_path, device=self.device) for dot_object in self.dot_objects
        ]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        actions[:, -1] = -0.6  # Keep gripper closed
        # Call parent to process actions
        super()._pre_physics_step(actions)

        # Check contact force and remove stains if force exceeds threshold
        self._remove_stains_on_contact()

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return all-stains-removed success and per-stain criteria."""
        stains_removed = {
            f"stain_{dot_idx}_removed": ~self._stain_visible_buf[:, dot_idx] for dot_idx in range(self.cfg.num_dots)
        }
        return ~self._stain_visible_buf.any(dim=1), stains_removed

    def _get_stain_state_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-dot stain positions and visibility in world frame."""
        stain_pos_w = torch.stack([dot_object.data.root_pos_w for dot_object in self.dot_objects], dim=1)
        return stain_pos_w, self._stain_visible_buf.float()

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        # End-effector and base state
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        gripper_state = self.robot.data.joint_pos[:, self.gripper_joint_ids]

        # Stain state
        stain_pos, stain_visible = self._get_stain_state_w()

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
        stain_pos = stain_pos - env_origins.unsqueeze(1)

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
                "stain_pos": stain_pos[env_idx],
                "stain_visible": stain_visible[env_idx],
            }

            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the window wiping task."""
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset environment indices."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset all dots (stains) visibility for the reset environments
        visible = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        for visibility_view in self._dot_visibility_views:
            visibility_view.set_visibility(visible, indices=env_ids)

        # Reset contact tracking buffers
        self._dot_was_in_contact[env_ids] = False
        self._dot_contact_count[env_ids] = 0
        self._stain_visible_buf[env_ids] = True

        self._grasp_rigid_object_at_ee(
            self.sponge_object,
            env_ids=env_ids,
            grasp_width=self.cfg.sponge_size[1],
        )

    def _remove_stains_on_contact(self) -> None:
        """Remove dots when sponge contact conditions are met."""
        grace_steps = 10
        grace_mask = self.episode_length_buf < grace_steps
        if grace_mask.all():
            return

        num_dots = self.cfg.num_dots
        sponge_pos = self.sponge_object.data.root_pos_w

        net_forces = self.contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        force_magnitudes = torch.norm(net_forces[:, 0, :], dim=-1)  # (num_envs,)

        # Check if force exceeds threshold and sponge is in contact with any visible dot
        has_force = (force_magnitudes > self.cfg.stain_removal_force_threshold) & (~grace_mask)
        dot_positions = torch.stack([dot_object.data.root_pos_w for dot_object in self.dot_objects], dim=1)
        distances = torch.norm(dot_positions - sponge_pos.unsqueeze(1), dim=-1)
        current_contact = has_force.unsqueeze(1) & (distances <= self.cfg.stain_removal_contact_radius)
        current_contact &= self._stain_visible_buf

        # Detect rising edges (transition from no-contact to contact)
        rising_edge = current_contact & (~self._dot_was_in_contact)

        # Increment contact count on rising edge
        self._dot_contact_count += rising_edge.int()
        self._dot_was_in_contact = current_contact

        ready_to_remove = self._dot_contact_count >= self.cfg.stain_removal_contact_count

        for dot_idx in range(num_dots):
            envs_to_remove = torch.where(ready_to_remove[:, dot_idx])[0]
            if envs_to_remove.numel() == 0:
                continue

            invisible = torch.zeros(envs_to_remove.numel(), dtype=torch.bool, device=self.device)
            self._dot_visibility_views[dot_idx].set_visibility(invisible, indices=envs_to_remove)
            self._stain_visible_buf[envs_to_remove, dot_idx] = False
