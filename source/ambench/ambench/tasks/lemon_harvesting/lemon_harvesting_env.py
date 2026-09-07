# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from pxr import UsdPhysics

from ambench.tasks.base.base_env import BaseEnv

from .lemon_harvesting_env_cfg import LemonHarvestingEnvDefaultCfg


class LemonHarvesting(BaseEnv):
    """Lemon harvesting environment."""

    cfg: LemonHarvestingEnvDefaultCfg

    def __init__(self, cfg: LemonHarvestingEnvDefaultCfg, render_mode: str | None = None, **kwargs):
        """Initialize task-local buffers derived from config values."""
        super().__init__(cfg, render_mode, **kwargs)

        # Per-env wall-local lime attachment offsets; the reset event overwrites active rows.
        n_limes = int(self.cfg.max_limes_in_scene)
        self._lime_wall_local_pos = torch.zeros(self.num_envs, n_limes, 3, device=self.device, dtype=torch.float32)

        # Cache container goal-region bounds.
        container_size_x, container_size_y, container_size_z = self.cfg.container_size
        self._container_goal_region_min_offset = torch.tensor(
            (-0.5 * container_size_x, -0.5 * container_size_y, 0.0),
            device=self.device,
            dtype=torch.float32,
        )
        self._container_goal_region_max_offset = torch.tensor(
            (0.5 * container_size_x, 0.5 * container_size_y, container_size_z),
            device=self.device,
            dtype=torch.float32,
        )
        self._lemon_grasp_anchor_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._lemon_grasped_and_detached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._lemon_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._gripper_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

    def _setup_scene(self):
        super()._setup_scene()

        self.wall = RigidObject(self.cfg.wall_cfg)
        self.scene.rigid_objects["wall"] = self.wall

        self.lemon_object = RigidObject(self.cfg.lemon_object_cfg)
        self.scene.rigid_objects["lemon_object"] = self.lemon_object

        self.lime_objects = [RigidObject(cfg) for cfg in self.cfg.lime_object_cfgs]
        for i, lime_obj in enumerate(self.lime_objects, start=1):
            self.scene.rigid_objects[f"lime_object_{i:02d}"] = lime_obj

        self.table_object = RigidObject(self.cfg.table_object_cfg)
        self.scene.rigid_objects["table"] = self.table_object

        self.container = RigidObject(self.cfg.container_object_cfg)
        self.scene.rigid_objects["container"] = self.container

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return lemon-delivery success and subtask criteria."""
        in_goal_region = self._is_lemon_in_container()
        gripper_width = self.robot.data.joint_pos[:, self.gripper_joint_ids].sum(dim=1)
        gripper_open = gripper_width > self.cfg.gripper_open_threshold
        lemon_delivered_released = in_goal_region & gripper_open
        return lemon_delivered_released, {
            "lemon_grasped_and_detached": self._update_lemon_grasped_and_detached(),
            "lemon_delivered_released": lemon_delivered_released,
        }

    def _is_lemon_in_container(self) -> torch.Tensor:
        lemon_pos = self.lemon_object.data.root_pos_w
        container_pos = self.container.data.root_pos_w
        goal_region_min = container_pos + self._container_goal_region_min_offset
        goal_region_max = container_pos + self._container_goal_region_max_offset
        return ((lemon_pos >= goal_region_min) & (lemon_pos <= goal_region_max)).all(dim=1)

    def _update_lemon_grasped_and_detached(self) -> torch.Tensor:
        lemon_pos = self.lemon_object.data.root_pos_w
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        gripper_tip_pos = self.robot_io.compute_tool_tip_pos_w(ee_state)
        gripper_width = self.robot.data.joint_pos[:, self.gripper_joint_ids].sum(dim=1)

        gripper_closed = gripper_width < self.cfg.gripper_closed_threshold
        lemon_near_gripper = torch.norm(lemon_pos - gripper_tip_pos, dim=-1) < self.cfg.grasp_distance_threshold
        can_anchor = gripper_closed & lemon_near_gripper & (~self._lemon_grasped_and_detached)

        new_anchor = can_anchor & (~self._lemon_grasp_anchor_active)
        self._lemon_anchor_pos_w[new_anchor] = lemon_pos[new_anchor]
        self._gripper_anchor_pos_w[new_anchor] = gripper_tip_pos[new_anchor]
        self._lemon_grasp_anchor_active |= new_anchor

        lost_anchor = self._lemon_grasp_anchor_active & (~can_anchor) & (~self._lemon_grasped_and_detached)
        self._lemon_grasp_anchor_active[lost_anchor] = False

        lemon_delta = lemon_pos - self._lemon_anchor_pos_w
        gripper_delta = gripper_tip_pos - self._gripper_anchor_pos_w
        lemon_distance = torch.norm(lemon_delta, dim=-1)
        gripper_distance = torch.norm(gripper_delta, dim=-1)
        cosine = torch.sum(lemon_delta * gripper_delta, dim=-1) / (lemon_distance * gripper_distance).clamp_min(1.0e-6)
        moved_together = (
            self._lemon_grasp_anchor_active
            & gripper_closed
            & lemon_near_gripper
            & (lemon_distance >= self.cfg.carry_distance_threshold)
            & (gripper_distance >= self.cfg.carry_distance_threshold)
            & (cosine >= self.cfg.grasp_direction_cosine_threshold)
        )
        self._lemon_grasped_and_detached |= moved_together
        return self._lemon_grasped_and_detached

    def _get_observations(self) -> dict:
        """Compute per-environment policy observations."""
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        base_state = self.robot.data.body_link_state_w[:, self.base_link_idx, :]
        lemon_pos = self.lemon_object.data.root_pos_w
        container_pos = self.container.data.root_pos_w

        ee_pos = ee_state[:, 0:3]
        ee_quat = ee_state[:, 3:7]
        ee_lin_vel = ee_state[:, 7:10]
        ee_ang_vel = ee_state[:, 10:13]
        base_pos = base_state[:, 0:3]
        base_quat = base_state[:, 3:7]
        base_lin_vel = base_state[:, 7:10]
        base_ang_vel = base_state[:, 10:13]
        gripper_widths = self.robot.data.joint_pos[:, self.gripper_joint_ids].sum(dim=1)

        if self.robot_io.has_arm:
            arm_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
            arm_joint_vel = self.robot.data.joint_vel[:, self.arm_joint_ids]
        else:
            arm_joint_pos = None
            arm_joint_vel = None

        # Convert task assets into the environment-local frame.
        env_origins = self.scene.env_origins
        ee_pos = ee_pos - env_origins
        base_pos = base_pos - env_origins
        lemon_pos = lemon_pos - env_origins
        container_pos = container_pos - env_origins

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
                "lemon_pos": lemon_pos[env_idx],
                "container_pos": container_pos[env_idx],
            }
            if arm_joint_pos is not None:
                env_obs["arm_joint_pos"] = arm_joint_pos[env_idx]
            if arm_joint_vel is not None:
                env_obs["arm_joint_vel"] = arm_joint_vel[env_idx]

            obs_list.append(env_obs)

        return {"policy": obs_list}

    def _get_rewards(self) -> torch.Tensor:
        """Use sparse task completion only."""
        return torch.zeros((self.num_envs,), device=self.device)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset task assets while preserving event-randomized wall attachments."""
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        num_env = env_ids.numel()
        env_origins = self.scene.env_origins[env_ids]
        zero_velocity = torch.zeros((num_env, 6), device=self.device, dtype=torch.float32)

        # Restore the lemon pose from the wall-local fixed-joint attachment sampled during reset.
        wall_pos_base = torch.as_tensor(self.cfg.wall_position, device=self.device, dtype=torch.float32)
        wall_quat_base = torch.as_tensor(self.cfg.wall_rotation, device=self.device, dtype=torch.float32)
        wall_pos = wall_pos_base.view(1, 3) + env_origins
        wall_quat = wall_quat_base.view(1, 4).repeat(num_env, 1)
        lemon_wall_local_pos = torch.zeros((num_env, 3), device=self.device, dtype=torch.float32)
        for row, env_index in enumerate(env_ids.tolist()):
            joint_prim = self.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}/lemon_wall_fixed_joint")
            if not joint_prim.IsValid():
                raise RuntimeError(f"Missing fixed joint 'lemon_wall_fixed_joint' under '/World/envs/env_{env_index}'.")

            joint = UsdPhysics.FixedJoint(joint_prim)
            joint_local_pos = joint.GetLocalPos1Attr().Get()
            lemon_wall_local_pos[row] = torch.tensor(
                [float(joint_local_pos[0]), float(joint_local_pos[1]), float(joint_local_pos[2])],
                device=self.device,
                dtype=torch.float32,
            )
        lemon_pose = torch.cat([wall_pos - math_utils.quat_apply(wall_quat, lemon_wall_local_pos), wall_quat], dim=-1)
        self.lemon_object.write_root_pose_to_sim(lemon_pose, env_ids)
        self.lemon_object.write_root_velocity_to_sim(zero_velocity, env_ids)
        self._lemon_grasp_anchor_active[env_ids] = False
        self._lemon_grasped_and_detached[env_ids] = False
        self._lemon_anchor_pos_w[env_ids] = lemon_pose[:, :3]
        self._gripper_anchor_pos_w[env_ids] = 0.0

        # Restore kinematic limes from the wall-local buffer written by the reset event.
        lime_wall_local_pos = self._lime_wall_local_pos[env_ids]
        for lime_index, lime_obj in enumerate(self.lime_objects):
            local_pos = lime_wall_local_pos[:, lime_index, :]
            pose = torch.cat([wall_pos - math_utils.quat_apply(wall_quat, local_pos), wall_quat], dim=-1)
            lime_obj.write_root_pose_to_sim(pose, env_ids)
            lime_obj.write_root_velocity_to_sim(zero_velocity, env_ids)

        # Static task assets return to their default root state every reset.
        for asset in (self.container, self.table_object):
            default_root_state = asset.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += env_origins
            asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
