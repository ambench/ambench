# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted policy for toss-ball task."""

import isaaclab.utils.math as math_utils
import torch

from .base import BasePolicy, Waypoint


class TossBallPolicy(BasePolicy):
    """Scripted policy for toss-ball task.

    The trajectory is box-aware:
    1. Tilt down in place to stabilize the camera/viewing direction
    2. Align laterally with the target box in y
    3. Move forward toward the target x position
    4. Execute a softened throwing motion with a short closed pre-throw
    """

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate a target-aware tossing trajectory."""
        obs_dict = obs["policy"][env_id]
        target_pos = obs_dict["target_pos"]
        init_ee_pose = self._ee_pos_init_local[env_id]

        def uniform_offset(
            x_range: tuple[float, float] = (0.0, 0.0),
            y_range: tuple[float, float] = (0.0, 0.0),
            z_range: tuple[float, float] = (0.0, 0.0),
        ) -> torch.Tensor:
            return torch.tensor(
                [
                    math_utils.sample_uniform(*x_range, size=1, device=self.device).item(),
                    math_utils.sample_uniform(*y_range, size=1, device=self.device).item(),
                    math_utils.sample_uniform(*z_range, size=1, device=self.device).item(),
                ],
                dtype=init_ee_pose.dtype,
                device=self.device,
            )

        identity_quat = self.identity_quat()

        release_pitch_delta = math_utils.sample_uniform(lower=-0.04, upper=0.04, size=1, device=self.device).item()
        load_pitch_delta = math_utils.sample_uniform(lower=-0.05, upper=0.05, size=1, device=self.device).item()

        pitch_down_move_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([torch.pi / 5.0], device=self.device),
            torch.tensor([0.0], device=self.device),
        )[0]

        pitch_down_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([torch.pi / 3.0 + load_pitch_delta], device=self.device),
            torch.tensor([0.0], device=self.device),
        )[0]

        pitch_down_release_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([0.0 + release_pitch_delta], device=self.device),
            torch.tensor([0.0], device=self.device),
        )[0]

        # Stage 1: tilt down in place before translating.
        tilt_down_pose = init_ee_pose.clone()
        tilt_down_pose += uniform_offset(
            x_range=(-0.03, 0.03),
            y_range=(-0.03, 0.03),
            z_range=(-0.02, 0.02),
        )

        # Stage 2: align laterally with the box while staying near the initial x position.
        align_y_pose = init_ee_pose.clone()
        align_y_pose[1] = (
            target_pos[1] + math_utils.sample_uniform(lower=-0.06, upper=0.06, size=1, device=self.device).item()
        )
        align_y_pose[0] += math_utils.sample_uniform(lower=-0.05, upper=0.10, size=1, device=self.device).item()
        align_y_pose[2] += math_utils.sample_uniform(lower=-0.03, upper=0.03, size=1, device=self.device).item()

        # Stage 3: move forward toward the box before the throw sequence starts.
        forward_pose = align_y_pose.clone()
        forward_pose[0] = (
            init_ee_pose[0].item()
            + math_utils.sample_uniform(lower=0.25, upper=0.45, size=1, device=self.device).item()
        )
        forward_pose[2] += math_utils.sample_uniform(lower=-0.03, upper=0.04, size=1, device=self.device).item()

        # Throwing waypoints retain the successful original geometry, but spread the release over more frames.
        wp_throw_load = align_y_pose.clone()
        wp_throw_load[0] = (
            forward_pose[0].item()
            + math_utils.sample_uniform(lower=0.05, upper=0.10, size=1, device=self.device).item()
        )
        wp_throw_load[2] = (
            init_ee_pose[2] + math_utils.sample_uniform(lower=-0.16, upper=-0.08, size=1, device=self.device).item()
        )

        wp_throw_hold = wp_throw_load.clone()

        wp_release = align_y_pose.clone()
        wp_release[0] = (
            wp_throw_load[0].item()
            + math_utils.sample_uniform(lower=0.64, upper=0.84, size=1, device=self.device).item()
        )
        wp_release[2] = (
            init_ee_pose[2] + math_utils.sample_uniform(lower=0.08, upper=0.18, size=1, device=self.device).item()
        )

        wp_pre_release = wp_throw_hold + 0.42 * (wp_release - wp_throw_hold)

        wp_follow_through = align_y_pose.clone()
        wp_follow_through[0] = wp_release[0].item()
        wp_follow_through[2] = (
            init_ee_pose[2] + math_utils.sample_uniform(lower=-0.14, upper=-0.06, size=1, device=self.device).item()
        )

        t_tilt = int(math_utils.sample_uniform(lower=100, upper=140, size=1, device=self.device).item())
        t_align = int(math_utils.sample_uniform(lower=230, upper=290, size=1, device=self.device).item())
        t_forward = int(math_utils.sample_uniform(lower=400, upper=410, size=1, device=self.device).item())
        t_load = int(math_utils.sample_uniform(lower=420, upper=440, size=1, device=self.device).item())
        t_hold = t_load + int(math_utils.sample_uniform(lower=80, upper=120, size=1, device=self.device).item())
        t_pre_release = t_hold + int(math_utils.sample_uniform(lower=8, upper=12, size=1, device=self.device).item())
        t_release = t_pre_release + int(math_utils.sample_uniform(lower=9, upper=12, size=1, device=self.device).item())
        t_follow = 800

        # Gripper convention in this repo:
        # 1.0 = open, -1.0 = close
        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=t_tilt, xyz=tilt_down_pose, quat=pitch_down_move_quat, gripper=-1.0),
            Waypoint(t=t_align, xyz=align_y_pose, quat=pitch_down_move_quat, gripper=-1.0),
            Waypoint(t=t_forward, xyz=forward_pose, quat=pitch_down_move_quat, gripper=-1.0),
            Waypoint(t=t_load, xyz=wp_throw_load, quat=pitch_down_quat, gripper=-1.0),
            Waypoint(t=t_hold, xyz=wp_throw_hold, quat=pitch_down_quat, gripper=-1.0),
            Waypoint(t=t_pre_release, xyz=wp_pre_release, quat=pitch_down_release_quat, gripper=-1.0),
            Waypoint(t=t_release, xyz=wp_release, quat=pitch_down_release_quat, gripper=1.0),
            Waypoint(t=t_follow, xyz=wp_follow_through, quat=pitch_down_quat, gripper=1.0),
        ]

        self._log_trajectory_summary(f"target_pos={target_pos}")
