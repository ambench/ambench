# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.utils.math as math_utils
import torch

from .base import BasePolicy, Waypoint


class LemonHarvestingPolicy(BasePolicy):
    """Scripted policy for lemon pick and place task."""

    fly_to_lemon_time: int | None = 400
    hover_near_lemon_rand_time = 550
    hover_near_lemon_time = 600
    grasp_time = 650
    pull_pose_time: int | None = None
    pull_farther_time = 1200
    tilt_time = 1250
    align_time = 1470
    above_container_time = 1570
    release_time = 1620
    maintain_time = 1660

    hover_near_lemon_rand_y_range = (-0.01, 0.01)
    hover_near_lemon_rand_z_range = (0.025, 0.045)
    pull_farther_x_range = (0.40, 0.50)
    pull_farther_y_range = (0.0, 0.30)
    tilt_pitch_rad = 0.30
    align_z_range = (0.30, 0.33)

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for lemon pick and place task.

        Args:
            obs: Observation dict with "policy" key containing tensor of shape (1, 29)
        """
        # Extract observation dictionary for first environment
        obs_dict = obs["policy"][env_id]

        # Parse observation (keep as tensors on correct device)
        lemon_pos = obs_dict["lemon_pos"]
        container_pos = obs_dict["container_pos"]

        # Use initial ee_pos as starting point
        init_ee_pose = self._ee_pos_init_local[env_id]

        # Identity quaternion
        identity_quat = self.identity_quat()

        # Fly to lemon
        fly_to_lemon = lemon_pos.clone()
        fly_to_lemon[0] -= math_utils.sample_uniform(lower=0.25, upper=0.35, size=1, device=self.device).item()
        # fly_to_lemon[2] += 0.035
        fly_to_lemon[1] += math_utils.sample_uniform(lower=-0.05, upper=0.05, size=1, device=self.device).item()
        fly_to_lemon[2] += math_utils.sample_uniform(lower=-0.05, upper=0.05, size=1, device=self.device).item()

        # Hover near the lemon but with some randomization
        hover_near_lemon_rand = lemon_pos.clone()
        hover_near_lemon_rand[0] -= 0.17
        hover_near_lemon_rand[0] += math_utils.sample_uniform(lower=0.0, upper=0.01, size=1, device=self.device).item()
        hover_near_lemon_rand[1] += math_utils.sample_uniform(
            lower=self.hover_near_lemon_rand_y_range[0],
            upper=self.hover_near_lemon_rand_y_range[1],
            size=1,
            device=self.device,
        ).item()
        hover_near_lemon_rand[2] += math_utils.sample_uniform(
            lower=self.hover_near_lemon_rand_z_range[0],
            upper=self.hover_near_lemon_rand_z_range[1],
            size=1,
            device=self.device,
        ).item()

        # Hover near the lemon
        hover_near_lemon = lemon_pos.clone()
        hover_near_lemon[0] -= 0.17
        # hover_near_lemon[2] += 0.035
        hover_near_lemon[1] += math_utils.sample_uniform(lower=-0.005, upper=0.005, size=1, device=self.device).item()
        hover_near_lemon[2] += math_utils.sample_uniform(lower=0.03, upper=0.04, size=1, device=self.device).item()

        # Pull backward
        pull_pose = hover_near_lemon.clone()
        pull_pose[0] -= 0.50

        # Pull backward furthermore to try to see the bin
        pull_farther_pose_x = (
            pull_pose[0]
            - math_utils.sample_uniform(
                lower=self.pull_farther_x_range[0],
                upper=self.pull_farther_x_range[1],
                size=1,
                device=self.device,
            ).item()
        )
        pull_farther_pose_y = math_utils.sample_uniform(
            lower=self.pull_farther_y_range[0],
            upper=self.pull_farther_y_range[1],
            size=1,
            device=self.device,
        ).item()
        pull_farther_pose_z = (
            math_utils.sample_uniform(lower=0.70, upper=0.80, size=1, device=self.device).item() * pull_pose[2]
        )
        pull_farther_pose = torch.tensor(
            [pull_farther_pose_x, pull_farther_pose_y, pull_farther_pose_z],
            dtype=torch.float32,
            device=self.device,
        )

        # Tilt down for visibility
        pitch_down_rad = torch.tensor(self.tilt_pitch_rad, device=self.device)
        tilt_quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            pitch_down_rad.unsqueeze(0),
            torch.tensor([0.0], device=self.device),
        )[0]

        # Align in y as the container
        align_y_pos = container_pos.clone()
        coeff = math_utils.sample_uniform(lower=0.30, upper=0.60, size=1, device=self.device).item()
        align_y_pos[0] = pull_farther_pose[0] + coeff * (container_pos[0] - pull_farther_pose[0])
        align_y_pos[2] += math_utils.sample_uniform(
            lower=self.align_z_range[0],
            upper=self.align_z_range[1],
            size=1,
            device=self.device,
        ).item()

        # Move above container
        above_container = container_pos.clone()
        above_container[2] = align_y_pos[2]
        above_container[0] -= 0.20

        # Define waypoints
        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
        ]
        if self.fly_to_lemon_time is not None:
            self.waypoints.append(Waypoint(t=self.fly_to_lemon_time, xyz=fly_to_lemon, quat=identity_quat, gripper=1.0))
        self.waypoints.extend([
            Waypoint(t=self.hover_near_lemon_rand_time, xyz=hover_near_lemon_rand, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.hover_near_lemon_time, xyz=hover_near_lemon, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.grasp_time, xyz=hover_near_lemon, quat=identity_quat, gripper=-1.0),
        ])
        if self.pull_pose_time is not None:
            self.waypoints.append(Waypoint(t=self.pull_pose_time, xyz=pull_pose, quat=identity_quat, gripper=-1.0))
        self.waypoints.extend([
            Waypoint(t=self.pull_farther_time, xyz=pull_farther_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=self.tilt_time, xyz=pull_farther_pose, quat=tilt_quat, gripper=-1.0),
            Waypoint(t=self.align_time, xyz=align_y_pos, quat=tilt_quat, gripper=-1.0),
            Waypoint(t=self.above_container_time, xyz=above_container, quat=tilt_quat, gripper=-1.0),
            Waypoint(t=self.release_time, xyz=above_container, quat=tilt_quat, gripper=1.0),
            Waypoint(t=self.maintain_time, xyz=above_container, quat=tilt_quat, gripper=1.0),
        ])

        self._log_trajectory_summary(f"lemon_pos={lemon_pos}", f"container_pos={container_pos}")


class LemonHarvestingPolicyFast(LemonHarvestingPolicy):
    """Scripted policy for lemon pick and place task."""

    fly_to_lemon_time = None
    hover_near_lemon_rand_time = 200
    hover_near_lemon_time = 230
    grasp_time = 240
    pull_pose_time = 340
    pull_farther_time = 400
    tilt_time = 420
    align_time = 470
    above_container_time = 560
    release_time = 590
    maintain_time = 600

    hover_near_lemon_rand_y_range = (-0.02, 0.02)
    hover_near_lemon_rand_z_range = (0.02, 0.05)
    pull_farther_x_range = (0.40, 0.60)
    pull_farther_y_range = (-0.30, 0.30)
    tilt_pitch_rad = 0.50
    align_z_range = (0.35, 0.40)
