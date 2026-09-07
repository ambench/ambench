# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from ..utils import compute_wall_aligned_quat, compute_wall_axes
from .base import BasePolicy, Waypoint


class PegInHolePolicy(BasePolicy):
    """Scripted policy for peg-in-hole task.

    Observation structure (nested dict):
    - ee_state: dict with end-effector state (pos, quat, vel, ang_vel)
    - goal_pos: position of the hole (tensor of shape [3])
    """

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for peg-in-hole insertion.

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts
        """
        env = self.env

        # Extract observation dictionary for first environment
        obs_dict = obs["policy"][env_id]  # Get dict with keys: ee_pos, ee_quat, goal_pos, etc.
        goal_pos = obs_dict["goal_pos"]  # hole position

        # Use current end-effector position as starting point
        init_ee_pose = self._ee_pos_init_local[env_id]

        # Get wall orientation and compute axes - MUST BE DONE FIRST
        wall = env.scene["wall"]
        wall_quat = wall.data.root_quat_w[env_id]
        wall_normal, _, wall_up = compute_wall_axes(wall_quat)
        wall_aligned_quat = compute_wall_aligned_quat(wall_normal)

        # Offset goal position 2.5cm up (orientation-aware)
        # goal_pos_offset = goal_pos + wall_up * 0.02
        goal_pos_offset = goal_pos

        # Calculate waypoints based on wall normal
        # Approach from the front of the wall
        approach_dist = torch.empty(1, device=self.device).uniform_(0.40, 0.60).item()
        approach_pose = goal_pos_offset - wall_normal * approach_dist

        # Linear interpolation for the first waypoint
        # random coeff between 0.2 to 0.4
        coeff = torch.empty(1, device=self.device).uniform_(0.2, 0.4).item()
        wp1_pose = init_ee_pose + coeff * (approach_pose - init_ee_pose)

        # WP2 is the aligned approach position
        wp2_pose = approach_pose

        # Account for the robot-specific grasp offset instead of assuming one EE geometry.
        peg_pos = env.peg_object.data.root_pos_w[env_id] - env.scene.env_origins[env_id]
        peg_center_offset = torch.dot(peg_pos - init_ee_pose, wall_normal).clamp_min(0.0)
        peg_tip_offset = peg_center_offset + env.cfg.peg_length * 0.5
        goal_pose = goal_pos_offset - wall_normal * peg_tip_offset

        # Fixed orientation (identity quaternion [w, x, y, z])
        identity_quat = self.identity_quat()

        # Define trajectory with waypoints
        # Gripper: -1 = close, 1 = open
        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=200, xyz=wp1_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=800, xyz=wp2_pose, quat=wall_aligned_quat, gripper=-1.0),
            Waypoint(t=1100, xyz=goal_pose, quat=wall_aligned_quat, gripper=-1.0),
        ]

        self._log_trajectory_summary()
