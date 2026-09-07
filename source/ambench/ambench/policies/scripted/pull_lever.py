# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted policy for pull lever task."""

import torch

from .base import BasePolicy, Waypoint


class PullLeverPolicy(BasePolicy):
    """Scripted policy for pull lever task."""

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for pull lever task."""
        obs_dict = obs["policy"][env_id]

        init_ee_pose = self._ee_pos_init_local[env_id]

        roll_90_quat = torch.tensor([0.7071, 0.7071, 0, 0], dtype=torch.float32, device=self.device)

        lever_pos = obs_dict["lever_pos"].clone()

        lever_handle_pos = lever_pos.clone()
        lever_handle_pos[0] -= 0.25
        lever_handle_pos[1] += 0.04

        approach_dist = 0.50
        approach_pos = lever_handle_pos.clone()
        approach_pos[0] -= approach_dist

        grasp_pos = lever_handle_pos.clone()
        grasp_pos[0] -= 0.08

        pull_height = 0.25
        pull_pos = lever_handle_pos.clone()
        pull_pos[0] += 0.06
        pull_pos[2] += pull_height

        retract_pos = pull_pos.clone()
        retract_pos[0] -= 0.65

        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=roll_90_quat, gripper=1.0),
            Waypoint(t=500, xyz=approach_pos, quat=roll_90_quat, gripper=1.0),
            Waypoint(t=750, xyz=grasp_pos, quat=roll_90_quat, gripper=1.0),
            Waypoint(t=780, xyz=grasp_pos, quat=roll_90_quat, gripper=-1.0),
            Waypoint(t=800, xyz=grasp_pos, quat=roll_90_quat, gripper=-1.0),
            Waypoint(t=1000, xyz=pull_pos, quat=roll_90_quat, gripper=-1.0),
            Waypoint(t=1050, xyz=pull_pos, quat=roll_90_quat, gripper=-1.0),
            Waypoint(t=1080, xyz=pull_pos, quat=roll_90_quat, gripper=1.0),
            Waypoint(t=1100, xyz=pull_pos, quat=roll_90_quat, gripper=1.0),
            Waypoint(t=1400, xyz=retract_pos, quat=roll_90_quat, gripper=1.0),
        ]

        self._log_trajectory_summary(f"lever_pos={lever_pos}", f"lever_handle_pos={lever_handle_pos}")
