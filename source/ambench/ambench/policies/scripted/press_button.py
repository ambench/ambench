# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.utils.math as math_utils
import torch

from ..utils import compute_wall_axes
from .base import BasePolicy, Waypoint


class PressButtonPolicy(BasePolicy):
    """Scripted policy for press-button task.

    Observation structure (nested dict):
    - ee_pos: end-effector position (3,)
    - ee_quat: end-effector quaternion (4,)
    - goal_pos: button plunger position (3,)

    """

    waypoint_times = (200, 800, 1200)

    def generate_trajectory(self, obs: dict, env_id: int = 0):
        """Generate waypoint trajectory for pressing the button.

        Logic mirrors the 3-waypoint structure used in PegInHole:
        - wp1: partial x motion
        - wp2: near target x + full y/z alignment
        - goal: press into the button along +x

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts.
        """
        env = self.env

        obs_dict = obs["policy"][env_id]

        goal_pos = obs_dict["goal_pos"]

        # Use initial ee_pos as starting point
        init_ee_pose = self._ee_pos_init_local[env_id]

        # Get wall orientation and compute axes
        wall = env.scene["wall"]
        wall_quat = wall.data.root_quat_w[env_id]
        wall_normal, _, _ = compute_wall_axes(wall_quat)

        # Approach position: backed off along wall normal
        # Assuming normal points OUT (towards robot).
        approach_dist = math_utils.sample_uniform(lower=0.3, upper=0.5, size=1, device=self.device).item()
        # approach_dist = 0.3
        approach_pose = goal_pos - wall_normal * approach_dist

        # Press position: push into the wall (along -normal)
        press_depth = 0.00
        z_offset = -0.01
        press_pose = goal_pos + wall_normal * press_depth + torch.tensor([0.0, 0.0, z_offset], device=self.device)

        # WP1: Halfway to approach, linear interp
        coeff = math_utils.sample_uniform(lower=0.2, upper=0.5, size=1, device=self.device).item()
        # coeff = 0.3
        wp1_pose = init_ee_pose + coeff * (approach_pose - init_ee_pose)

        # WP2: Aligned approach
        wp2_pose = approach_pose

        # Goal: Pressed state
        goal_pose = press_pose

        # Fixed orientation (identity quaternion [w, x, y, z])
        identity_quat = self.identity_quat()

        # Gripper convention follows existing scripted policies in this repo:
        # 1.0 = open, -1.0 = close

        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[0], xyz=wp1_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[1], xyz=wp2_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=self.waypoint_times[2], xyz=goal_pose, quat=identity_quat, gripper=-1.0),
        ]
        self._log_trajectory_summary(f"goal_pos={goal_pos}")


class PressButtonPolicyFast(PressButtonPolicy):
    """Fast Scripted policy for press-button task.

    Observation structure (nested dict):
    - ee_pos: end-effector position (3,)
    - ee_quat: end-effector quaternion (4,)
    - goal_pos: button plunger position (3,)

    """

    waypoint_times = (120, 420, 620)
