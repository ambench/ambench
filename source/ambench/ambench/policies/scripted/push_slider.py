# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.utils.math as math_utils

from ..utils import compute_wall_aligned_quat, compute_wall_axes
from .base import BasePolicy, Waypoint


class PushSliderPolicy(BasePolicy):
    """Scripted policy for push-slider task."""

    grasp_dist_offset = 0.23
    waypoint_times = (300, 600, 800, 820, 840, 1440)

    def generate_trajectory(self, obs: dict, env_id: int = 0):
        """Generate waypoint trajectory for pushing the slider."""
        env = self.env

        init_ee_pose = self._ee_pos_init_local[env_id]

        slider = env.scene["slider"]
        wall = env.scene["wall"]

        wall_quat = wall.data.root_quat_w[env_id]

        slider_link_idx = slider.find_bodies("Slider")[0][0]
        slider_pos_w = slider.data.body_pos_w[env_id, slider_link_idx]

        env_origin = env.scene.env_origins[env_id]
        slider_pos_local_frame = slider_pos_w - env_origin

        wall_normal_w, wall_y_w, _ = compute_wall_axes(wall_quat)

        slide_direction = -wall_y_w
        slide_length = 1.0

        start_pos = slider_pos_local_frame - wall_normal_w * self.grasp_dist_offset

        approach_dist = math_utils.sample_uniform(lower=0.3, upper=0.5, size=1, device=self.device).item()
        approach_pos = start_pos - wall_normal_w * approach_dist

        coeff = math_utils.sample_uniform(lower=0.2, upper=0.5, size=1, device=self.device).item()
        wp1_pose = init_ee_pose + coeff * (approach_pos - init_ee_pose)

        end_pos = start_pos + slide_direction * slide_length

        target_quat = compute_wall_aligned_quat(wall_normal_w)

        identity_quat = self.identity_quat()

        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[0], xyz=wp1_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[1], xyz=approach_pos, quat=target_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[2], xyz=start_pos, quat=target_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[3], xyz=start_pos, quat=target_quat, gripper=1.0),
            Waypoint(t=self.waypoint_times[4], xyz=start_pos, quat=target_quat, gripper=-1.0),
            Waypoint(t=self.waypoint_times[5], xyz=end_pos, quat=target_quat, gripper=-1.0),
        ]
        self._log_trajectory_summary(f"start_pos={start_pos}", f"end_pos={end_pos}")


class PushSliderPolicyFast(PushSliderPolicy):
    """Scripted policy for push-slider task."""

    grasp_dist_offset = 0.18
    waypoint_times = (80, 200, 240, 250, 260, 620)
