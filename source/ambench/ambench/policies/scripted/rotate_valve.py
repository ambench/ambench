# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.utils.math as math_utils
import torch

from ..utils import compute_wall_axes
from .base import BasePolicy, Waypoint


class RotateValvePolicy(BasePolicy):
    """Scripted policy for rotate valve task.

    Observation structure (nested dict):
    - ee_state: dict with end-effector state (pos, quat, vel, ang_vel)
    - goal_pos: position of the valve handle (tensor of shape [3])
    """

    t_navigate_to_valve = 450
    t_approach_valve = 600
    t_grasp_valve = 650
    t_start_arc = 700
    t_end_arc = 1300
    t_release_valve = 1400

    approach_offset_range = (0.3, 0.5)
    grasp_offset = 0.18
    valve_arc_length = 0.24
    arc_sweep_angle = float(1.8 * torch.pi)

    def __init__(self, env=None, inject_noise: bool = False):
        super().__init__(env, inject_noise)
        self.circle_generated = False
        self.circle_points: torch.Tensor | None = None

        self.approach_offset = math_utils.sample_uniform(
            lower=self.approach_offset_range[0],
            upper=self.approach_offset_range[1],
            size=1,
            device=self.device,
        ).item()

    def generate_circle(
        self,
        point1: torch.Tensor,
        point2: torch.Tensor,
        total_theta: float,
        total_points: int,
        wall_y_axis: torch.Tensor,
        wall_z_axis: torch.Tensor,
    ) -> torch.Tensor:
        """Generate arc points in the wall's Y-Z plane (parallel to wall surface).

        Args:
            point1: Start point of arc (3,)
            point2: End point of arc (3,)
            total_theta: Total angle sweep in radians
            total_points: Number of points to generate
            wall_y_axis: Wall's Y axis direction in world frame (3,)
            wall_z_axis: Wall's Z axis direction in world frame (3,)
        """
        center = 0.5 * (point1 + point2)
        radius = 0.5 * torch.norm(point2 - point1)
        theta = torch.linspace(0.0, total_theta, total_points, device=self.device)

        # Generate circle in wall's Y-Z plane
        y_comp = radius * torch.sin(theta)
        z_comp = -radius * torch.cos(theta)

        # Transform to world coordinates
        points = (
            center.unsqueeze(0)
            + y_comp.unsqueeze(1) * wall_y_axis.unsqueeze(0)
            + z_comp.unsqueeze(1) * wall_z_axis.unsqueeze(0)
        )
        return points

    def interpolate_circle(
        self, curr_waypoint: Waypoint, next_waypoint: Waypoint, t: int
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Interpolate along precomputed circular arc between two time indices."""
        if not self.circle_generated or self.circle_points is None:
            raise RuntimeError("Circle points not generated before calling interpolate_circle.")

        t_seg = t - curr_waypoint.t  # index into circle_points
        # Clamp to valid range
        t_seg = max(0, min(t_seg, len(self.circle_points) - 1))
        xyz = self.circle_points[t_seg]

        # Use SLERP for quaternion interpolation (not linear!)
        t_frac = (t - curr_waypoint.t) / float(next_waypoint.t - curr_waypoint.t)
        t_frac = max(0.0, min(1.0, t_frac))  # Clamp to [0, 1]
        curr_quat = curr_waypoint.quat
        next_quat = next_waypoint.quat
        curr_grip = curr_waypoint.gripper
        next_grip = next_waypoint.gripper

        quat = math_utils.quat_slerp(curr_quat, next_quat, t_frac)
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def generate_trajectory(self, obs: dict, env_id: int = 0):
        """Generate waypoint trajectory for rotating the valve.

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts
        """
        env = self.env

        obs_dict = obs["policy"][env_id]
        goal_pos = obs_dict["goal_pos"]  # valve handle position

        init_ee_pose = self._ee_pos_init_local[env_id]

        # Get wall orientation and compute axes
        wall = env.scene["wall"]
        wall_quat = wall.data.root_quat_w[env_id]
        wall_normal, wall_y_axis, wall_z_axis = compute_wall_axes(wall_quat)

        # Compute waypoint positions
        approach_pos = goal_pos - wall_normal * self.approach_offset
        grasp_pos = goal_pos - wall_normal * self.grasp_offset

        # End position: offset along wall's Z axis in wall's local frame
        valve_end_pos_local = torch.tensor([0.0, 0.0, self.valve_arc_length], dtype=torch.float32, device=self.device)
        valve_end_pos = grasp_pos + math_utils.quat_apply(
            wall_quat.unsqueeze(0), valve_end_pos_local.unsqueeze(0)
        ).squeeze(0)

        # Precompute circle arc points
        num_arc_points = self.t_end_arc - self.t_start_arc
        self.circle_points = self.generate_circle(
            grasp_pos, valve_end_pos, self.arc_sweep_angle, num_arc_points, wall_y_axis, wall_z_axis
        )
        self.circle_generated = True

        identity_quat = self.identity_quat()

        # Build waypoints
        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.t_navigate_to_valve, xyz=approach_pos, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.t_approach_valve, xyz=grasp_pos, quat=identity_quat, gripper=1.0),
            Waypoint(t=self.t_grasp_valve, xyz=grasp_pos, quat=identity_quat, gripper=1.0),
            Waypoint(
                t=self.t_start_arc,
                xyz=grasp_pos,
                quat=identity_quat,
                gripper=-1.0,
                interpolation_type="circle",
            ),
            Waypoint(t=self.t_end_arc, xyz=valve_end_pos, quat=identity_quat, gripper=-1.0),
            Waypoint(t=self.t_release_valve, xyz=valve_end_pos, quat=identity_quat, gripper=1.0),
        ]

        self._log_trajectory_summary(f"goal_pos={goal_pos}")


class RotateValvePolicyFast(RotateValvePolicy):
    """Scripted policy for rotate valve task.

    Observation structure (nested dict):
    - ee_state: dict with end-effector state (pos, quat, vel, ang_vel)
    - goal_pos: position of the valve handle (tensor of shape [3])
    """

    t_navigate_to_valve = 250
    t_approach_valve = 300
    t_grasp_valve = 320
    t_start_arc = 330
    t_end_arc = 620
    t_release_valve = 650
