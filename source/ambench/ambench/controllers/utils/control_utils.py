# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Robot utility functions.

This module provides common utility functions for working with robot articulations,
including physical-parameter computation and motor-frame transforms.
"""

import torch
from isaaclab.assets import Articulation


def get_aggregate_physical_parameters(robot: Articulation, scene):
    """Compute aggregate mass and inertia matrix of all links relative to the base link coordinate frame.

    Uses zero configuration as the reference.
    Also retrieves gravity vector from the simulation.

    Args:
        robot: Articulation object representing the robot
        scene: Scene object used to read the simulation gravity.

    Returns:
        tuple: (total_mass, total_inertia_matrix, gravity) - scalar mass, 3x3 inertia matrix, and gravity vector
    """
    if scene is None:
        raise ValueError("scene is required to compute the simulation gravity.")

    device = robot.device
    # Get num_envs from data shape (first dimension of robot data tensors)
    body_masses = robot.data.default_mass
    num_envs = body_masses.shape[0]
    num_bodies = robot.num_bodies

    # 1. Mass of all links [num_envs, num_bodies]
    body_masses = body_masses.to(device)
    total_mass = torch.sum(body_masses, dim=1)  # [num_envs]

    # 2. Get inertia tensors for all links and convert to principal moments format
    body_inertias_raw = robot.root_physx_view.get_inertias().to(device)

    # Convert to [num_envs, num_bodies, 3] format (principal moments)
    body_inertias_3x3 = body_inertias_raw.view(num_envs, num_bodies, 3, 3)
    body_inertias = torch.diagonal(body_inertias_3x3, dim1=2, dim2=3)  # [num_envs, num_bodies, 3]

    # 3. Local pose of each link relative to base link
    body_pos_w = robot.data.body_pos_w.to(device)  # [num_envs, num_bodies, 3]

    # Base link is typically at index 0, but we'll use the first body
    base_link_pos_w = body_pos_w[:, 0, :]  # [num_envs, 3]

    # body_pos_relative: distance vector r from base link origin to each link origin
    body_pos_relative = body_pos_w - base_link_pos_w.unsqueeze(1)  # [num_envs, num_bodies, 3]

    # Initialize total inertia matrix [num_envs, 3, 3]
    total_inertia_matrix = torch.zeros((num_envs, 3, 3), device=device)

    for i in range(num_bodies):
        m = body_masses[:, i]  # [num_envs]
        r = body_pos_relative[:, i, :]  # [num_envs, 3]

        # (1) Create local diagonal inertia matrix
        I_local = torch.diag_embed(body_inertias[:, i, :])  # [num_envs, 3, 3]

        # (Note) If the link's local coordinate frame is rotated relative to base,
        # R*I*R^T would be needed, but most robot arms spawn with aligned axes at zero pose.
        I_rotated = I_local

        # (2) Apply parallel axis theorem
        # I_transport = m * ( (r·r)E - r⊗r )
        r_sq = torch.sum(r**2, dim=1, keepdim=True)  # [num_envs, 1]
        r_outer = torch.bmm(r.unsqueeze(-1), r.unsqueeze(1))  # [num_envs, 3, 3]
        eye = torch.eye(3, device=device).unsqueeze(0).expand(num_envs, 3, 3)

        I_transport = m.view(-1, 1, 1) * (r_sq.view(-1, 1, 1) * eye - r_outer)

        total_inertia_matrix += I_rotated + I_transport

    # Controllers use the positive gravity-compensation vector.
    gravity = -torch.tensor(scene.sim.cfg.gravity, device=device, dtype=torch.float32)

    # Return result for first environment only (usually all envs are identical)
    return total_mass[0].item(), total_inertia_matrix[0], gravity


def compute_motor_positions_from_base(
    robot: Articulation,
    base_positions_w: torch.Tensor,
    base_quaternions_w: torch.Tensor,
    motor_positions_b: torch.Tensor,
    motor_axis_directions_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute motor positions and axis directions in world frame from base pose.

    This function is useful when motor links are not defined in the robot model.
    It transforms motor positions and axes from body frame to world frame using base pose.

    Args:
        robot: Robot articulation object (for device and num_envs)
        base_positions_w: Base positions in world frame [num_envs, 3]
        base_quaternions_w: Base quaternions in world frame [num_envs, 4] (w, x, y, z)
        motor_positions_b: Motor positions in body frame [num_motors, 3] or [num_envs, num_motors, 3]
        motor_axis_directions_b: Motor axis directions in body frame [num_motors, 3] or [num_envs, num_motors, 3]
                                 Typically [0, 0, -1] for downward propellers

    Returns:
        motor_positions_w: Motor positions in world frame [num_envs, num_motors, 3]
        motor_axis_directions_w: Motor axis directions (normalized) in world frame [num_envs, num_motors, 3]
    """
    import isaaclab.utils.math as math_utils

    num_envs = robot.num_envs

    # Ensure motor_positions_b and motor_axis_directions_b have correct shape
    if motor_positions_b.dim() == 2:
        # [num_motors, 3] -> [num_envs, num_motors, 3]
        motor_positions_b = motor_positions_b.unsqueeze(0).expand(num_envs, -1, -1)
    if motor_axis_directions_b.dim() == 2:
        # [num_motors, 3] -> [num_envs, num_motors, 3]
        motor_axis_directions_b = motor_axis_directions_b.unsqueeze(0).expand(num_envs, -1, -1)

    # Rotate motor positions from body frame to world frame
    # motor_pos_w = base_pos_w + R_wb * motor_pos_b
    motor_positions_w = math_utils.quat_apply(base_quaternions_w, motor_positions_b)  # [num_envs, num_motors, 3]
    motor_positions_w = motor_positions_w + base_positions_w.unsqueeze(1)  # Add base position

    # Rotate motor axis directions from body frame to world frame
    motor_axis_directions_w = math_utils.quat_apply(
        base_quaternions_w, motor_axis_directions_b
    )  # [num_envs, num_motors, 3]
    # Normalize
    motor_axis_directions_w = motor_axis_directions_w / torch.norm(motor_axis_directions_w, dim=2, keepdim=True)

    return motor_positions_w, motor_axis_directions_w
