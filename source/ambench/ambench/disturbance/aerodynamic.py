# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Aerodynamic force models used by multirotor controllers."""

from __future__ import annotations

import math
from typing import Any

import torch


def compute_drag_force(
    body_velocity_b: torch.Tensor,
    drag_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Compute linear body-frame drag ``F = -c * v``."""

    if body_velocity_b.ndim != 2 or body_velocity_b.shape[-1] != 3:
        raise ValueError(f"Expected body velocities with shape (num_envs, 3), got {body_velocity_b.shape}.")
    if drag_coefficients.shape not in ((3,), body_velocity_b.shape):
        raise ValueError(
            "Expected drag coefficients with shape (3,) or the same shape as body velocities, "
            f"got {drag_coefficients.shape}."
        )

    coefficients = drag_coefficients.to(device=body_velocity_b.device, dtype=body_velocity_b.dtype)
    return -coefficients * body_velocity_b


def apply_ground_effect(
    rotor_thrust_vectors: torch.Tensor,
    motor_positions_w: torch.Tensor,
    motor_axis_directions_w: torch.Tensor,
    aerodynamic_coeffs: dict[str, Any],
    d_ground: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Augment rotor thrust using the configured distance-based ground model.

    The model uses
    ``T_commanded / T_actual = b - k * (R / (4 z))**2``. The ratio is capped at
    one so this term only represents thrust augmentation. Passing ``d_ground``
    bypasses PhysX raycasts, which is useful for deterministic tests.
    """

    propeller_radius = float(aerodynamic_coeffs.get("propeller_radius", 0.1))
    ground_effect_b = float(aerodynamic_coeffs.get("ground_effect_b", 1.0))
    ground_effect_k = float(aerodynamic_coeffs.get("ground_effect_k", 0.171))
    max_raycast_distance = float(aerodynamic_coeffs.get("max_raycast_distance", 5.0))
    if propeller_radius <= 0.0:
        raise ValueError(f"Expected a positive propeller radius, got {propeller_radius}.")
    if not math.isfinite(ground_effect_b) or ground_effect_b <= 0.0:
        raise ValueError(f"Expected a positive finite ground-effect b coefficient, got {ground_effect_b}.")
    if not math.isfinite(ground_effect_k) or ground_effect_k < 0.0:
        raise ValueError(f"Expected a non-negative finite ground-effect k coefficient, got {ground_effect_k}.")

    if d_ground is None:
        from .utils.aerodynamic_utils import compute_ground_effect_distance

        d_ground = compute_ground_effect_distance(
            motor_positions_w=motor_positions_w,
            motor_axis_directions_w=motor_axis_directions_w,
            max_raycast_distance=max_raycast_distance,
        )

    if d_ground.shape != rotor_thrust_vectors.shape[:2]:
        raise ValueError(
            f"Expected ground distances with shape {rotor_thrust_vectors.shape[:2]}, got {d_ground.shape}."
        )
    distance = d_ground.to(device=rotor_thrust_vectors.device, dtype=rotor_thrust_vectors.dtype)
    distance = distance.clamp_min(propeller_radius / 4.0)
    ratio = ground_effect_b - ground_effect_k * (propeller_radius / (4.0 * distance)).square()
    ratio = ratio.clamp(min=torch.finfo(ratio.dtype).eps, max=1.0)

    modified_thrust_vectors = rotor_thrust_vectors / ratio.unsqueeze(-1)
    thrust_difference_b = modified_thrust_vectors - rotor_thrust_vectors
    return modified_thrust_vectors, d_ground, thrust_difference_b


def apply_near_wall_effect(
    rotor_thrust_vectors: torch.Tensor,
    motor_positions_w: torch.Tensor,
    drone_position_w: torch.Tensor,
    base_rot_wb: torch.Tensor,
    aerodynamic_coeffs: dict[str, Any],
    d_wall_norm: torch.Tensor | None = None,
    wall_direction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Apply the configured near-wall force model to rotor thrust."""

    num_envs, num_motors, _ = rotor_thrust_vectors.shape
    expected_distance_shape = (num_envs, num_motors)
    expected_direction_shape = (num_envs, num_motors, 3)

    propeller_radius = float(aerodynamic_coeffs.get("propeller_radius", 0.1))
    wall_effect_a1 = float(aerodynamic_coeffs.get("wall_effect_a1", 0.05))
    wall_effect_b1 = float(aerodynamic_coeffs.get("wall_effect_b1", 0.34))
    wall_effect_a2 = float(aerodynamic_coeffs.get("wall_effect_a2", 0.02))
    wall_effect_b2 = float(aerodynamic_coeffs.get("wall_effect_b2", 0.25))
    max_raycast_distance = float(aerodynamic_coeffs.get("max_raycast_distance", 5.0))
    if propeller_radius <= 0.0:
        raise ValueError(f"Expected a positive propeller radius, got {propeller_radius}.")

    d_wall = None
    if d_wall_norm is None:
        from .utils.aerodynamic_utils import compute_near_wall_effect_distance

        d_wall, d_wall_norm, wall_direction = compute_near_wall_effect_distance(
            motor_positions_w=motor_positions_w,
            drone_position_w=drone_position_w,
            propeller_radius=propeller_radius,
            max_raycast_distance=max_raycast_distance,
        )
    elif wall_direction is None:
        raise ValueError("wall_direction is required when d_wall_norm is provided.")

    if d_wall_norm.shape != expected_distance_shape:
        raise ValueError(f"Expected wall distances with shape {expected_distance_shape}, got {d_wall_norm.shape}.")
    if wall_direction.shape != expected_direction_shape:
        raise ValueError(f"Expected wall directions with shape {expected_direction_shape}, got {wall_direction.shape}.")

    d_wall_norm = d_wall_norm.to(device=rotor_thrust_vectors.device, dtype=rotor_thrust_vectors.dtype)
    wall_direction = wall_direction.to(device=rotor_thrust_vectors.device, dtype=rotor_thrust_vectors.dtype)
    wall_direction = wall_direction / torch.linalg.vector_norm(wall_direction, dim=-1, keepdim=True).clamp_min(1e-8)

    horizontal_gain = wall_effect_a1 * wall_effect_b1**d_wall_norm
    vertical_gain = 1.0 + wall_effect_a2 * wall_effect_b2**d_wall_norm

    thrust_w = torch.einsum("bij,bmj->bmi", base_rot_wb, rotor_thrust_vectors)
    vertical_thrust_w = torch.zeros_like(thrust_w)
    vertical_thrust_w[..., 2] = thrust_w[..., 2]
    horizontal_thrust_w = thrust_w - vertical_thrust_w
    vertical_magnitude = torch.linalg.vector_norm(vertical_thrust_w, dim=-1, keepdim=True)

    modified_thrust_w = (
        horizontal_thrust_w
        + horizontal_gain.unsqueeze(-1) * vertical_magnitude * wall_direction
        + vertical_gain.unsqueeze(-1) * vertical_thrust_w
    )
    modified_thrust_b = torch.einsum("bij,bmj->bmi", base_rot_wb.transpose(-2, -1), modified_thrust_w)
    thrust_difference_b = modified_thrust_b - rotor_thrust_vectors
    return modified_thrust_b, d_wall, thrust_difference_b
