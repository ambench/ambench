# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Control allocation utilities for multirotor systems.

This module provides functions for mapping desired wrenches (forces and torques)
to motor thrusts using control allocation matrices.
"""

import torch


def _unwrap_angles_to_previous(
    angles: torch.Tensor,
    prev_angles: torch.Tensor | None,
) -> torch.Tensor:
    """Unwrap ``angles`` (A) element-wise against ``prev_angles`` (A_prev).

    Compare atan2 output A to measured joint position A_prev (not the previous command).
    Per element: if ``A - A_prev >= pi`` subtract ``2*pi``; elif ``A - A_prev <= -pi`` add ``2*pi``.
    """
    if prev_angles is None:
        return angles
    if prev_angles.shape != angles.shape:
        return angles

    two_pi = 2.0 * torch.pi
    pi = torch.pi
    diff = angles - prev_angles
    return torch.where(
        diff >= pi,
        angles - two_pi,
        torch.where(diff <= -pi, angles + two_pi, angles),
    )


def _build_wrench_columns(
    motor_positions_b: torch.Tensor,
    force_dirs_b: torch.Tensor,
    k_f: float,
    spin_dirs: torch.Tensor,
) -> torch.Tensor:
    """Build wrench columns [num_envs, 6, num_cols] from force directions."""
    r_b = motor_positions_b
    thrust_torques = torch.cross(r_b, force_dirs_b, dim=-1)
    drag_torques = spin_dirs.unsqueeze(-1) * k_f * force_dirs_b
    total_torques = thrust_torques + drag_torques
    return torch.cat([force_dirs_b, total_torques], dim=-1)


def _variable_tilt_allocation_matrix(
    motor_positions_b: torch.Tensor,
    motor_cos_dirs_b: torch.Tensor,
    motor_sin_dirs_b: torch.Tensor,
    k_f: float,
    spin_dirs: torch.Tensor,
) -> torch.Tensor:
    """Build 6x12 allocation matrix for F_i cos(alpha_i), F_i sin(alpha_i) unknowns."""
    cos_wrench = _build_wrench_columns(motor_positions_b, motor_cos_dirs_b, k_f, spin_dirs)
    sin_wrench = _build_wrench_columns(motor_positions_b, motor_sin_dirs_b, k_f, spin_dirs)
    # [num_envs, num_motors, 6] -> [num_envs, 6 wrench rows, 12 unknown cols] with interleaved cos/sin.
    paired = torch.stack([cos_wrench, sin_wrench], dim=2)
    return paired.permute(0, 3, 1, 2).reshape(paired.shape[0], 6, -1)


def _apply_thrust_limits_on_components(
    thrust_components: torch.Tensor,
    min_thrust: float | torch.Tensor | None,
    max_thrust: float | torch.Tensor | None,
) -> torch.Tensor:
    """Scale F_i cos/sin pairs so thrust magnitudes respect per-rotor limits."""
    magnitudes = torch.linalg.norm(thrust_components, dim=-1)
    scale = torch.ones_like(magnitudes)

    if max_thrust is not None:
        max_val = max_thrust if isinstance(max_thrust, (int, float)) else max_thrust.to(thrust_components.device)
        scale = torch.minimum(scale, max_val / magnitudes.clamp_min(1.0e-6))

    if min_thrust is not None:
        min_val = min_thrust if isinstance(min_thrust, (int, float)) else min_thrust.to(thrust_components.device)
        scale = torch.maximum(scale, min_val / magnitudes.clamp_min(1.0e-6))

    return thrust_components * scale.unsqueeze(-1)


def ctrl_alloc(
    target_wrench_b: torch.Tensor,
    motor_positions_b: torch.Tensor,
    motor_dirs_b: torch.Tensor,
    k_f: float,
    spin_dirs: torch.Tensor,
    min_thrust: float | torch.Tensor | None = None,
    max_thrust: float | torch.Tensor | None = None,
    fully_actuated: bool = True,
    motor_cos_dirs_b: torch.Tensor | None = None,
    motor_sin_dirs_b: torch.Tensor | None = None,
    prev_motor_arm_angles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute motor thrust vectors from desired wrench using pseudo-inverse or least squares.

    Fixed-tilt platforms pass ``motor_dirs_b`` only. Variable-tilt platforms also pass
    ``motor_cos_dirs_b`` and ``motor_sin_dirs_b``; the allocator solves for
    ``F_i cos(alpha_i)`` and ``F_i sin(alpha_i)`` with a 6x12 matrix, then recovers
    ``F_i = ||[F_i cos, F_i sin]||_2`` and ``alpha_i = atan2(F_i sin, F_i cos)``.

    Args:
        target_wrench_b: Desired [fx, fy, fz, tx, ty, tz] in body frame [num_envs, 6]
        motor_positions_b: Motor positions in body frame [num_motors, 3] or [num_envs, num_motors, 3]
        motor_dirs_b: Motor thrust directions in body frame [num_motors, 3] or [num_envs, num_motors, 3]
        k_f: Thrust to torque ratio
        spin_dirs: Motor spin directions (+1 or -1) [num_motors] or [num_envs, num_motors]
        min_thrust: Minimum thrust per motor (float, tensor [num_motors], or None)
        max_thrust: Maximum thrust per motor (float, tensor [num_motors], or None)
        fully_actuated: If True, uses pseudo-inverse. If False, uses least squares for underactuated systems.
        motor_cos_dirs_b: Variable-tilt cos(alpha) force basis [num_motors, 3] or [num_envs, num_motors, 3]
        motor_sin_dirs_b: Variable-tilt sin(alpha) force basis [num_motors, 3] or [num_envs, num_motors, 3]
        prev_motor_arm_angles: Measured motor-arm joint positions for atan2 unwrap
            [num_envs, num_motors] (e.g. ``robot.data.joint_pos``; not the previous command)

    Returns:
        rotor_thrust_vectors: Motor thrust vectors in body frame [num_envs, num_motors, 3]
        motor_thrust_magnitudes: Motor thrust magnitudes [num_envs, num_motors]
        motor_arm_angles: Motor-arm joint angles [num_envs, num_motors] when variable tilt is used, else None
    """
    device = target_wrench_b.device
    num_envs = target_wrench_b.shape[0]

    if motor_dirs_b.dim() == 2:
        motor_dirs_b = motor_dirs_b.unsqueeze(0).expand(num_envs, -1, -1)
    if spin_dirs.dim() == 1:
        spin_dirs = spin_dirs.unsqueeze(0).expand(num_envs, -1)
    if motor_positions_b.dim() == 2:
        motor_positions_b = motor_positions_b.unsqueeze(0).expand(num_envs, -1, -1)

    if motor_cos_dirs_b is not None and motor_sin_dirs_b is not None:
        if motor_cos_dirs_b.dim() == 2:
            motor_cos_dirs_b = motor_cos_dirs_b.unsqueeze(0).expand(num_envs, -1, -1)
        if motor_sin_dirs_b.dim() == 2:
            motor_sin_dirs_b = motor_sin_dirs_b.unsqueeze(0).expand(num_envs, -1, -1)

        allocation_matrix = _variable_tilt_allocation_matrix(
            motor_positions_b,
            motor_cos_dirs_b,
            motor_sin_dirs_b,
            k_f,
            spin_dirs,
        )
        allocation_pinv = torch.linalg.pinv(allocation_matrix)
        thrust_components_flat = torch.einsum("bij,bj->bi", allocation_pinv, target_wrench_b)
        thrust_components = thrust_components_flat.reshape(num_envs, -1, 2)

        if min_thrust is not None or max_thrust is not None:
            thrust_components = _apply_thrust_limits_on_components(thrust_components, min_thrust, max_thrust)

        motor_thrust_magnitudes = torch.linalg.norm(thrust_components, dim=-1)
        motor_arm_angles = torch.atan2(thrust_components[..., 1], thrust_components[..., 0])
        motor_arm_angles = _unwrap_angles_to_previous(motor_arm_angles, prev_motor_arm_angles)
        rotor_thrust_vectors = (
            thrust_components[..., 0:1] * motor_cos_dirs_b + thrust_components[..., 1:2] * motor_sin_dirs_b
        )
        return rotor_thrust_vectors, motor_thrust_magnitudes, motor_arm_angles

    r_b = motor_positions_b
    thrust_torques = torch.cross(r_b, motor_dirs_b, dim=-1)
    drag_torques = spin_dirs.unsqueeze(-1) * k_f * motor_dirs_b
    total_torques = thrust_torques + drag_torques

    if fully_actuated:
        allocation_matrix = torch.cat([motor_dirs_b, total_torques], dim=-1).transpose(-2, -1)
        allocation_pinv = torch.linalg.pinv(allocation_matrix)
        motor_thrust_magnitudes = torch.einsum("bij,bj->bi", allocation_pinv, target_wrench_b)
    else:
        allocation_matrix = torch.stack(
            [
                motor_dirs_b[:, :, 2],
                total_torques[:, :, 0],
                total_torques[:, :, 1],
                total_torques[:, :, 2],
            ],
            dim=1,
        )
        target_wrench_reduced = target_wrench_b[:, [2, 3, 4, 5]]
        allocation_pinv = torch.linalg.pinv(allocation_matrix)
        motor_thrust_magnitudes = torch.einsum("bij,bj->bi", allocation_pinv, target_wrench_reduced)

    if min_thrust is not None or max_thrust is not None:
        min_val = (
            min_thrust
            if isinstance(min_thrust, (int, float))
            else (min_thrust.to(device) if min_thrust is not None else None)
        )
        max_val = (
            max_thrust
            if isinstance(max_thrust, (int, float))
            else (max_thrust.to(device) if max_thrust is not None else None)
        )
        motor_thrust_magnitudes = torch.clamp(motor_thrust_magnitudes, min=min_val, max=max_val)

    rotor_thrust_vectors = motor_thrust_magnitudes.unsqueeze(-1) * motor_dirs_b
    return rotor_thrust_vectors, motor_thrust_magnitudes, None


def inv_ctrl_alloc(
    rotor_thrust_vectors: torch.Tensor,
    motor_positions_b: torch.Tensor,
    k_f: float,
    spin_dirs: torch.Tensor,
) -> torch.Tensor:
    """Compute net wrench from rotor thrust vectors (inverse of ctrl_alloc).

    Args:
        rotor_thrust_vectors: Rotor thrust vectors in body frame [num_envs, num_motors, 3]
        motor_positions_b: Motor positions in body frame [num_motors, 3] or [num_envs, num_motors, 3]
        k_f: Steady-state thrust-to-reaction-torque ratio
        spin_dirs: Motor spin directions (+1 or -1) [num_motors] or [num_envs, num_motors]

    Returns:
        wrench_b: Net wrench in body frame [num_envs, 6]
                 Format: [fx, fy, fz, tx, ty, tz]
    """
    num_envs = rotor_thrust_vectors.shape[0]
    if motor_positions_b.dim() == 2:
        motor_positions_b = motor_positions_b.unsqueeze(0).expand(num_envs, -1, -1)
    if spin_dirs.dim() == 1:
        spin_dirs = spin_dirs.unsqueeze(0).expand(num_envs, -1)

    net_force_b = torch.sum(rotor_thrust_vectors, dim=1)
    thrust_torques = torch.cross(motor_positions_b, rotor_thrust_vectors, dim=-1)
    drag_torques = spin_dirs.unsqueeze(-1) * k_f * rotor_thrust_vectors
    net_torque_b = torch.sum(thrust_torques + drag_torques, dim=1)

    return torch.cat([net_force_b, net_torque_b], dim=1)
