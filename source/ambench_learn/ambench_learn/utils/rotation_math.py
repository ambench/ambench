# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def rotvec_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to quaternions in wxyz order."""

    if rotvec.ndim == 1:
        rotvec = rotvec.unsqueeze(0)
        squeeze = True
    elif rotvec.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected rotvec with shape (3,) or (N, 3). Got {tuple(rotvec.shape)}.")

    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_angle = 0.5 * angle

    identity = torch.zeros(rotvec.shape[0], 4, dtype=rotvec.dtype, device=rotvec.device)
    identity[:, 0] = 1.0

    safe_angle = torch.where(angle > 1.0e-9, angle, torch.ones_like(angle))
    axis = rotvec / safe_angle
    sin_half = torch.sin(half_angle)
    quat = torch.cat((torch.cos(half_angle), axis * sin_half), dim=-1)
    quat = torch.where(angle > 1.0e-9, quat, identity)
    quat = normalize_quat(quat)
    return quat[0] if squeeze else quat


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in wxyz order."""

    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-9)
    return quat / norm


def slerp_quat(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between quaternions in wxyz order.

    Args:
        q0: Start quaternion with shape (..., 4)
        q1: End quaternion with shape (..., 4)
        t: Interpolation fraction broadcastable to q0/q1 with trailing singleton dim.
    """

    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    close = dot > 0.9995
    lerped = normalize_quat((1.0 - t) * q0 + t * q1)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1.0e-9)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    slerped = normalize_quat(w0 * q0 + w1 * q1)

    return torch.where(close, lerped, slerped)


def quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in wxyz order to axis-angle vectors."""

    if quat.ndim == 1:
        quat = quat.unsqueeze(0)
        squeeze = True
    elif quat.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected quat with shape (4,) or (N, 4). Got {tuple(quat.shape)}.")

    quat = normalize_quat(quat)
    scalar = quat[:, :1].clamp(-1.0, 1.0)
    xyz = quat[:, 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, scalar)
    safe_sin_half = torch.where(sin_half > 1.0e-9, sin_half, torch.ones_like(sin_half))
    axis = xyz / safe_sin_half
    rotvec = axis * angle
    # For very small angles, use the first-order quaternion -> rotvec linearization.
    rotvec = torch.where(sin_half > 1.0e-9, rotvec, 2.0 * xyz)
    return rotvec[0] if squeeze else rotvec
