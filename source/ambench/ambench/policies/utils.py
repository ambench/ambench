# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.utils.math as math_utils
import torch


def compute_wall_axes(wall_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute wall's local axes in world frame.

    Args:
        wall_quat: Wall quaternion (4,)

    Returns:
        Tuple of (wall_normal, wall_y_axis, wall_z_axis) each of shape (3,)
    """
    device = wall_quat.device

    # Wall's local axes: X=normal, Y=horizontal, Z=vertical
    local_axes = torch.eye(3, device=device)  # [[1,0,0], [0,1,0], [0,0,1]]
    world_axes = math_utils.quat_apply(wall_quat.unsqueeze(0), local_axes.unsqueeze(0)).squeeze(0)
    return world_axes[0], world_axes[1], world_axes[2]  # normal, y_axis, z_axis


def compute_wall_aligned_quat(wall_normal: torch.Tensor) -> torch.Tensor:
    """Compute end-effector quaternion aligned with wall normal.

    Args:
        wall_normal: Wall normal vector (3,)

    Returns:
        Quaternion (4,) that aligns gripper's X-axis with wall_normal
    """
    device = wall_normal.device

    gripper_forward = torch.tensor([1.0, 0.0, 0.0], device=device)
    v1 = gripper_forward / torch.norm(gripper_forward)
    v2 = wall_normal / torch.norm(wall_normal)

    dot = torch.dot(v1, v2)
    cross = torch.linalg.cross(v1, v2)

    # Optimized Shortest Arc Quaternion: q = normalize([1 + dot, cross])
    quat = torch.cat([(1.0 + dot).unsqueeze(0), cross])

    # Handle singularity at 180 degree (dot == -1) where quat becomes zero
    if torch.linalg.norm(quat) < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    return math_utils.normalize(quat)
