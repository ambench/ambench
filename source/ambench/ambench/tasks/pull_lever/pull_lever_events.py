# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for PullLever environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ambench.tasks.pull_lever.pull_lever_env import PullLever


def randomize_lever_position_on_wall(
    env: PullLever,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    lever_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize lever position on the wall surface (YZ plane relative to wall).

    The lever stays at the same X distance from the wall (on the surface) but can
    move in the YZ plane. The lever also inherits the wall's orientation so they
    stay aligned when the wall is tilted.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        wall_cfg: The wall asset configuration.
        lever_cfg: The lever asset configuration.
        y_range: Range for randomizing Y position relative to wall (min, max).
        z_range: Range for randomizing Z position relative to wall (min, max).
    """
    # Get the assets
    wall: RigidObject = env.scene[wall_cfg.name]
    lever: Articulation = env.scene[lever_cfg.name]

    # Get wall pose (after randomization by the wall event)
    wall_pos = wall.data.root_pos_w[env_ids]
    wall_quat = wall.data.root_quat_w[env_ids]

    # Sample random offsets in YZ plane (in wall's local frame)
    n_envs = len(env_ids)
    y_offset = math_utils.sample_uniform(y_range[0], y_range[1], n_envs, device=env.device)
    z_offset = math_utils.sample_uniform(z_range[0], z_range[1], n_envs, device=env.device)

    # Compute lever center offset in wall's local frame [x, y, z]
    x_offset = -0.5 * (env.cfg.wall_thickness + env.cfg.lever_thickness)
    lever_center_local = torch.stack([torch.full((n_envs,), x_offset, device=env.device), y_offset, z_offset], dim=-1)

    # Transform lever center to world frame
    lever_center_world = wall_pos + math_utils.quat_apply(wall_quat, lever_center_local)

    # Calculate lever rotation: first rotate -90° around Y axis, then apply wall rotation
    # Quaternion for -90° rotation around Y: [0.7071, 0, -0.7071, 0]
    rotation_y_neg90 = torch.tensor([0.7071, 0.0, -0.7071, 0.0], device=env.device)
    rotation_y_neg90 = rotation_y_neg90.unsqueeze(0).expand(n_envs, -1)  # [n_envs, 4]
    lever_quat = math_utils.quat_mul(wall_quat, rotation_y_neg90)

    # Update lever position and orientation
    lever.data.root_pos_w[env_ids] = lever_center_world
    lever.data.root_quat_w[env_ids] = lever_quat

    # Reset velocities
    lever.data.root_lin_vel_w[env_ids] = 0.0
    lever.data.root_ang_vel_w[env_ids] = 0.0

    # Write to simulation
    lever.write_root_pose_to_sim(lever.data.root_pose_w[env_ids], env_ids=env_ids)
    lever.write_root_velocity_to_sim(lever.data.root_vel_w[env_ids], env_ids=env_ids)
