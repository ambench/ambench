# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for WipeWindow environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ambench.tasks.wipe_window.wipe_window_env import WipeWindow


def set_dots_position_on_window(
    env: WipeWindow,
    env_ids: torch.Tensor,
    window_cfg: SceneEntityCfg,
):
    """Set random (non-connected) stain dot positions on the window surface."""
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    num_dots = env.cfg.num_dots
    window: RigidObject = env.scene[window_cfg.name]

    window_pos = window.data.root_pos_w[env_ids]
    window_quat = window.data.root_quat_w[env_ids]

    # Place dots on the back surface of the glass in window-local coordinates.
    glass_thickness = env.cfg.window_thickness
    frame_thickness = 0.05
    glass_offset_x = 0.5 * (frame_thickness - glass_thickness)
    dot_x_local = glass_offset_x - glass_thickness * 0.5

    n_envs = len(env_ids)
    dot_radius = 0.01
    window_width = env.cfg.window_width
    window_height = env.cfg.window_height
    position_margin = 0.40

    y_min = -window_width * 0.5 + position_margin + dot_radius
    y_max = window_width * 0.5 - position_margin - dot_radius
    z_min = -window_height * 0.5 + position_margin + dot_radius
    z_max = window_height * 0.5 - position_margin - dot_radius

    dot_y_local = math_utils.sample_uniform(y_min, y_max, (n_envs, num_dots), device=env.device)
    dot_z_local = math_utils.sample_uniform(z_min, z_max, (n_envs, num_dots), device=env.device)
    dot_x_local_tensor = torch.full((n_envs, num_dots), dot_x_local, device=env.device)

    dot_pos_local = torch.stack([dot_x_local_tensor, dot_y_local, dot_z_local], dim=-1)

    # Transform all local dot positions to world frame in one batch.
    dot_pos_world = window_pos.unsqueeze(1) + math_utils.quat_apply(
        window_quat.unsqueeze(1).expand(-1, num_dots, -1).reshape(-1, 4),
        dot_pos_local.reshape(-1, 3),
    ).reshape(n_envs, num_dots, 3)

    for i in range(num_dots):
        dot_name = f"dot_{i}_object"
        if dot_name not in env.scene.rigid_objects:
            continue

        dot: RigidObject = env.scene[dot_name]

        dot.data.root_pos_w[env_ids] = dot_pos_world[:, i, :]
        dot.data.root_quat_w[env_ids] = window_quat
        # Dots are kinematic, so only their poses may be written.
        dot.write_root_pose_to_sim(dot.data.root_pose_w[env_ids], env_ids=env_ids)
