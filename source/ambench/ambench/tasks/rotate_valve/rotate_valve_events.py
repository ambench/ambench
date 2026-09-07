# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for RotateValve environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ambench.tasks.rotate_valve.rotate_valve_env import RotateValve


def randomize_valve_position_on_wall(
    env: RotateValve,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    valve_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    """Randomize the valve in the wall-local YZ plane."""
    wall: RigidObject = env.scene[wall_cfg.name]
    valve: Articulation = env.scene[valve_cfg.name]

    # The wall event runs first, so this pose includes its randomization.
    wall_pos = wall.data.root_pos_w[env_ids]
    wall_quat = wall.data.root_quat_w[env_ids]

    n_envs = len(env_ids)
    y_offset = math_utils.sample_uniform(y_range[0], y_range[1], n_envs, device=env.device)
    z_offset = math_utils.sample_uniform(z_range[0], z_range[1], n_envs, device=env.device)

    x_offset = env.cfg.valve_offset_x
    valve_center_local = torch.stack([torch.full((n_envs,), x_offset, device=env.device), y_offset, z_offset], dim=-1)

    valve_center_world = wall_pos + math_utils.quat_apply(wall_quat, valve_center_local)

    valve.data.root_pos_w[env_ids] = valve_center_world
    valve_default_quaternion = torch.tensor(env.cfg.valve_object_cfg.init_state.rot, device=env.device).expand(
        n_envs, -1
    )
    valve_quat_align_wall = math_utils.quat_mul(wall_quat, valve_default_quaternion)
    valve.data.root_quat_w[env_ids] = valve_quat_align_wall

    valve.data.root_lin_vel_w[env_ids] = 0.0
    valve.data.root_ang_vel_w[env_ids] = 0.0

    valve.write_root_pose_to_sim(valve.data.root_pose_w[env_ids], env_ids=env_ids)
    valve.write_root_velocity_to_sim(valve.data.root_vel_w[env_ids], env_ids=env_ids)
