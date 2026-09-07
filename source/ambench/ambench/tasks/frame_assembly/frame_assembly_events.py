# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for FrameAssembly environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ambench.tasks.frame_assembly.frame_assembly_env import FrameAssembly


def randomize_wall_and_pegs(
    env: FrameAssembly,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    peg_1_cfg: SceneEntityCfg,
    peg_2_cfg: SceneEntityCfg,
    peg_3_cfg: SceneEntityCfg,
    peg_4_cfg: SceneEntityCfg,
    wall_pose_range: dict,
):
    """Randomize wall pose and move pegs to match the wall frame."""
    # Resolve assets once for batched state updates.
    wall: RigidObject = env.scene[wall_cfg.name]
    pegs = [
        env.scene[peg_1_cfg.name],
        env.scene[peg_2_cfg.name],
        env.scene[peg_3_cfg.name],
        env.scene[peg_4_cfg.name],
    ]

    n_envs = len(env_ids)

    # Match mdp.reset_root_state_uniform: use default root state as reset baseline.
    root_states = wall.data.default_root_state[env_ids].clone()

    # Sample pose deltas in xyz + rpy order.
    range_list = [wall_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device, dtype=root_states.dtype)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n_envs, 6), device=env.device)

    # Compose randomized wall pose in world frame from default state + env origin.
    rot_offset_quat = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    env_origins = env.scene.env_origins[env_ids]

    wall_new_pos = root_states[:, 0:3] + env_origins + rand_samples[:, 0:3]
    wall_new_quat = math_utils.quat_mul(root_states[:, 3:7], rot_offset_quat)

    # Update wall buffers before writing to simulation.
    wall.data.root_pos_w[env_ids] = wall_new_pos
    wall.data.root_quat_w[env_ids] = wall_new_quat

    # Keep wall static at reset by zeroing linear and angular velocity.
    wall.data.root_lin_vel_w[env_ids] = 0.0
    wall.data.root_ang_vel_w[env_ids] = 0.0

    # Apply wall pose and zero velocity to sim.
    wall.write_root_pose_to_sim(wall.data.root_pose_w[env_ids], env_ids=env_ids)
    wall.write_root_velocity_to_sim(wall.data.root_vel_w[env_ids], env_ids=env_ids)

    # Keep peg offsets fixed in wall-local coordinates.
    peg_positions_local = env.cfg.peg_local_offsets

    for peg, peg_pos_local in zip(pegs, peg_positions_local):
        peg_local = torch.tensor(peg_pos_local, device=env.device, dtype=wall_new_pos.dtype).expand(n_envs, -1)

        # Transform local peg offsets into world frame from randomized wall pose.
        peg_world = wall_new_pos + math_utils.quat_apply(wall_new_quat, peg_local)

        # Pegs inherit wall orientation and remain kinematic.
        peg.data.root_pos_w[env_ids] = peg_world
        peg.data.root_quat_w[env_ids] = wall_new_quat
        peg.data.root_lin_vel_w[env_ids] = 0.0
        peg.data.root_ang_vel_w[env_ids] = 0.0

        # Write peg state after wall update to preserve wall-relative placement.
        peg.write_root_pose_to_sim(peg.data.root_pose_w[env_ids], env_ids=env_ids)
        peg.write_root_velocity_to_sim(peg.data.root_vel_w[env_ids], env_ids=env_ids)
