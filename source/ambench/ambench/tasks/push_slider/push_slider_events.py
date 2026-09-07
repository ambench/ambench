# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for PushSlider environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ambench.tasks.push_slider.push_slider_env import PushSlider


def randomize_slider_position_on_wall(
    env: PushSlider,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    slider_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize slider position on the wall surface (YZ plane relative to wall).

    The slider stays at the same X distance from the wall (on the surface) but can
    move in the YZ plane. The slider also inherits the wall's orientation so they
    stay aligned when the wall is tilted.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        wall_cfg: The wall asset configuration.
        slider_cfg: The slider asset configuration.
        y_range: Range for randomizing Y position relative to wall (min, max).
        z_range: Range for randomizing Z position relative to wall (min, max).
    """
    # Get the assets
    wall: RigidObject = env.scene[wall_cfg.name]
    slider: Articulation = env.scene[slider_cfg.name]

    # Get wall pose (after randomization by the wall event)
    wall_pos = wall.data.root_pos_w[env_ids]
    wall_quat = wall.data.root_quat_w[env_ids]

    # Sample random offsets in YZ plane (in wall's local frame)
    n_envs = len(env_ids)
    y_offset = math_utils.sample_uniform(y_range[0], y_range[1], n_envs, device=env.device)
    z_offset = math_utils.sample_uniform(z_range[0], z_range[1], n_envs, device=env.device)

    # Compute slider center offset in wall's local frame [x, y, z]
    # Assuming slider origin is at its back which touches the wall
    # Offset by half thickness so slider is on the wall surface
    x_offset = -0.5 * env.cfg.wall_thickness
    slider_center_local = torch.stack([torch.full((n_envs,), x_offset, device=env.device), y_offset, z_offset], dim=-1)

    # Transform slider center to world frame
    slider_center_world = wall_pos + math_utils.quat_apply(wall_quat, slider_center_local)

    # Rotate slider 180 degrees around Z axis relative to wall
    # Quaternion for 180 deg rotation around Z: (0, 0, 0, 1) in (w, x, y, z)
    rot_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(n_envs, 1)
    slider_quat = math_utils.quat_mul(wall_quat, rot_quat)

    # Update slider position and orientation
    slider.data.root_pos_w[env_ids] = slider_center_world
    slider.data.root_quat_w[env_ids] = slider_quat

    # Reset velocities
    slider.data.root_lin_vel_w[env_ids] = 0.0
    slider.data.root_ang_vel_w[env_ids] = 0.0

    # Write to simulation
    slider.write_root_pose_to_sim(slider.data.root_pose_w[env_ids], env_ids=env_ids)
    slider.write_root_velocity_to_sim(slider.data.root_vel_w[env_ids], env_ids=env_ids)


def randomize_slider_joint_friction(
    env: PushSlider,
    env_ids: torch.Tensor,
    slider_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] = (0.5, 2.5),
    operation: str = "scale",
):
    """Randomize slider joint friction coefficient.

    This function randomizes the friction coefficient of the slider joint to simulate
    different friction conditions.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize friction for.
        slider_cfg: The slider asset configuration.
        friction_distribution_params: Range (min, max) for friction scaling factor (default: 0.5-2.5).
        operation: Operation mode - "scale" scales existing friction, "set" sets absolute values.
    """
    # Get the slider asset
    slider: Articulation = env.scene[slider_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(env.device)

    # Get joint names
    joint_names = slider_cfg.joint_names if hasattr(slider_cfg, "joint_names") else ["SliderJoint"]

    # Find joint indices using the exact same pattern as push_slider_env.py: find_joints(name)[0][0]
    # This extracts the first joint index from the first environment
    joint_ids = []
    for joint_name in joint_names:
        joint_indices, _ = slider.find_joints(joint_name)
        joint_ids.append(int(joint_indices[0]))

    # Convert to tensor - shape should be (num_joints,) = (1,)
    joint_ids_tensor = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
    joint_id = joint_ids_tensor[0].item()  # Get scalar value

    # Index with single joint ID to get shape (num_envs, 1)
    current_friction = slider.data.joint_friction_coeff[env_ids, joint_id : joint_id + 1].clone()
    # Sample random scaling factors
    n_envs = len(env_ids)
    n_joints = 1  # Single joint

    if operation == "scale":
        # Scale current friction values
        scale_factors = math_utils.sample_uniform(
            friction_distribution_params[0],
            friction_distribution_params[1],
            (n_envs, n_joints),
            device=env.device,
        )
        new_friction = current_friction * scale_factors
    else:  # "set"
        # Set absolute friction values
        new_friction = math_utils.sample_uniform(
            friction_distribution_params[0],
            friction_distribution_params[1],
            (n_envs, n_joints),
            device=env.device,
        )

    # Write to simulation - shape should be (num_envs, num_joints) = (num_envs, 1)
    slider.write_joint_friction_coefficient_to_sim(
        joint_friction_coeff=new_friction,
        joint_dynamic_friction_coeff=new_friction,
        joint_ids=joint_ids_tensor,
        env_ids=env_ids,
    )
