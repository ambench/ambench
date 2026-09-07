# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for PegInHole environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from omni.usd import get_context
from pxr import Gf, Sdf, UsdGeom, Vt

if TYPE_CHECKING:
    from ambench.tasks.peg_in_hole.peg_in_hole_env import PegInHole


def randomize_hole_position_on_wall(
    env: PegInHole,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize hole position on the wall surface (YZ plane relative to wall).

    The hole stays at the same X distance from the wall (on the surface) but can
    move in the YZ plane. The hole also inherits the wall's orientation so they
    stay aligned when the wall is tilted.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize.
        wall_cfg: The wall asset configuration.
        hole_cfg: The hole asset configuration.
        y_range: Range for randomizing Y position relative to wall (min, max).
        z_range: Range for randomizing Z position relative to wall (min, max).
    """
    # Get the assets
    wall: RigidObject = env.scene[wall_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]

    # Get wall pose (after randomization by the wall event)
    wall_pos = wall.data.root_pos_w[env_ids]
    wall_quat = wall.data.root_quat_w[env_ids]

    # Sample random offsets in YZ plane (in wall's local frame)
    n_envs = len(env_ids)
    y_offset = math_utils.sample_uniform(y_range[0], y_range[1], n_envs, device=env.device)
    z_offset = math_utils.sample_uniform(z_range[0], z_range[1], n_envs, device=env.device)

    # Compute hole center offset in wall's local frame [x, y, z]
    x_offset = -0.5 * (env.cfg.wall_thickness + env.cfg.hole_thickness)
    hole_center_local = torch.stack([torch.full((n_envs,), x_offset, device=env.device), y_offset, z_offset], dim=-1)

    # Transform hole center to world frame
    hole_center_world = wall_pos + math_utils.quat_apply(wall_quat, hole_center_local)

    # Update hole position and orientation
    hole.data.root_pos_w[env_ids] = hole_center_world
    hole.data.root_quat_w[env_ids] = wall_quat

    # The hole is kinematic, so only its pose may be written.
    hole.write_root_pose_to_sim(hole.data.root_pose_w[env_ids], env_ids=env_ids)


def randomize_hole_scale(
    env: PegInHole,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
):
    """Randomize the hole scale in Y and Z directions uniformly.

    This function is similar to `mdp.randomize_rigid_body_scale` but only scales
    the Y and Z dimensions of the hole, keeping X scale fixed. This is useful for
    the peg-in-hole task where we want to vary the hole size without changing its
    thickness.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'prestartup' mode."
        )

    # get the hole
    asset_cfg = SceneEntityCfg("hole_object")
    asset: RigidObject = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    # only sample once, and apply in y and z direction
    rand_scale_yz = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
    rand_samples = torch.cat([torch.ones((len(env_ids), 1), device="cpu"), rand_scale_yz, rand_scale_yz], dim=-1)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id]
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec,
                        UsdGeom.Tokens.xformOpOrder,
                        Sdf.ValueTypeNames.TokenArray,
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    if not hasattr(env, "hole_side_length_buf"):
        env.hole_side_length_buf = torch.full(
            (env.scene.num_envs,),
            float(env.cfg.hole_side_length),
            device=env.device,
            dtype=torch.float32,
        )

    env_ids_device = env_ids.to(device=env.device)
    env.hole_side_length_buf[env_ids_device] = float(env.cfg.hole_side_length) * rand_scale_yz.squeeze(-1).to(
        device=env.device
    )
