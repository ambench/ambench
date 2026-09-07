# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom event functions for PressButton environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from omni.usd import get_context
from pxr import Gf, Sdf, UsdGeom, Vt

if TYPE_CHECKING:
    from ambench.tasks.press_button.press_button_env import PressButton


def randomize_button_position_on_wall(
    env: PressButton,
    env_ids: torch.Tensor,
    wall_cfg: SceneEntityCfg,
    button_cfg: SceneEntityCfg,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
):
    """Randomize button position on the wall surface (YZ plane relative to wall).

    The button stays at the same X distance from the wall surface but can move in
    the wall-local YZ plane. The button also inherits the wall orientation so the
    two remain aligned when the wall is tilted.
    """
    # Get the assets
    wall: RigidObject = env.scene[wall_cfg.name]
    button: Articulation = env.scene[button_cfg.name]

    # Get wall pose after wall randomization
    wall_pos = wall.data.root_pos_w[env_ids]
    wall_quat = wall.data.root_quat_w[env_ids]

    # Sample random offsets in the wall-local YZ plane
    n_envs = len(env_ids)
    y_offset = math_utils.sample_uniform(y_range[0], y_range[1], n_envs, device=env.device)
    z_offset = math_utils.sample_uniform(z_range[0], z_range[1], n_envs, device=env.device)

    # Keep the button attached to the wall surface along local X
    x_offset = -0.5 * env.cfg.wall_thickness
    button_center_local = torch.stack([torch.full((n_envs,), x_offset, device=env.device), y_offset, z_offset], dim=-1)

    # Transform button center to world frame
    button_center_world = wall_pos + math_utils.quat_apply(wall_quat, button_center_local)

    # Update button pose to follow the wall
    button.data.root_pos_w[env_ids] = button_center_world
    button.data.root_quat_w[env_ids] = wall_quat

    # Reset velocities after teleporting the button
    button.data.root_lin_vel_w[env_ids] = 0.0
    button.data.root_ang_vel_w[env_ids] = 0.0

    # Write randomized state to simulation
    button.write_root_pose_to_sim(button.data.root_pose_w[env_ids], env_ids=env_ids)
    button.write_root_velocity_to_sim(button.data.root_vel_w[env_ids], env_ids=env_ids)


def randomize_button_scale(
    env: PressButton,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
):
    """Randomize the button scale in Y and Z directions.

    This mirrors `mdp.randomize_rigid_body_scale` but keeps the X scale fixed so
    the button thickness stays unchanged while the visible face size varies.
    """
    # Scaling must happen before simulation starts.
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'prestartup' mode."
        )

    # Get the button articulation
    asset_cfg = SceneEntityCfg("button")
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Acquire stage and resolve cloned prim paths
    stage = get_context().get_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # Sample one scale and apply it to both Y and Z
    rand_scale_yz = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")

    rand_samples = torch.cat(
        [
            torch.ones((len(env_ids), 1), device="cpu"),
            rand_scale_yz,
            rand_scale_yz,
        ],
        dim=-1,
    )
    rand_samples = rand_samples.tolist()

    # Edit USD scale attributes in a single change block
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # Path to the cloned button prim
            prim_path = prim_paths[env_id]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # Create the scale op if the asset does not already have one
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # Apply the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # Ensure the xform op order stays valid if we created the scale op
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec,
                        UsdGeom.Tokens.xformOpOrder,
                        Sdf.ValueTypeNames.TokenArray,
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
