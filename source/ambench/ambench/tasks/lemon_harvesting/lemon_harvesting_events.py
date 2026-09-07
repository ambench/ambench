# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom reset events for the lemon harvesting task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from pxr import Gf, Sdf, UsdPhysics

if TYPE_CHECKING:
    from ambench.tasks.lemon_harvesting.lemon_harvesting_env import LemonHarvesting


def _is_wall_local_position_valid(
    env: LemonHarvesting, candidate: torch.Tensor, threshold: float = 0.08
) -> torch.Tensor:
    """Return whether wall-local fruit positions stay within the usable wall margin."""
    position = candidate.to(device=env.device, dtype=torch.float32)
    margin = position.new_tensor(threshold)
    half_wall_y = position.new_tensor(0.5 * float(env.cfg.wall.size[1]))
    half_wall_z = position.new_tensor(0.5 * float(env.cfg.wall.size[2]))

    inside_margin_y = half_wall_y - position[1].abs()
    inside_margin_z = half_wall_z - position[2].abs()
    return (inside_margin_y > margin) & (inside_margin_z > margin)


def _is_far_enough(candidate_pos: torch.Tensor, accepted_positions: list[torch.Tensor], min_sep: float) -> torch.Tensor:
    """Return whether a wall-local candidate clears existing fruit in the YZ plane."""
    accepted_yz = torch.stack(accepted_positions, dim=0)[:, 1:3]
    candidate_yz = candidate_pos[1:3].unsqueeze(0)
    dist_sq = torch.sum((accepted_yz - candidate_yz) ** 2, dim=-1)
    min_sep_sq = candidate_pos.new_tensor(min_sep * min_sep)
    return torch.all(dist_sq >= min_sep_sq)


def _sample_attach_position(
    env: LemonHarvesting,
    accepted_positions: list[torch.Tensor],
    pos_range: dict[str, tuple[float, float]],
    wall_pos: torch.Tensor,
    wall_quat: torch.Tensor,
    min_sep_yz: float,
    max_tries: int = 200,
) -> torch.Tensor | None:
    """Choose a wall-local lime attachment that stays on the wall and away from neighbors."""
    for _ in range(int(max_tries)):
        sampled_pos = torch.empty(3, device=env.device, dtype=torch.float32)
        sampled_pos[0] = math_utils.sample_uniform(*pos_range["x"], (1,), env.device)[0]
        sampled_pos[1] = math_utils.sample_uniform(*pos_range["y"], (1,), env.device)[0]
        sampled_pos[2] = math_utils.sample_uniform(*pos_range["z"], (1,), env.device)[0]
        candidate = math_utils.quat_apply_inverse(wall_quat.unsqueeze(0), (wall_pos - sampled_pos).unsqueeze(0))[0]
        if not bool(_is_wall_local_position_valid(env, candidate)):
            continue
        if not bool(_is_far_enough(candidate, accepted_positions, min_sep_yz)):
            continue
        return candidate

    return None


def randomize_wall_fruit_attachments(
    env: LemonHarvesting,
    env_ids: torch.Tensor,
    pos_range: dict[str, tuple[float, float]] | None = None,
    break_force: float = 5.0,
    break_torque: float = 5.0,
    min_sep_yz: float = 0.10,
) -> None:
    """Randomize the lemon joint and wall-local lime attachment points."""
    stage = sim_utils.get_current_stage()
    if pos_range is None:
        pos_range = {"y": (-0.2, 0.2), "z": (1.1, 1.5)}
    else:
        pos_range = dict(pos_range)

    wall_pos = torch.tensor(env.cfg.wall_position, device=env.device, dtype=torch.float32)
    wall_quat = torch.tensor(env.cfg.wall_rotation, device=env.device, dtype=torch.float32)
    fruit_x = float(env.cfg.wall_position[0]) - float(env.cfg.wall_attach_x_offset)
    pos_range.setdefault("x", (fruit_x, fruit_x))
    max_scene = int(env.cfg.max_limes_in_scene)

    for env_index in env_ids.tolist():
        env_path = f"/World/envs/env_{env_index}"

        # Limes are reset as kinematic bodies, so old joint prims must be removed first.
        for lime_index in range(1, max_scene + 1):
            lime_joint_path = f"{env_path}/lime_wall_fixed_joint_{lime_index:02d}"
            lime_joint_prim = stage.GetPrimAtPath(lime_joint_path)
            if lime_joint_prim.IsValid():
                stage.RemovePrim(lime_joint_prim.GetPath())

        # Attach the lemon to a randomized wall-local position.
        sampled_lemon_pos = torch.empty(3, device=env.device, dtype=torch.float32)
        sampled_lemon_pos[0] = math_utils.sample_uniform(*pos_range["x"], (1,), env.device)[0]
        sampled_lemon_pos[1] = math_utils.sample_uniform(*pos_range["y"], (1,), env.device)[0]
        sampled_lemon_pos[2] = math_utils.sample_uniform(*pos_range["z"], (1,), env.device)[0]
        lemon_pos = math_utils.quat_apply_inverse(wall_quat.unsqueeze(0), (wall_pos - sampled_lemon_pos).unsqueeze(0))[
            0
        ]

        wall_prim_path = f"{env_path}/Wall"
        lemon_prim_path = f"{env_path}/lemon/lemon_01"
        lemon_joint_path = f"{env_path}/lemon_wall_fixed_joint"
        lemon_joint = UsdPhysics.FixedJoint.Define(stage, lemon_joint_path)
        lemon_joint.CreateBody0Rel().SetTargets([Sdf.Path(wall_prim_path)])
        lemon_joint.CreateBody1Rel().SetTargets([Sdf.Path(lemon_prim_path)])
        lemon_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        lemon_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*lemon_pos.tolist()))
        lemon_joint.CreateBreakForceAttr().Set(float(break_force))
        lemon_joint.CreateBreakTorqueAttr().Set(float(break_torque))

        # Store non-overlapping wall-local positions for the kinematic limes.
        accepted_positions: list[torch.Tensor] = [lemon_pos]

        for lime_index in range(1, max_scene + 1):
            lime_prim_path = f"{env_path}/lime{lime_index:02d}/lime01"
            lime_prim = stage.GetPrimAtPath(lime_prim_path)
            if not lime_prim.IsValid():
                continue

            lime_pos = _sample_attach_position(
                env=env,
                accepted_positions=accepted_positions,
                pos_range=pos_range,
                wall_pos=wall_pos,
                wall_quat=wall_quat,
                min_sep_yz=min_sep_yz,
            )
            if lime_pos is None:
                raise RuntimeError(
                    "Failed to place lime without overlap after random sampling. "
                    f"env_index={env_index}, slot_index={lime_index}, pos_range={pos_range}, "
                    f"min_sep_yz={min_sep_yz}."
                )
            accepted_positions.append(lime_pos)
            env._lime_wall_local_pos[env_index, lime_index - 1] = lime_pos
