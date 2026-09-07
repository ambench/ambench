# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Parametric room scene helpers."""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

ROOM_FLOOR_SURFACE_NAME = "floor"
ROOM_CEILING_SURFACE_NAME = "ceiling"
ROOM_WALL_SURFACE_NAMES = ("wall_north", "wall_south", "wall_east", "wall_west")
ROOM_SURFACE_GEOMETRY_REL_PATH = "geometry/mesh"


@sim_utils.clone
def spawn_room(
    prim_path: str,
    cfg: RoomSceneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a nested static room root with mesh-cuboid floor, wall, and ceiling surfaces."""
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    room_prim = sim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )

    width, length = cfg.size
    wall_north, wall_south, wall_east, wall_west = ROOM_WALL_SURFACE_NAMES
    surface_specs = [
        (
            ROOM_FLOOR_SURFACE_NAME,
            (width, length, cfg.floor_thickness),
            (0.0, 0.0, -0.5 * cfg.floor_thickness),
            cfg.floor_visual_material,
        ),
        (
            wall_north,
            (width, cfg.wall_thickness, cfg.wall_height),
            (0.0, 0.5 * length + 0.5 * cfg.wall_thickness, 0.5 * cfg.wall_height),
            cfg.wall_visual_material,
        ),
        (
            wall_south,
            (width, cfg.wall_thickness, cfg.wall_height),
            (0.0, -0.5 * length - 0.5 * cfg.wall_thickness, 0.5 * cfg.wall_height),
            cfg.wall_visual_material,
        ),
        (
            wall_east,
            (cfg.wall_thickness, length, cfg.wall_height),
            (0.5 * width + 0.5 * cfg.wall_thickness, 0.0, 0.5 * cfg.wall_height),
            cfg.wall_visual_material,
        ),
        (
            wall_west,
            (cfg.wall_thickness, length, cfg.wall_height),
            (-0.5 * width - 0.5 * cfg.wall_thickness, 0.0, 0.5 * cfg.wall_height),
            cfg.wall_visual_material,
        ),
    ]

    for surface_name, size, center, visual_material in surface_specs:
        surface_cfg = sim_utils.MeshCuboidCfg(
            size=size,
            collision_props=cfg.collision_props,
            visual_material=visual_material,
            physics_material=cfg.physics_material,
        )
        surface_cfg.func(f"{prim_path}/{surface_name}", surface_cfg, translation=center)

    if cfg.ceiling_enabled:
        ceiling_cfg = sim_utils.MeshCuboidCfg(
            size=(width, length, cfg.ceiling_thickness),
            visible=cfg.ceiling_visible,
            collision_props=cfg.collision_props if cfg.ceiling_collision_enabled else None,
            visual_material=cfg.wall_visual_material,
            physics_material=cfg.physics_material if cfg.ceiling_collision_enabled else None,
        )
        ceiling_cfg.func(
            f"{prim_path}/{ROOM_CEILING_SURFACE_NAME}",
            ceiling_cfg,
            translation=(0.0, 0.0, cfg.wall_height + 0.5 * cfg.ceiling_thickness),
        )

    return room_prim


@configclass
class RoomSceneCfg(SpawnerCfg):
    """Configuration and spawner for a nested parametric navigation room."""

    func: Callable = spawn_room
    size: tuple[float, float] = (15.0, 15.0)
    wall_height: float = 2.7
    wall_thickness: float = 0.05
    floor_thickness: float = 0.10
    ceiling_enabled: bool = True
    ceiling_visible: bool = False
    ceiling_collision_enabled: bool = True
    ceiling_thickness: float = 0.10

    # Fallback visuals used before room-surface MDL randomization, or when that event is disabled.
    floor_visual_material: sim_utils.PreviewSurfaceCfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.42, 0.44, 0.43),
        roughness=0.8,
    )
    wall_visual_material: sim_utils.PreviewSurfaceCfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.78, 0.80, 0.77),
        roughness=0.85,
    )
    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    collision_props: sim_utils.CollisionPropertiesCfg | None = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
    )
