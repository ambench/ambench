# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Event terms for parametric room scenes."""

from __future__ import annotations

import logging

import isaaclab.sim as sim_utils
import torch
from isaaclab.managers import ManagerTermBase

from .room import (
    ROOM_FLOOR_SURFACE_NAME,
    ROOM_SURFACE_GEOMETRY_REL_PATH,
    ROOM_WALL_SURFACE_NAMES,
)

logger = logging.getLogger(__name__)


class RandomizeRoomSurroundings(ManagerTermBase):
    """Randomize visual materials on a parametric room's floor and walls."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        stage = sim_utils.get_current_stage()
        material_root = cfg.params["material_root"]
        if not stage.GetPrimAtPath("/World/Looks").IsValid():
            sim_utils.create_prim("/World/Looks", prim_type="Scope", stage=stage)
        if not stage.GetPrimAtPath(material_root).IsValid():
            sim_utils.create_prim(material_root, prim_type="Scope", stage=stage)

        self.floor_material_paths = []
        for material_id, material_cfg in enumerate(cfg.params.get("floor_material_cfgs", ())):
            material_path = f"{material_root}/floor_{material_id:03d}"
            if not stage.GetPrimAtPath(material_path).IsValid():
                material_cfg.func(material_path, material_cfg)
            self.floor_material_paths.append(material_path)

        self.wall_material_paths = []
        for material_id, material_cfg in enumerate(cfg.params.get("wall_material_cfgs", ())):
            material_path = f"{material_root}/wall_{material_id:03d}"
            if not stage.GetPrimAtPath(material_path).IsValid():
                material_cfg.func(material_path, material_cfg)
            self.wall_material_paths.append(material_path)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        material_root: str,
        floor_material_cfgs: tuple[sim_utils.MdlFileCfg, ...],
        wall_material_cfgs: tuple[sim_utils.MdlFileCfg, ...],
    ):
        del material_root, floor_material_cfgs, wall_material_cfgs

        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        if not self.floor_material_paths and not self.wall_material_paths:
            return

        stage = sim_utils.get_current_stage()
        floor_indices = (
            torch.randint(len(self.floor_material_paths), (len(env_ids),), device=env.device).tolist()
            if self.floor_material_paths
            else []
        )
        wall_indices = (
            torch.randint(len(self.wall_material_paths), (len(env_ids),), device=env.device).tolist()
            if self.wall_material_paths
            else []
        )

        for local_idx, env_id in enumerate(env_ids.tolist()):
            room_prim_path = env.cfg.room_prim_path
            concrete_env_path = f"{env.scene.env_ns}/env_{env_id}"
            if "{ENV_REGEX_NS}" in room_prim_path:
                room_prim_path = room_prim_path.replace("{ENV_REGEX_NS}", concrete_env_path)
            elif f"{env.scene.env_ns}/env_.*/" in room_prim_path:
                room_prim_path = room_prim_path.replace(f"{env.scene.env_ns}/env_.*/", f"{concrete_env_path}/", 1)
            else:
                room_prim_path = room_prim_path.replace("env_.*", f"env_{env_id}", 1)

            if self.floor_material_paths:
                target_path = f"{room_prim_path}/{ROOM_FLOOR_SURFACE_NAME}/{ROOM_SURFACE_GEOMETRY_REL_PATH}"
                if stage.GetPrimAtPath(target_path).IsValid():
                    sim_utils.bind_visual_material(target_path, self.floor_material_paths[floor_indices[local_idx]])
                else:
                    logger.warning("Skipping room floor material binding for missing prim: %s", target_path)

            if self.wall_material_paths:
                wall_material_path = self.wall_material_paths[wall_indices[local_idx]]
                for wall_surface_name in ROOM_WALL_SURFACE_NAMES:
                    target_path = f"{room_prim_path}/{wall_surface_name}/{ROOM_SURFACE_GEOMETRY_REL_PATH}"
                    if stage.GetPrimAtPath(target_path).IsValid():
                        sim_utils.bind_visual_material(target_path, wall_material_path)
                    else:
                        logger.warning("Skipping room wall material binding for missing prim: %s", target_path)
