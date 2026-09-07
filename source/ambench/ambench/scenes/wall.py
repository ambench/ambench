# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

##
# Scene definition
##


@configclass
class WallSceneCfg:
    """Configuration for the wall scene with an aerial manipulator."""

    size: tuple[float, float, float] = (0.1, 2.0, 2.0)

    visual_material: sim_utils.MdlFileCfg = sim_utils.MdlFileCfg(
        mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block.mdl",
        project_uvw=True,
    )

    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )


def spawn_wall(
    prim_path: str,
    cfg: WallSceneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a wall with given configuration.

    Args:
        prim_path (str): The prim path where the wall will be spawned.
        cfg (WallSceneCfg): The wall scene configuration.
        translation (tuple[float, float, float] | None): The wall translation.
        orientation (tuple[float, float, float, float) | None): The wall orientation.
        **kwargs: Additional keyword arguments for RigidObjectCfg.
    Returns:
        RigidObjectCfg: The wall rigid object configuration.
    """
    if translation is None:
        translation = (0.0, 0.0, 0.0)
    if orientation is None:
        orientation = (1.0, 0.0, 0.0, 0.0)

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCuboidCfg(
            size=cfg.size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=cfg.visual_material,
            physics_material=cfg.physics_material,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=translation, rot=orientation),
        **kwargs,
    )
