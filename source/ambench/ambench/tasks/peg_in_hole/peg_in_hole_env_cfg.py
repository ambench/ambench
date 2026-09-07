# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR  # noqa: F401

from ambench.assets.object_specs import (
    HOLE_DEFAULT_SIDE_LENGTH,
    HOLE_DEFAULT_THICKNESS,
    PEG_DEFAULT_LENGTH,
    PEG_DEFAULT_RADIUS,
)
from ambench.scenes import WallSceneCfg, spawn_wall
from ambench.tasks.base.base_env_cfg import BaseEnvCfg
from ambench.tasks.base.robot_profiles import (
    EE_ABS_L1,
    EE_ABS_PID,
    FA_HEXA_ABS_L1,
    FA_HEXA_ABS_MPC,
    FA_HEXA_ABS_PID,
    OMNI_HEXA_ABS_PID,
    UA_HEXA_ABS_PID,
    UA_QUAD_ABS_PID,
)
from ambench.tasks.peg_in_hole import peg_in_hole_events
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class EventCfg:
    """
    Configuration for domain randomization.
    """

    # Randomize wall position and orientation
    # Wall and hole rotate together to simulate a tilted wall
    randomize_wall_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),  # Wall can move ±0.2m in X
                "y": (-0.2, 0.2),  # Wall can move ±0.2m in Y
                "pitch": (-np.pi / 6.0, np.pi / 6.0),  # 30 deg pitch
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wall"),
        },
    )

    # Randomize hole position on wall surface (YZ plane relative to wall)
    randomize_hole_position = EventTerm(
        func=peg_in_hole_events.randomize_hole_position_on_wall,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "hole_cfg": SceneEntityCfg("hole_object"),
            "y_range": (-0.3, 0.3),  # Hole can move ±0.3m in Y on wall
            "z_range": (-0.3, 0.3),  # Hole can move ±0.3m in Z on wall
        },
    )

    # Randomize wall texture/material
    # wall_texture_randomizer = EventTerm(
    #     func=mdp.randomize_visual_texture_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("wall"),
    #         "texture_paths": [
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Adobe_Brick/Adobe_Brick_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Wall_Brown/Brick_Wall_Brown_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Wall_Red/Brick_Wall_Red_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Rough/Concrete_Rough_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block/Concrete_Block_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/Cardboard/Cardboard_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
    #             f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut/Walnut_BaseColor.png",
    #         ],
    #         "event_name": "wall_texture_randomizer",
    #     },
    # )

    # Randomize hole scale
    randomize_hole_scale = EventTerm(
        func=peg_in_hole_events.randomize_hole_scale,  # this custom function scales only in Y and Z, uniformly
        mode="prestartup",
        params={
            "scale_range": (0.75, 1.5),
        },
    )


@configclass
class PegInHoleEnvDefaultCfg(BaseEnvCfg):
    # env
    episode_length_s = 20

    # Environment-specific parameters

    # environment assets - wall and hole are independent assets at the same level
    wall_position = (2.0, 0.0, 1.0)  # 2m in front of robot, 1m above ground
    wall_rotation = (1, 0, 0, 0)
    wall: WallSceneCfg = WallSceneCfg()  # Default wall with size (0.1, 2.0, 2.0)
    wall_thickness = wall.size[0]
    wall_cfg: RigidObjectCfg = spawn_wall(
        prim_path="/World/envs/env_.*/Wall",
        cfg=wall,
        translation=wall_position,
        orientation=wall_rotation,
    )

    # Hole positioned at same location as wall, facing same direction
    # Offset by half thickness so hole is on the wall surface
    hole_side_length = HOLE_DEFAULT_SIDE_LENGTH
    hole_thickness = HOLE_DEFAULT_THICKNESS
    hole_offset_x = -0.5 * (wall_thickness + hole_thickness)
    hole_position = (
        wall_position[0] + hole_offset_x,
        wall_position[1],
        wall_position[2],
    )

    hole_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/hole.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=hole_position,
            rot=wall_rotation,
        ),
    )

    # Peg object
    peg_length = PEG_DEFAULT_LENGTH
    peg_radius = PEG_DEFAULT_RADIUS
    peg_mass = 0.01
    peg_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/peg.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=peg_mass),
        ),
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    peg_insertion_depth_min = 0.00  # Minimum depth the peg tip must reach into the hole (in meters)

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["goal_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class PegInHoleEnvEEAbsPIDCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class PegInHoleEnvEEAbsL1Cfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class PegInHoleEnvOmniHexaAbsPIDCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class PegInHoleEnvFAHexaAbsPIDCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class PegInHoleEnvFAHexaAbsL1Cfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class PegInHoleEnvUAHexaAbsPIDCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class PegInHoleEnvFAHexaAbsMPCCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class PegInHoleEnvUAQuadAbsPIDCfg(PegInHoleEnvDefaultCfg):
    """Peg in hole environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
