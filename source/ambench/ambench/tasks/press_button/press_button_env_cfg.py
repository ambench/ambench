# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR  # noqa: F401

from ambench.assets.object_specs import BUTTON_DEFAULT_RADIUS
from ambench.scenes import WallSceneCfg, spawn_wall
from ambench.tasks.base.base_env_cfg import BaseEnvCfg
from ambench.tasks.base.robot_profiles import (
    EE_ABS_L1,
    EE_ABS_PID,
    FA_HEXA_ABS_L1,
    FA_HEXA_ABS_MPC,
    FA_HEXA_ABS_PID,
    FA_HEXA_BASE_JOINT_ABS_L1,
    FA_HEXA_BASE_JOINT_ABS_PID,
    OMNI_HEXA_ABS_PID,
    UA_HEXA_ABS_PID,
    UA_QUAD_ABS_PID,
)
from ambench.tasks.press_button import press_button_events
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class EventCfg:
    """
    Configuration for domain randomization.
    """

    # Randomize wall position and orientation
    randomize_wall_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),  # Wall can move ±0.2m in X
                "y": (-0.2, 0.2),  # Wall can move ±0.2m in Y
                "pitch": (-0.2, 0.2),  # Wall can rotate ±0.2 rad in pitch
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wall"),
        },
    )

    # Randomize button position on wall surface (YZ plane relative to wall)
    randomize_button_position = EventTerm(
        func=press_button_events.randomize_button_position_on_wall,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "button_cfg": SceneEntityCfg("button"),
            "y_range": (-0.3, 0.3),  # Button can move ±0.3m in Y on wall
            "z_range": (-0.3, 0.3),  # Button can move ±0.3m in Z on wall
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

    # Randomize button scale
    randomize_button_scale = EventTerm(
        func=press_button_events.randomize_button_scale,
        mode="prestartup",
        params={
            "scale_range": (0.5, 1.0),  # Scale button between 0.5x to 1.0x
        },
    )


@configclass
class PressButtonEnvDefaultCfg(BaseEnvCfg):
    # env
    episode_length_s = 20

    # Environment-specific parameters

    # environment assets - wall and button are independent assets at the same level
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

    # Button positioned at same location as wall, facing same direction
    # Offset by half thickness so button is on the wall surface
    button_offset_x = -0.5 * wall_thickness  # Position button on wall surface
    button_position = (
        wall_position[0] + button_offset_x,
        wall_position[1],
        wall_position[2],
    )
    button_radius = BUTTON_DEFAULT_RADIUS
    button_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Button",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/push_button.usd",
            scale=(
                1.0,
                button_radius / BUTTON_DEFAULT_RADIUS,
                button_radius / BUTTON_DEFAULT_RADIUS,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=button_position,
            rot=wall_rotation,
            joint_pos={
                "PlungerSlideJoint": 0.0,
            },
            joint_vel={},
        ),
        actuators={
            "PlungerSlideJoint": ImplicitActuatorCfg(
                joint_names_expr=["PlungerSlideJoint"],
                stiffness=1000.0,
                damping=80.0,
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
            ),
        },
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    # The slider joint limit is 0.0 (unpressed) to 0.01 (fully pressed).
    # Button is considered pressed when its plunger joint position exceeds this threshold.
    button_pressed_threshold = 0.004

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["button_pressed"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space.spaces["goal_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class PressButtonEnvEEAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class PressButtonEnvEEAbsL1Cfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class PressButtonEnvOmniHexaAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class PressButtonEnvFAHexaAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class PressButtonEnvFAHexaAbsL1Cfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class PressButtonEnvFAHexaBaseJointAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button FA-Hexa no-IK configuration with Base+joints absolute PID targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_PID


@configclass
class PressButtonEnvFAHexaBaseJointAbsL1Cfg(PressButtonEnvDefaultCfg):
    """Press button FA-Hexa no-IK configuration with Base+joints absolute L1 targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_L1


@configclass
class PressButtonEnvUAHexaAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class PressButtonEnvFAHexaAbsMPCCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class PressButtonEnvUAQuadAbsPIDCfg(PressButtonEnvDefaultCfg):
    """Press button environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
