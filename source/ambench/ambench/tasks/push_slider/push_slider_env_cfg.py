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
from ambench.tasks.push_slider import push_slider_events
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
                "pitch": (-0, 0),  # Wall can rotate ±0.2 rad in pitch
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wall"),
        },
    )

    # Randomize slider position on wall surface (YZ plane relative to wall)
    randomize_slider_position = EventTerm(
        func=push_slider_events.randomize_slider_position_on_wall,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "slider_cfg": SceneEntityCfg("slider"),
            "y_range": (-0.3, 0.3),  # Slider can move ±0.3m in Y on wall
            "z_range": (-0.3, 0.3),  # Slider can move ±0.3m in Z on wall
        },
    )

    # Randomize slider joint stiffness/damping
    # randomize_slider_friction = EventTerm(
    #     func=push_slider_events.randomize_slider_joint_friction,
    #     mode="reset",
    #     params={
    #         "slider_cfg": SceneEntityCfg("slider", joint_names=["SliderJoint"]),
    #         "friction_distribution_params": (4.0, 20.0),
    #         "operation": "set",
    #     },
    # )

    # Randomize slider joint friction
    # randomize_slider_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("slider", joint_names=["SliderJoint"]),
    #         "friction_distribution_params": (0.75, 1.25), # Scale friction by 75% to 125%
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

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


@configclass
class PushSliderEnvDefaultCfg(BaseEnvCfg):
    # env
    episode_length_s = 26

    # Environment-specific parameters

    # environment assets
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

    # Slider positioned at same location as wall, facing same direction
    # Offset by half thickness so slider is on the wall surface
    slider_offset_x = -0.5 * wall_thickness  # Position slider on wall surface
    slider_position = (
        wall_position[0] + slider_offset_x,
        wall_position[1],
        wall_position[2],
    )
    slider_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SliderWithRail",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/slider_with_rail.usd",
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=slider_position,
            rot=wall_rotation,
            joint_pos={
                "SliderJoint": -0.5,
            },
            joint_vel={},
        ),
        actuators={
            "SliderJoint": ImplicitActuatorCfg(
                joint_names_expr=["SliderJoint"],
                stiffness=0.0,
                damping=5.0,
                friction=4.0,
                dynamic_friction=4.0,
                effort_limit_sim=1000.0,
                velocity_limit_sim=2.0,
            ),
        },
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    slider_pushed_threshold = 0.45

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["slider_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space.spaces["slider_pushed"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


@configclass
class PushSliderEnvEEAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class PushSliderEnvEEAbsL1Cfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class PushSliderEnvOmniHexaAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class PushSliderEnvFAHexaAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class PushSliderEnvFAHexaAbsL1Cfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class PushSliderEnvFAHexaBaseJointAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider FA-Hexa no-IK configuration with Base+joints absolute PID targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_PID


@configclass
class PushSliderEnvFAHexaBaseJointAbsL1Cfg(PushSliderEnvDefaultCfg):
    """Push slider FA-Hexa no-IK configuration with Base+joints absolute L1 targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_L1


@configclass
class PushSliderEnvUAHexaAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class PushSliderEnvUAQuadAbsPIDCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID


@configclass
class PushSliderEnvFAHexaAbsMPCCfg(PushSliderEnvDefaultCfg):
    """Push slider environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC
