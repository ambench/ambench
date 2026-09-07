# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

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
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class EventCfg:
    """
    Configuration for domain randomization.
    """

    # Randomize door position and orientation
    randomize_door_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-np.pi / 12.0, np.pi / 12.0),  # +/- 15 degrees
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("door"),
        },
    )

    # Randomize hinge joint stiffness/damping
    randomize_door_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["Hinge"]),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # # Randomize hinge joint friction
    # randomize_door_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("door", joint_names=["Hinge"]),
    #         "friction_distribution_params": (0.5, 2.0),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # # Randomize door mass
    # randomize_door_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("door"),
    #         "mass_distribution_params": (0.5, 1.5),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )


@configclass
class OpenDoorEnvDefaultCfg(BaseEnvCfg):
    # env
    episode_length_s = 15

    # Environment-specific parameters

    # Door asset
    door_position = (1.0, 0.0, 0.0)  # Default door position relative to env origin
    door_rotation = (1, 0, 0, 0)
    door_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/door.usd",
            scale=(1.0, 1.0, 1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 1.0kg
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=door_position,
            rot=door_rotation,
            joint_pos={
                "Hinge": 0.0,  # 0.0 is closed
            },
            joint_vel={},
        ),
        actuators={
            "DoorJoint": ImplicitActuatorCfg(
                joint_names_expr=["Hinge"],
                stiffness=0.0,
                damping=5.0,
                friction=0.2,
                dynamic_friction=0.2,
                effort_limit_sim=20.0,
            ),
        },
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    door_opened_threshold_deg = 20
    door_pass_dist = 0.3

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["door_joint_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.observation_space.spaces["door_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class OpenDoorEnvEEAbsPIDCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class OpenDoorEnvEEAbsL1Cfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class OpenDoorEnvOmniHexaAbsPIDCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class OpenDoorEnvFAHexaAbsPIDCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class OpenDoorEnvFAHexaAbsL1Cfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class OpenDoorEnvUAHexaAbsPIDCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class OpenDoorEnvUAQuadAbsPIDCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID


@configclass
class OpenDoorEnvFAHexaAbsMPCCfg(OpenDoorEnvDefaultCfg):
    """Open door environment configuration for FA-Hexa robot, absolute action with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC
