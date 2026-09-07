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
    """Configuration for domain randomization."""

    # Randomize can position on the cabinet shelf.
    randomize_can_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.05, 0.05),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("can"),
        },
    )


@configclass
class CabinetPickPlaceEnvDefaultCfg(BaseEnvCfg):
    episode_length_s = 60

    # Cabinet configuration
    pulling_door_position = (1.5, 0.0, 0.0)
    pulling_door_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Cabinet",
        articulation_root_prim_path="/pulling_cabinet/pulling_cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/cabinet.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pulling_door_position,
            joint_pos={},
            joint_vel={},
        ),
        actuators={
            "DoorJoint": ImplicitActuatorCfg(
                joint_names_expr=["pull_door_slide"],
                stiffness=0.0,
                damping=10.0,
                friction=2.0,
                dynamic_friction=2.0,
                effort_limit_sim=50.0,
            ),
        },
    )

    # Can configuration
    can_position = (
        pulling_door_position[0] + 0.76,
        pulling_door_position[1] + 0.3,
        pulling_door_position[2] + 1.17,
    )  # Hand-tuned shelf pose from the previous stable reset configuration.

    can_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Can",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/cabinet_can/can.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=can_position,
        ),
    )

    # Domain randomization
    events = EventCfg()

    # Success criteria
    drawer_top_z = 1.67  # Authored placement surface height in the env-local frame.
    can_place_z_min = 0.00
    can_place_z_max = 0.05
    can_place_velocity_threshold = 0.1

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["pulling_door_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space.spaces["can_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class CabinetPickPlaceEnvEEAbsPIDCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class CabinetPickPlaceEnvEEAbsL1Cfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class CabinetPickPlaceEnvOmniHexaAbsPIDCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class CabinetPickPlaceEnvFAHexaAbsPIDCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class CabinetPickPlaceEnvFAHexaAbsL1Cfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class CabinetPickPlaceEnvUAHexaAbsPIDCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class CabinetPickPlaceEnvFAHexaAbsMPCCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class CabinetPickPlaceEnvUAQuadAbsPIDCfg(CabinetPickPlaceEnvDefaultCfg):
    """Cabinet pick place environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
