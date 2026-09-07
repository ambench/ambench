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

from ambench.assets.object_specs import CONTAINER_DEFAULT_SIZE
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
    """Configuration for toss-ball domain randomization."""

    randomize_container_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("container"),
        },
    )


@configclass
class TossBallEnvDefaultCfg(BaseEnvCfg):
    """Default configuration for the toss-ball task."""

    episode_length_s = 10

    # Container
    container_size = (0.4, 0.3, 0.17)
    container_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Container",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/container.usd",
            scale=(
                container_size[0] / CONTAINER_DEFAULT_SIZE[0],
                container_size[1] / CONTAINER_DEFAULT_SIZE[1],
                container_size[2] / CONTAINER_DEFAULT_SIZE[2],
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.3, 0.0, 0.0),
        ),
    )

    # Ball setup. Ball pose is overridden on reset to place it at the gripper.
    ball_radius = 0.04
    ball_mass = 0.01
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.05, 0.05),
                roughness=0.4,
                metallic=0.0,
            ),
        ),
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    # Success is measured against a conservative inner volume, not the outer bin bounds.
    # This avoids counting balls that settle on the side wall or near the top rim.
    success_robot_behind_bin_threshold = 0.7
    release_distance_threshold = 0.12

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["ball_pos"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space.spaces["ball_lin_vel"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space.spaces["target_pos"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )


@configclass
class TossBallEnvEEAbsPIDCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class TossBallEnvEEAbsL1Cfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class TossBallEnvOmniHexaAbsPIDCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class TossBallEnvFAHexaAbsPIDCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class TossBallEnvFAHexaAbsL1Cfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class TossBallEnvUAHexaAbsPIDCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class TossBallEnvFAHexaAbsMPCCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class TossBallEnvUAQuadAbsPIDCfg(TossBallEnvDefaultCfg):
    """Toss ball environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
