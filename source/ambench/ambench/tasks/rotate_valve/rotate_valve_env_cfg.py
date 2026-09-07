# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as sim_schemas
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

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
from ambench.tasks.rotate_valve import rotate_valve_events
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class ValveCfg:
    """Configuration for the valve object."""

    mass: float = 0.05
    scale: tuple[float, float, float] = (1.8, 1.0, 1.8)


@configclass
class EventCfg:
    """Domain randomization for the wall and valve."""

    # The valve placement event depends on this wall pose.
    randomize_wall_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "pitch": (0, 0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wall"),
        },
    )

    randomize_valve_position = EventTerm(
        func=rotate_valve_events.randomize_valve_position_on_wall,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "valve_cfg": SceneEntityCfg("valve_object"),
            "y_range": (-0.2, 0.2),
            "z_range": (-0.2, 0.2),
        },
    )


@configclass
class RotateValveEnvDefaultCfg(BaseEnvCfg):
    """Default configuration for the rotate-valve task."""

    # Episode settings
    episode_length_s = 20

    # Wall configuration
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

    # Valve configuration
    valve: ValveCfg = ValveCfg()
    valve_offset_x = -0.06
    valve_position = (
        wall_position[0] + valve_offset_x,
        wall_position[1],
        wall_position[2],
    )

    valve_object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/valve.usd",
            mass_props=sim_utils.MassPropertiesCfg(mass=valve.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            scale=valve.scale,
            articulation_props=sim_schemas.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=valve_position,
            rot=(0.5, 0.5, 0.5, 0.5),  # 90 deg about Y, 90 deg about X
            joint_pos={
                "base_to_shaft": 0.0,
                "disk_to_handle": 0,
            },
            joint_vel={},
        ),
        actuators={
            "base_to_shaft": ImplicitActuatorCfg(
                joint_names_expr=["base_to_shaft"],
                stiffness=0.0,
                damping=15.0,
                friction=0.2,
                dynamic_friction=0.2,
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
            ),
            "disk_to_handle": ImplicitActuatorCfg(
                joint_names_expr=["disk_to_handle"],
                stiffness=0.0,
                damping=10.0,
                friction=0.1,
                dynamic_friction=0.1,
                effort_limit_sim=20.0,
                velocity_limit_sim=5.0,
            ),
        },
    )

    # Domain randomization
    events = EventCfg()

    # Success criteria
    valve_target_angle: float = 17 / 18 * torch.pi
    valve_engagement_ratio: float = 0.05

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["goal_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class RotateValveEnvEEAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class RotateValveEnvEEAbsL1Cfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class RotateValveEnvOmniHexaAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class RotateValveEnvFAHexaAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class RotateValveEnvFAHexaAbsL1Cfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class RotateValveEnvFAHexaBaseJointAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve FA-Hexa no-IK configuration with Base+joints absolute PID targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_PID


@configclass
class RotateValveEnvFAHexaBaseJointAbsL1Cfg(RotateValveEnvDefaultCfg):
    """Rotate valve FA-Hexa no-IK configuration with Base+joints absolute L1 targets."""

    robot_profile = FA_HEXA_BASE_JOINT_ABS_L1


@configclass
class RotateValveEnvUAHexaAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class RotateValveEnvFAHexaAbsMPCCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class RotateValveEnvUAQuadAbsPIDCfg(RotateValveEnvDefaultCfg):
    """Rotate valve environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
