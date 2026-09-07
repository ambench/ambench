# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    OMNI_HEXA_ABS_PID,
    UA_HEXA_ABS_PID,
    UA_QUAD_ABS_PID,
)
from ambench.tasks.pull_lever import pull_lever_events
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class EventCfg:
    """
    Configuration for domain randomization.
    """

    # Randomize wall position and orientation
    # Wall and lever rotate together to simulate a tilted wall
    randomize_wall_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),  # Wall can move ±0.2m in X
                "y": (-0.2, 0.2),  # Wall can move ±0.2m in Y
                "pitch": (-np.pi / 36.0, np.pi / 36.0),  # ±5 deg pitch
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("wall"),
        },
    )

    # Randomize lever position on wall surface
    randomize_lever_position = EventTerm(
        func=pull_lever_events.randomize_lever_position_on_wall,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "lever_cfg": SceneEntityCfg("lever_object"),
            "y_range": (-0.3, 0.3),  # Lever can move ±0.3m in Y on wall
            "z_range": (-0.3, 0.3),  # Lever can move ±0.3m in Z on wall
        },
    )

    # Randomize lever handle visual color without changing the base.
    randomize_lever_collision_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("lever_object"),
            "mesh_name": "lever/lever/lever_collision",
            "colors": {
                "r": (0.05, 0.95),
                "g": (0.05, 0.95),
                "b": (0.05, 0.95),
            },
            "event_name": "lever_collision_color_randomizer",
        },
    )

    # Randomize lever tip visual color separately from the handle and base.
    randomize_tip_collision_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("lever_object"),
            "mesh_name": "lever/lever/tip_collision",
            "colors": {
                "r": (0.05, 0.95),
                "g": (0.05, 0.95),
                "b": (0.05, 0.95),
            },
            "event_name": "tip_collision_color_randomizer",
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


@configclass
class PullLeverEnvDefaultCfg(BaseEnvCfg):
    # env
    episode_length_s = 20

    # Environment-specific parameters

    # environment assets - wall and lever are independent assets at the same level
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

    # Position lever on the wall surface
    lever_thickness = 0.06
    lever_offset_x = -0.5 * (wall_thickness + lever_thickness)
    lever_position = (
        wall_position[0] + lever_offset_x,
        wall_position[1],
        wall_position[2],
    )

    # Use pre-generated lever.usd file
    lever_usd_path = f"{LOCAL_ASSET_DIR}/objects/lever.usd"
    lever_object_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Lever",
        spawn=sim_utils.UsdFileCfg(
            usd_path=lever_usd_path,
            scale=(2.0, 2.0, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # Disable gravity for root body to prevent lever from falling
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=lever_position,
            rot=(0.7071, 0.0, -0.7071, 0.0),  # Rotate lever to be flush with wall (90° around Y)
            joint_pos={"lever_joint": 0.0},  # Initial joint position (lever at center, 0 = middle of range)
            joint_vel={},
        ),
        actuators={
            "lever_joint": ImplicitActuatorCfg(
                joint_names_expr=["lever_joint"],
                stiffness=20.0,  # Low stiffness to allow easy movement
                damping=100.0,  # Moderate damping to eliminate inertia - lever stops when no force is applied
                effort_limit_sim=50.0,  # Maximum torque - gripper can overcome this to pull lever
                velocity_limit_sim=5.0,  # Maximum joint velocity
            ),
        },  # Moderate damping ensures lever requires continuous force to move and has minimal inertia
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    lever_pull_angle_min = np.deg2rad(40)


@configclass
class PullLeverEnvEECfg(PullLeverEnvDefaultCfg):
    """Base for pull lever EE configs. Overrides EE init rotation for lever grasping."""

    def __post_init__(self):
        super().__post_init__()
        asset = self.robot_profile.robot.asset
        init_state = asset.init_state
        asset = asset.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_state.pos,
                rot=(0.7071, 0.7071, 0.0, 0.0),
                joint_pos=init_state.joint_pos,
                joint_vel=init_state.joint_vel,
            )
        )
        self.robot_profile = self.robot_profile.replace(robot=self.robot_profile.robot.replace(asset=asset))


@configclass
class PullLeverEnvFAHexaCfg(PullLeverEnvDefaultCfg):
    """Base for pull lever FA-Hexa configs. Overrides EE init roll to 90° for lever grasping.

    FA-Hexa initial state is defined by full DOF (base pose + joint positions).
    EE roll is achieved by setting arm_link4_roll_joint (wrist roll joint).
    """

    def __post_init__(self):
        super().__post_init__()
        asset = self.robot_profile.robot.asset
        init_state = asset.init_state
        joint_pos = dict(init_state.joint_pos)
        joint_pos["arm_link4_roll_joint"] = float(np.deg2rad(90))
        asset = asset.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_state.pos,
                rot=init_state.rot,
                joint_pos=joint_pos,
                joint_vel=init_state.joint_vel,
            )
        )
        self.robot_profile = self.robot_profile.replace(robot=self.robot_profile.robot.replace(asset=asset))


@configclass
class PullLeverEnvUAHexaCfg(PullLeverEnvDefaultCfg):
    """Base for pull lever UA-Hexa configs. Overrides EE init roll to 90° for lever grasping."""

    def __post_init__(self):
        super().__post_init__()
        asset = self.robot_profile.robot.asset
        init_state = asset.init_state
        joint_pos = dict(init_state.joint_pos)
        joint_pos["arm_link4_roll_joint"] = float(np.deg2rad(90))
        asset = asset.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_state.pos,
                rot=init_state.rot,
                joint_pos=joint_pos,
                joint_vel=init_state.joint_vel,
            )
        )
        self.robot_profile = self.robot_profile.replace(robot=self.robot_profile.robot.replace(asset=asset))


@configclass
class PullLeverEnvFAHexaMPCBaseCfg(PullLeverEnvDefaultCfg):
    """Base for pull lever FA-Hexa MPC configs. Overrides EE init roll to 90° for lever grasping."""

    def __post_init__(self):
        super().__post_init__()
        asset = self.robot_profile.robot.asset
        init_state = asset.init_state
        joint_pos = dict(init_state.joint_pos)
        joint_pos["arm_link4_roll_joint"] = float(np.deg2rad(90))
        asset = asset.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_state.pos,
                rot=init_state.rot,
                joint_pos=joint_pos,
                joint_vel=init_state.joint_vel,
            )
        )
        self.robot_profile = self.robot_profile.replace(robot=self.robot_profile.robot.replace(asset=asset))


@configclass
class PullLeverEnvEEAbsPIDCfg(PullLeverEnvEECfg):
    """Pull lever environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class PullLeverEnvEEAbsL1Cfg(PullLeverEnvEECfg):
    """Pull lever environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class PullLeverEnvOmniHexaAbsPIDCfg(PullLeverEnvDefaultCfg):
    """Pull lever environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class PullLeverEnvFAHexaAbsPIDCfg(PullLeverEnvFAHexaCfg):
    """Pull lever environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class PullLeverEnvFAHexaAbsL1Cfg(PullLeverEnvFAHexaCfg):
    """Pull lever environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class PullLeverEnvUAHexaAbsPIDCfg(PullLeverEnvUAHexaCfg):
    """Pull lever environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class PullLeverEnvFAHexaAbsMPCCfg(PullLeverEnvFAHexaMPCBaseCfg):
    """Pull lever environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class PullLeverEnvUAQuadAbsPIDCfg(PullLeverEnvDefaultCfg):
    """Pull lever environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
