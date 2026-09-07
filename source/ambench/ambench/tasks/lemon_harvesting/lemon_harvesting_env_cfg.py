# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import MISSING

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ambench.assets.object_specs import CONTAINER_DEFAULT_SIZE
from ambench.controllers.pyroki_ik_ctrl import PyrokiIKControllerConfig
from ambench.robots.fa_hexa import URDF_FILE_PATH as FA_HEXA_URDF_FILE_PATH
from ambench.robots.ua_quad import URDF_FILE_PATH as UA_QUAD_URDF_FILE_PATH
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
from ambench.tasks.lemon_harvesting import lemon_harvesting_events
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class EventCfg:
    """Domain randomization for wall-mounted fruit placement."""

    randomize_lemon_lime_on_wall = EventTerm(
        func=lemon_harvesting_events.randomize_wall_fruit_attachments,
        mode="reset",
        params={
            "pos_range": {
                "y": (-0.5, 0.5),
                "z": (1.1, 1.5),
            },
            "min_sep_yz": 0.15,
            "break_force": 10.0,
            "break_torque": 10.0,
        },
    )


@configclass
class LemonHarvestingEnvDefaultCfg(BaseEnvCfg):
    """Default configuration for the lemon harvesting environment."""

    # Episode settings
    episode_length_s = 30

    # Environment-specific parameters
    max_limes_in_scene: int = 6
    wall_attach_x_offset: float = 0.08

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

    # Lemon configuration
    lemon_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/lemon",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/lemon.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.05,
            ),
        ),
    )

    # Lime configuration
    lime_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/lime",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/lime.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.02,
            ),
        ),
    )
    lime_object_cfgs: list[RigidObjectCfg] = MISSING

    # Table configuration
    table_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cline",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/cline.usd",
            scale=(0.4, 0.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.72, 0.5, 0.0),
        ),
    )

    # Container configuration
    container_size: tuple[float, float, float] = (0.4, 0.3, 0.17)
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
            pos=(1.72, 0.5, 0.65),
        ),
    )

    # Domain randomization
    events = EventCfg()

    # Success criteria
    carry_distance_threshold: float = 0.02
    grasp_distance_threshold: float = 0.12
    grasp_direction_cosine_threshold: float = 0.75
    gripper_closed_threshold: float = 0.05
    gripper_open_threshold: float = 0.0898

    def __post_init__(self):
        super().__post_init__()

        num_limes = int(self.max_limes_in_scene)
        self.lime_object_cfgs = []
        for lime_index in range(1, num_limes + 1):
            lime_cfg = copy.deepcopy(self.lime_object_cfg)
            lime_cfg.prim_path = f"/World/envs/env_.*/lime{lime_index:02d}"
            self.lime_object_cfgs.append(lime_cfg)

        self.observation_space.spaces["lemon_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.observation_space.spaces["container_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class LemonHarvestingEnvOmniHexaPIDBaseCfg(LemonHarvestingEnvDefaultCfg):
    """Shared Omni-Hexa lemon configuration with task-specific IK tuning."""

    def __post_init__(self):
        super().__post_init__()
        robot = self.robot_profile.robot
        ik = self.robot_profile.control.ik
        if ik is None:
            raise ValueError("The OmniHexa lemon profile requires IK.")
        self.robot_profile.control.ik = PyrokiIKControllerConfig(
            urdf_path=ik.urdf_path,
            target_link_name=robot.ee_body_name,
            fix_base_position=(False, False, False),
            fix_base_orientation=(False, False, False),
            initial_base_position=(0.0, 0.0, 1.0),
            initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
            enable_collision=False,
            cost_weights={
                "smoothness_base_pos": 1.0,
                "smoothness_base_ori": 1.0,
                "rest_arm": 0.1,
            },
            base_pos_limits={
                "min_z": 0.0,
            },
            safety_margin=0.5,
            arm_joint_names=list(robot.arm_joint_names),
            initial_joint_positions=dict(robot.asset.init_state.joint_pos),
        )


@configclass
class LemonHarvestingEnvFAHexaPIDBaseCfg(LemonHarvestingEnvDefaultCfg):
    """Shared FA-Hexa lemon configuration with task-specific IK tuning."""

    def __post_init__(self):
        super().__post_init__()
        self.robot_profile.control.ik = PyrokiIKControllerConfig(
            urdf_path=FA_HEXA_URDF_FILE_PATH,
            target_link_name="ee_link",
            fix_base_position=(False, False, False),
            fix_base_orientation=(True, True, True),
            initial_base_position=(0.0, 0.0, 1.0),
            initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
            enable_collision=False,
            cost_weights={
                "smoothness_base_pos": 6.0,
                "smoothness_base_ori": 0.1,
                "rest_arm": 0.1,
            },
            base_pos_limits={
                "max_x": 1.0,
                "min_z": 0.0,
            },
            safety_margin=1.0,
        )


@configclass
class LemonHarvestingEnvUAQuadPIDBaseCfg(LemonHarvestingEnvDefaultCfg):
    """Shared UA-Quad lemon IK tuning for reaching wall-mounted fruit."""

    def __post_init__(self):
        super().__post_init__()
        self.robot_profile.control.ik = PyrokiIKControllerConfig(
            urdf_path=UA_QUAD_URDF_FILE_PATH,
            target_link_name="ee_link",
            fix_base_position=(False, False, False),
            fix_base_orientation=(True, True, True),
            initial_base_position=(0.0, 0.0, 1.0),
            initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
            enable_collision=False,
            cost_weights={
                "pose_pos": 15.0,
                "pose_ori": 4.0,
                "smoothness_base_pos": 6.0,
                "smoothness_base_ori": 1.0,
                "rest_arm": 0.1,
            },
            base_pos_limits={
                "max_x": 2.0,
                "min_z": 0.0,
            },
            safety_margin=0.0,
        )


@configclass
class LemonHarvestingEnvEEAbsPIDCfg(LemonHarvestingEnvDefaultCfg):
    """Lemon harvesting environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class LemonHarvestingEnvEEAbsL1Cfg(LemonHarvestingEnvDefaultCfg):
    """Lemon harvesting environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class LemonHarvestingEnvOmniHexaAbsPIDCfg(LemonHarvestingEnvOmniHexaPIDBaseCfg):
    """Lemon harvesting for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class LemonHarvestingEnvFAHexaAbsPIDCfg(LemonHarvestingEnvFAHexaPIDBaseCfg):
    """Lemon harvesting environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class LemonHarvestingEnvFAHexaAbsL1Cfg(LemonHarvestingEnvDefaultCfg):
    """Lemon harvesting environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class LemonHarvestingEnvUAHexaAbsPIDCfg(LemonHarvestingEnvDefaultCfg):
    """Lemon harvesting environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class LemonHarvestingEnvFAHexaAbsMPCCfg(LemonHarvestingEnvDefaultCfg):
    """Lemon harvesting environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class LemonHarvestingEnvUAQuadAbsPIDCfg(LemonHarvestingEnvUAQuadPIDBaseCfg):
    """Lemon harvesting environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID
