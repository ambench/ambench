# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
from ambench.utils.assets import LOCAL_ASSET_DIR


@configclass
class NDTEnvDefaultCfg(BaseEnvCfg):
    """Default configuration for the NDT inspection task."""

    episode_length_s = 20

    # Task parameters
    inspection_height: float = 7.0

    # Inspection scene configuration
    scene_translation = (28.3, 0.0, 0.0)
    scene_usd_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Samples/Replicator/Benchmark/full_warehouse_worker_benchmark_sdg.usd",
        scale=(1.0, 1.0, 1.0),
    )
    marker_translation = (1.962, 0.0, 7.0)
    marker_orientation = (0, 0.7071, 0, 0.7071)  # Rotate 90 deg around Y-axis
    marker_usd_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{LOCAL_ASSET_DIR}/objects/marked_region.usda",
        scale=(1.0, 1.0, 1.0),
    )

    # Wall configuration
    wall_position = (2.0, -10.0, 1.0)
    wall_rotation = (1.0, 0.0, 0.0, 0.0)
    wall: WallSceneCfg = WallSceneCfg()
    wall_cfg: RigidObjectCfg = spawn_wall(
        prim_path="/World/envs/env_.*/Wall",
        cfg=wall,
        translation=wall_position,
        orientation=wall_rotation,
    )

    # Success criteria
    goal_position = (1.77, 0.0, 7.0)
    success_x_tolerance: float = 0.03
    success_yz_tolerance: float = 0.08
    success_hold_steps: int = 100

    def __post_init__(self):
        super().__post_init__()

        self.marker_translation = (
            self.marker_translation[0],
            self.marker_translation[1],
            self.inspection_height,
        )
        self.goal_position = (
            self.goal_position[0],
            self.goal_position[1],
            self.inspection_height,
        )

        self.observation_space.spaces["goal_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )


@configclass
class NDTEnvEEAbsPIDCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class NDTEnvEEAbsL1Cfg(NDTEnvDefaultCfg):
    """NDT environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class NDTEnvOmniHexaAbsPIDCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class NDTEnvFAHexaAbsPIDCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class NDTEnvFAHexaAbsL1Cfg(NDTEnvDefaultCfg):
    """NDT environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class NDTEnvUAHexaAbsPIDCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class NDTEnvFAHexaAbsMPCCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC


@configclass
class NDTEnvUAQuadAbsPIDCfg(NDTEnvDefaultCfg):
    """NDT environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID


# Backward-compatible config aliases for older task IDs and imports.
