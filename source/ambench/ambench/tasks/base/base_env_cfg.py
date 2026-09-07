# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared direct-environment configuration."""

from __future__ import annotations

from dataclasses import MISSING

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    NoiseModelWithAdditiveBiasCfg,
)

from ambench.tasks.base.robot_profiles import RobotProfileCfg


def _base_observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "ee_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "ee_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "ee_lin_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "ee_ang_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "base_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "base_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        "base_lin_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "base_ang_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        "gripper_width": gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
    })


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    """Task-independent simulation settings plus one composed robot profile."""

    decimation: int = 1
    action_space: int = MISSING
    observation_space: gym.spaces.Dict = _base_observation_space()

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False)
    spawn_ground_plane: bool = True

    robot_profile: RobotProfileCfg = MISSING
    scene_camera_cfg: CameraCfg | None = None

    enable_action_noise: bool = False
    enable_observation_noise: bool = False
    enable_saturation: bool = False
    enable_aerodynamic_effects: bool = False
    enable_wind_effect: bool = False

    action_noise_model: NoiseModelCfg | NoiseModelWithAdditiveBiasCfg | None = None
    observation_noise_model: NoiseModelCfg | NoiseModelWithAdditiveBiasCfg | None = None

    def __post_init__(self) -> None:
        self.action_space = self.robot_profile.control.action_dim(len(self.robot_profile.robot.arm_joint_names))
        self.observation_space = _base_observation_space()
        arm_joint_count = len(self.robot_profile.robot.arm_joint_names)
        if arm_joint_count:
            self.observation_space.spaces["arm_joint_pos"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(arm_joint_count,),
                dtype=np.float32,
            )
            self.observation_space.spaces["arm_joint_vel"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(arm_joint_count,),
                dtype=np.float32,
            )

        if self.enable_action_noise and self.action_noise_model is None:
            self.action_noise_model = NoiseModelCfg(noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="add"))
        if self.enable_observation_noise and self.observation_noise_model is None:
            self.observation_noise_model = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
                bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
            )
