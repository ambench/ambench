# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Action-to-command pipelines used by the shared direct environment."""

from __future__ import annotations

from dataclasses import MISSING
from enum import Enum
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.utils import configclass

from ambench.controllers.controller_cfg import (
    BaseController,
    ControllerCfg,
    ControllerOutput,
)
from ambench.controllers.pyroki_ik_ctrl import (
    PyrokiIKController,
    PyrokiIKControllerConfig,
    ik_compute,
)
from ambench.robots.robot_io import RobotCommand

if TYPE_CHECKING:
    from ambench.tasks.base.base_env import BaseEnv


class ActionMode(Enum):
    """Supported public action representations."""

    ABSOLUTE_EE_POSE = "absolute_ee_pose"
    ABSOLUTE_BASE_JOINTS = "absolute_base_joints"


@configclass
class ControlPipelineCfg:
    """Controller algorithm, action representation, and optional IK settings."""

    class_type: type = MISSING
    action_mode: ActionMode = MISSING
    controller: ControllerCfg = MISSING
    ik: PyrokiIKControllerConfig | None = None

    def action_dim(self, arm_joint_count: int) -> int:
        if self.action_mode == ActionMode.ABSOLUTE_EE_POSE:
            return 8
        if self.action_mode == ActionMode.ABSOLUTE_BASE_JOINTS:
            return 8 + arm_joint_count
        raise ValueError(f"Unsupported action mode: {self.action_mode}.")


class ControlPipeline:
    """Common controller construction, targets, telemetry, and reset behavior."""

    def __init__(self, env: BaseEnv, cfg: ControlPipelineCfg) -> None:
        self.env = env
        self.cfg = cfg
        controller_kwargs: dict[str, Any] = dict(cfg.controller.params)
        controller_kwargs.update(
            robot_spec=env.cfg.robot_profile.robot,
            rotor_layout=env.robot_io.rotor_layout,
            rotor_actuator=env.robot_io.rotor_actuator,
            robot=env.robot,
            scene=env.scene,
            enable_saturation=env.cfg.enable_saturation,
            enable_aerodynamic_effects=env.cfg.enable_aerodynamic_effects,
            enable_wind_effect=env.cfg.enable_wind_effect,
        )
        self.controller: BaseController = cfg.controller.class_type(
            num_envs=env.num_envs,
            device=env.device,
            dt=env.dt,
            **controller_kwargs,
        )
        if env.robot_io.motor_arm_joint_ids:
            self.controller.set_motor_arm_joint_ids(env.robot_io.motor_arm_joint_ids)

        self.ik_controller = (
            PyrokiIKController(num_envs=env.num_envs, device=env.device, config=cfg.ik) if cfg.ik is not None else None
        )
        self.is_first_step = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.ee_cmd_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
        self.ee_cmd_quat_w = torch.zeros((env.num_envs, 4), device=env.device)
        self.ee_cmd_quat_w[:, 0] = 1.0
        self.last_output: ControllerOutput | None = None
        self.last_command: RobotCommand | None = None

    def _update_ee_target(self, actions: torch.Tensor) -> None:
        self.ee_cmd_pos_w = actions[:, :3] + self.env.scene.env_origins
        self.ee_cmd_quat_w = actions[:, 3:7]

    def _body_state_observation(self, body_idx: int) -> torch.Tensor:
        state_w = self.env.robot.data.body_link_state_w[:, body_idx, :]
        ang_vel_b = math_utils.quat_apply_inverse(state_w[:, 3:7], state_w[:, 10:13])
        return torch.cat([state_w[:, :10], ang_vel_b], dim=-1)

    def _command(
        self,
        output: ControllerOutput,
        gripper_targets: torch.Tensor,
        arm_targets: torch.Tensor | None = None,
    ) -> RobotCommand:
        motor_arm_targets = self.env.robot_io.motor_arm_targets
        if output.motor_arm_angles is not None:
            limits = self.env.robot.data.soft_joint_pos_limits[0, self.env.robot_io.motor_arm_joint_ids, :]
            motor_arm_targets = torch.clamp(output.motor_arm_angles, limits[:, 0], limits[:, 1])
            self.env.robot_io.motor_arm_targets = motor_arm_targets

        if output.arm_targets is not None:
            arm_targets = output.arm_targets
        if arm_targets is not None:
            self.env.robot_io.arm_targets = arm_targets
        self.env.robot_io.gripper_targets = gripper_targets
        command = RobotCommand(
            force_b=output.force_b,
            torque_b=output.torque_b,
            arm_position_targets=arm_targets,
            gripper_position_targets=gripper_targets,
            motor_arm_position_targets=motor_arm_targets if self.env.robot_io.motor_arm_joint_ids else None,
            motor_thrusts=output.motor_thrusts,
        )
        self.last_output = output
        self.last_command = command
        self.is_first_step.fill_(False)
        return command

    def process(self, actions: torch.Tensor) -> RobotCommand:
        raise NotImplementedError

    def reset(self, default_root_state: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.controller.reset(default_root_state, env_ids)
        self.is_first_step[env_ids] = True
        ee_link_idx = self.env.robot_io.ee_link_idx
        if ee_link_idx is not None:
            ee_state_w = self.env.robot.data.body_link_state_w[env_ids, ee_link_idx, :]
            self.ee_cmd_pos_w[env_ids] = ee_state_w[:, :3]
            self.ee_cmd_quat_w[env_ids] = self.env.robot_io.link_to_command_quat(ee_state_w[:, 3:7])


class EndEffectorPosePipeline(ControlPipeline):
    """Control a free end-effector articulation from EE pose actions."""

    def process(self, actions: torch.Tensor) -> RobotCommand:
        self._update_ee_target(actions)
        obs = self._body_state_observation(self.env.robot_io.control_body_idx)
        self.controller.compute_desired_states(self.ee_cmd_pos_w, self.ee_cmd_quat_w, self.is_first_step)
        output = self.controller.compute(obs)
        return self._command(output, self.env.robot_io.compute_gripper_targets(actions[:, -1]))


class ManipulatorIKPipeline(ControlPipeline):
    """Resolve EE actions through floating-base IK before controlling the base."""

    def process(self, actions: torch.Tensor) -> RobotCommand:
        if self.ik_controller is None:
            raise RuntimeError("ManipulatorIKPipeline requires an IK configuration.")
        self._update_ee_target(actions)
        ik_output = ik_compute(
            ik_controller=self.ik_controller,
            env=self.env,
            target_pos=self.ee_cmd_pos_w,
            target_quat=self.env.robot_io.command_to_link_quat(self.ee_cmd_quat_w),
        )
        arm_count = len(self.env.robot_io.arm_joint_ids)
        expected_dim = 8 + arm_count
        if ik_output.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} IK output values, got {ik_output.shape[1]}.")
        arm_targets = ik_output[:, 7 : 7 + arm_count]
        obs = self._body_state_observation(self.env.robot_io.base_link_idx)
        self.controller.compute_desired_states(ik_output[:, :3], ik_output[:, 3:7], self.is_first_step)
        output = self.controller.compute(obs)
        return self._command(output, self.env.robot_io.compute_gripper_targets(actions[:, -1]), arm_targets)


class ManipulatorDirectPipeline(ControlPipeline):
    """Control base pose and arm joints directly from one absolute action."""

    def process(self, actions: torch.Tensor) -> RobotCommand:
        arm_count = len(self.env.robot_io.arm_joint_ids)
        expected_dim = 8 + arm_count
        if actions.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} base/joint action values, got {actions.shape[1]}.")
        target_pos = actions[:, :3] + self.env.scene.env_origins
        target_quat = actions[:, 3:7]
        self.ee_cmd_pos_w = target_pos
        self.ee_cmd_quat_w = target_quat
        arm_targets = actions[:, 7 : 7 + arm_count]
        obs = self._body_state_observation(self.env.robot_io.base_link_idx)
        self.controller.compute_desired_states(target_pos, target_quat, self.is_first_step)
        output = self.controller.compute(obs)
        return self._command(output, self.env.robot_io.compute_gripper_targets(actions[:, -1]), arm_targets)


class ManipulatorMPCPipeline(ControlPipeline):
    """Track absolute EE targets with the existing whole-body MPC."""

    def process(self, actions: torch.Tensor) -> RobotCommand:
        self._update_ee_target(actions)
        base_obs = self._body_state_observation(self.env.robot_io.base_link_idx)
        arm_state = self.env.robot.data.joint_pos[:, self.env.robot_io.arm_joint_ids]
        obs = torch.cat([base_obs, arm_state], dim=-1)
        self.controller.compute_desired_states(self.ee_cmd_pos_w, self.ee_cmd_quat_w, self.is_first_step)
        self.controller.update_pos_max_from_wall(env=self.env)
        output = self.controller.compute(obs)
        return self._command(output, self.env.robot_io.compute_gripper_targets(actions[:, -1]))
