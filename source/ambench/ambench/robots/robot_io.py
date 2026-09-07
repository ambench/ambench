# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator-facing robot handles, state access, and command application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation

from ambench.robots.robot_cfg import RobotSpecCfg, RotorLayout
from ambench.robots.rotor_actuator import RotorActuator
from ambench.robots.visuals.propeller_animation import PropAnimation


def _joint_ids_in_order(robot: Articulation, joint_names: Sequence[str]) -> list[int]:
    joint_ids: list[int] = []
    for name in joint_names:
        ids, _ = robot.find_joints(name)
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one joint named {name!r}, found {len(ids)}.")
        joint_ids.append(int(ids[0]))
    return joint_ids


def _body_id(robot: Articulation, body_name: str) -> int:
    ids, _ = robot.find_bodies(body_name)
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one body named {body_name!r}, found {len(ids)}.")
    return int(ids[0])


@dataclass
class RobotCommand:
    """Complete simulator command for one environment step."""

    force_b: torch.Tensor
    torque_b: torch.Tensor
    arm_position_targets: torch.Tensor | None = None
    arm_effort_targets: torch.Tensor | None = None
    gripper_position_targets: torch.Tensor | None = None
    motor_arm_position_targets: torch.Tensor | None = None
    motor_thrusts: torch.Tensor | None = None


class RobotIO:
    """Resolve one robot specification into simulator handles and command buffers."""

    def __init__(
        self,
        robot: Articulation,
        spec: RobotSpecCfg,
        num_envs: int,
        device: str | torch.device,
        dt: float,
    ) -> None:
        self.robot = robot
        self.spec = spec
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = dt

        self.control_body_idx = _body_id(robot, spec.control_body_name)
        self.base_link_idx = _body_id(robot, spec.base_body_name)
        self.ee_link_idx = _body_id(robot, spec.ee_body_name) if spec.ee_body_name is not None else None
        self.arm_joint_ids = _joint_ids_in_order(robot, spec.arm_joint_names)
        gripper_names = spec.gripper.joint_names if spec.gripper is not None else ()
        self.gripper_joint_ids = _joint_ids_in_order(robot, gripper_names)
        multirotor = spec.multirotor
        self.rotor_layout: RotorLayout | None = (
            None if multirotor is None else multirotor.layout_provider(device=self.device)
        )
        motor_arm_names = multirotor.motor_arm_joint_names if multirotor is not None else ()
        if self.rotor_layout is not None:
            if self.rotor_layout.is_variable_tilt and len(motor_arm_names) != self.rotor_layout.num_rotors:
                raise ValueError("Variable-tilt layouts require one motor-arm joint per rotor.")
            if not self.rotor_layout.is_variable_tilt and motor_arm_names:
                raise ValueError("Fixed rotor layouts must not configure motor-arm joints.")
        actuator_cfg = multirotor.actuator if multirotor is not None else None
        self.rotor_actuator: RotorActuator | None = None
        if actuator_cfg is not None and actuator_cfg.has_dynamics:
            if self.rotor_layout is None:
                raise RuntimeError("Rotor dynamics require a resolved rotor layout.")
            self.rotor_actuator = RotorActuator(
                actuator_cfg,
                num_envs=num_envs,
                num_rotors=self.rotor_layout.num_rotors,
                dt=dt,
                device=self.device,
            )
        self.motor_arm_joint_ids = _joint_ids_in_order(robot, motor_arm_names)

        self.arm_targets = torch.zeros((num_envs, len(self.arm_joint_ids)), device=self.device)
        self.gripper_targets = torch.zeros((num_envs, len(self.gripper_joint_ids)), device=self.device)
        self.motor_arm_targets = torch.zeros((num_envs, len(self.motor_arm_joint_ids)), device=self.device)

        frame = spec.end_effector
        command_to_link_quat = None if frame is None else frame.command_to_link_quat_wxyz
        self._command_to_link_quat_wxyz = (
            None
            if command_to_link_quat is None
            else torch.tensor(command_to_link_quat, device=self.device, dtype=torch.float32)
        )
        self._link_to_command_quat_wxyz = (
            None
            if self._command_to_link_quat_wxyz is None
            else math_utils.quat_conjugate(self._command_to_link_quat_wxyz)
        )

        propeller_cfg = multirotor.propeller_viz if multirotor is not None else None
        self._propeller_animation = PropAnimation(
            robot_prim_path=spec.asset.prim_path,
            propeller_viz_cfg=propeller_cfg,
            spin_directions=None if self.rotor_layout is None else self.rotor_layout.spin_directions,
            reference_thrust=None if actuator_cfg is None else actuator_cfg.thrust_limits[1],
            num_envs=num_envs,
            device=self.device,
        )

    @property
    def has_arm(self) -> bool:
        return bool(self.spec.arm_joint_names)

    @property
    def has_gripper(self) -> bool:
        return self.spec.gripper is not None and bool(self.spec.gripper.joint_names)

    def compute_gripper_targets(self, action: torch.Tensor) -> torch.Tensor:
        """Map ``-1=closed`` and ``+1=open`` onto ordered gripper joint targets."""
        action = action.clamp(-1.0, 1.0)
        if action.ndim == 1:
            action = action.unsqueeze(-1)

        gripper = self.spec.gripper
        if gripper is not None and gripper.open_joint_positions is not None:
            if gripper.closed_joint_positions is None:
                raise ValueError("Explicit open gripper positions require closed positions.")
            open_positions = torch.tensor(gripper.open_joint_positions, device=self.device, dtype=torch.float32)
            closed_positions = torch.tensor(gripper.closed_joint_positions, device=self.device, dtype=torch.float32)
        else:
            limits = self.robot.data.soft_joint_pos_limits[0, self.gripper_joint_ids, :]
            closed_positions = limits[:, 0]
            open_positions = limits[:, 1]

        open_fraction = 0.5 * (action + 1.0)
        return closed_positions + open_fraction * (open_positions - closed_positions)

    def compute_tool_tip_pos_w(self, ee_state_w: torch.Tensor) -> torch.Tensor:
        """Return the configured tool-tip point for batched EE link states."""
        frame = self.spec.end_effector
        offset = (0.0, 0.0, 0.0) if frame is None else frame.tool_tip_offset_local
        offset_local = torch.tensor(offset, device=ee_state_w.device, dtype=ee_state_w.dtype)
        offset_local = offset_local.unsqueeze(0).expand(ee_state_w.shape[0], -1)
        return ee_state_w[:, :3] + math_utils.quat_apply(ee_state_w[:, 3:7], offset_local)

    def command_to_link_quat(self, quat_cmd_w: torch.Tensor) -> torch.Tensor:
        """Map command-frame orientations into the authored end-effector link frame."""
        if self._command_to_link_quat_wxyz is None:
            return quat_cmd_w
        offset = self._command_to_link_quat_wxyz.to(dtype=quat_cmd_w.dtype)
        return math_utils.quat_mul(offset.unsqueeze(0).expand(quat_cmd_w.shape[0], -1), quat_cmd_w)

    def link_to_command_quat(self, quat_link_w: torch.Tensor) -> torch.Tensor:
        """Map authored end-effector link orientations into the command frame."""
        if self._link_to_command_quat_wxyz is None:
            return quat_link_w
        offset = self._link_to_command_quat_wxyz.to(dtype=quat_link_w.dtype)
        return math_utils.quat_mul(offset.unsqueeze(0).expand(quat_link_w.shape[0], -1), quat_link_w)

    def apply(self, command: RobotCommand) -> None:
        """Apply a complete command to the articulation."""
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=command.force_b.unsqueeze(1),
            torques=command.torque_b.unsqueeze(1),
            body_ids=[self.control_body_idx],
        )
        if command.arm_effort_targets is not None:
            self.robot.set_joint_effort_target(command.arm_effort_targets, joint_ids=self.arm_joint_ids)
        elif command.arm_position_targets is not None:
            self.robot.set_joint_position_target(command.arm_position_targets, joint_ids=self.arm_joint_ids)
        if command.gripper_position_targets is not None:
            self.robot.set_joint_position_target(command.gripper_position_targets, joint_ids=self.gripper_joint_ids)
        if command.motor_arm_position_targets is not None:
            self.robot.set_joint_position_target(
                command.motor_arm_position_targets,
                joint_ids=self.motor_arm_joint_ids,
            )
        self.robot.write_data_to_sim()
        if command.motor_thrusts is not None:
            self._propeller_animation.update(command.motor_thrusts, self.dt)

    def reset(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
        """Reset command buffers from the articulation's reset joint state."""
        if self.arm_joint_ids:
            self.arm_targets[env_ids] = joint_pos[:, self.arm_joint_ids]
        if self.gripper_joint_ids:
            self.gripper_targets[env_ids] = joint_pos[:, self.gripper_joint_ids]
        if self.motor_arm_joint_ids:
            self.motor_arm_targets[env_ids] = joint_pos[:, self.motor_arm_joint_ids]
        if self.rotor_actuator is not None:
            self.rotor_actuator.reset(env_ids)
        self._propeller_animation.reset(env_ids)
