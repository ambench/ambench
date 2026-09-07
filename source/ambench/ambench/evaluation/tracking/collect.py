# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime tracking data collection from AM Isaac environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import isaaclab.utils.math as math_utils
import numpy as np
import torch


def _vector(value: Any, *, dtype=np.float64) -> np.ndarray | None:
    if value is None:
        return None
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return np.asarray(array, dtype=dtype).reshape(-1)


def _float_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(val) for val in np.asarray(value).reshape(-1).tolist()]


def _bool_list(value: Any) -> list[bool] | None:
    if value is None:
        return None
    return [bool(val) for val in _vector(value, dtype=np.bool_).tolist()]


def _quat_tensor(quat_wxyz: Any) -> torch.Tensor | None:
    quat = _vector(quat_wxyz, dtype=np.float64)
    if quat is None:
        return None
    quat_norm = np.linalg.norm(quat)
    if quat_norm <= 0.0:
        return None
    quat_tensor = torch.as_tensor(quat.reshape(1, 4), dtype=torch.float64)
    return math_utils.normalize(quat_tensor)


def _quat_error_rad(quat_a_wxyz: Any, quat_b_wxyz: Any) -> float:
    quat_a = _quat_tensor(quat_a_wxyz)
    quat_b = _quat_tensor(quat_b_wxyz)
    if quat_a is None or quat_b is None:
        return float("nan")
    return float(math_utils.quat_error_magnitude(quat_a, quat_b).detach().cpu().item())


def _tilt_from_quat_wxyz(quat_wxyz: Any) -> float:
    quat = _quat_tensor(quat_wxyz)
    if quat is None:
        return float("nan")
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
    return float(torch.linalg.vector_norm(torch.stack((roll, pitch))).detach().cpu().item())


@dataclass
class TrackingCollector:
    """Collect one canonical tracking timestep for an env instance."""

    env: Any
    env_idx: int = 0

    def __post_init__(self) -> None:
        self.env_unwrapped = self.env.unwrapped
        self.prev_desired_base_pos = None
        self.prev_desired_base_quat = None
        self.prev_desired_joint_pos = None
        self.arm_reach = self._arm_reach()
        self.thrust_limits = self._thrust_limits()

    def reset(self) -> None:
        self.prev_desired_base_pos = None
        self.prev_desired_base_quat = None
        self.prev_desired_joint_pos = None

    def _env_origin(self) -> np.ndarray:
        origins = getattr(getattr(self.env_unwrapped, "scene", None), "env_origins", None)
        if origins is None:
            return np.zeros(3, dtype=np.float64)
        return _vector(origins[self.env_idx])

    def _arm_reach(self) -> float:
        robot_spec = self.env_unwrapped.cfg.robot_profile.robot
        arm_reach = robot_spec.max_arm_reach
        if arm_reach is not None and arm_reach > 0.0:
            return float(arm_reach)
        try:
            ee_pos = self.env_unwrapped.robot.data.body_link_state_w[
                self.env_idx,
                self.env_unwrapped.ee_link_idx,
                0:3,
            ]
            base_pos = self.env_unwrapped.robot.data.body_link_state_w[
                self.env_idx,
                self.env_unwrapped.base_link_idx,
                0:3,
            ]
            reach = torch.linalg.norm(ee_pos - base_pos).detach().cpu().item()
            return float(reach) if reach > 0.0 else float("nan")
        except (AttributeError, IndexError, RuntimeError):
            return float("nan")

    def _thrust_limits(self) -> tuple[float, float] | None:
        multirotor = self.env_unwrapped.cfg.robot_profile.robot.multirotor
        if multirotor is None:
            return None
        thrust_limits = multirotor.actuator.thrust_limits
        return float(thrust_limits[0]), float(thrust_limits[1])

    def _desired_base(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        controller = getattr(self.env_unwrapped, "controller", None)
        x_d = getattr(controller, "x_d", None)
        if x_d is None:
            return None, None
        desired = _vector(x_d[self.env_idx])
        return desired[0:3] - self._env_origin(), desired[3:7]

    def _desired_ee(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        desired_pos = getattr(self.env_unwrapped, "ee_cmd_pos_w", None)
        desired_quat = getattr(self.env_unwrapped, "ee_cmd_quat_w", None)
        if desired_pos is None or desired_quat is None:
            return None, None
        return _vector(desired_pos[self.env_idx]) - self._env_origin(), _vector(desired_quat[self.env_idx])

    def _desired_joint(self) -> np.ndarray | None:
        targets = getattr(self.env_unwrapped, "arm_targets", None)
        if targets is None:
            return None
        return _vector(targets[self.env_idx])

    def _desired_gripper(self) -> float:
        targets = getattr(self.env_unwrapped, "gripper_targets", None)
        if targets is None:
            return float("nan")
        return float(np.sum(_vector(targets[self.env_idx])))

    def _motor(self) -> tuple[np.ndarray | None, np.ndarray | None, bool | None]:
        output = getattr(self.env_unwrapped, "control_output", None)
        thrusts = None if output is None else output.motor_thrusts
        if thrusts is None:
            return None, None, None
        commanded_thrusts = getattr(output, "commanded_motor_thrusts", None)
        commanded_motor_thrusts = None if commanded_thrusts is None else _vector(commanded_thrusts[self.env_idx])
        motor_thrusts = _vector(thrusts[self.env_idx])
        motor_saturated = None
        if self.thrust_limits is not None:
            min_thrust, max_thrust = self.thrust_limits
            atol = 1.0e-5
            motor_saturated = bool(
                np.any(motor_thrusts <= min_thrust + atol) or np.any(motor_thrusts >= max_thrust - atol)
            )
        actuator_output = getattr(output, "rotor_actuator_output", None)
        if actuator_output is not None:
            thrust_saturated = _vector(actuator_output.thrust_saturated[self.env_idx], dtype=np.bool_)
            motor_saturated = bool(np.any(thrust_saturated)) or bool(motor_saturated)
        return commanded_motor_thrusts, motor_thrusts, motor_saturated

    def _body_wrench(self) -> np.ndarray | None:
        output = getattr(self.env_unwrapped, "control_output", None)
        wrench = None if output is None else output.final_wrench_b
        if wrench is None:
            return None
        return _vector(wrench[self.env_idx])

    def collect(self, raw_obs: dict[str, Any], timestep: int) -> dict[str, Any]:
        obs_policy = raw_obs["policy"][self.env_idx]
        actual_ee_pos = _vector(obs_policy.get("ee_pos"))
        actual_ee_quat = _vector(obs_policy.get("ee_quat"))
        actual_base_pos = _vector(obs_policy.get("base_pos"))
        actual_base_quat = _vector(obs_policy.get("base_quat"))
        actual_joint_pos = _vector(obs_policy.get("arm_joint_pos"))
        actual_gripper_width = float(np.sum(_vector(obs_policy.get("gripper_width"))))

        desired_ee_pos, desired_ee_quat = self._desired_ee()
        desired_base_pos, desired_base_quat = self._desired_base()
        desired_joint_pos = self._desired_joint()
        desired_gripper_width = self._desired_gripper()

        ee_pos_error = (
            desired_ee_pos - actual_ee_pos if desired_ee_pos is not None and actual_ee_pos is not None else None
        )
        base_pos_error = (
            desired_base_pos - actual_base_pos if desired_base_pos is not None and actual_base_pos is not None else None
        )
        joint_error = (
            desired_joint_pos - actual_joint_pos
            if desired_joint_pos is not None and actual_joint_pos is not None
            else None
        )

        base_cmd_step_pos_norm = float("nan")
        base_cmd_step_rot_rad = float("nan")
        joint_cmd_step_l2 = float("nan")
        joint_cmd_step_max_abs = float("nan")
        if self.prev_desired_base_pos is not None and desired_base_pos is not None:
            base_cmd_step_pos_norm = float(np.linalg.norm(desired_base_pos - self.prev_desired_base_pos))
        if self.prev_desired_base_quat is not None and desired_base_quat is not None:
            base_cmd_step_rot_rad = _quat_error_rad(desired_base_quat, self.prev_desired_base_quat)
        if self.prev_desired_joint_pos is not None and desired_joint_pos is not None and desired_joint_pos.size > 0:
            joint_step = desired_joint_pos - self.prev_desired_joint_pos
            joint_cmd_step_l2 = float(np.linalg.norm(joint_step))
            joint_cmd_step_max_abs = float(np.max(np.abs(joint_step)))

        base_tilt = _tilt_from_quat_wxyz(actual_base_quat)
        commanded_motor_thrusts, motor_thrusts, motor_saturated = self._motor()
        control_output = getattr(self.env_unwrapped, "control_output", None)
        actuator_output = None if control_output is None else getattr(control_output, "rotor_actuator_output", None)
        actuator_env = None
        if actuator_output is not None:
            actuator_env = {
                "thrust_commands": actuator_output.commanded_thrust[self.env_idx],
                "normalized_speed_commands": actuator_output.commanded_normalized_speed[self.env_idx],
                "normalized_accelerations": actuator_output.normalized_acceleration[self.env_idx],
                "normalized_speeds": actuator_output.normalized_speed[self.env_idx],
                "thrust_saturated": actuator_output.thrust_saturated[self.env_idx],
                "acceleration_limited": actuator_output.acceleration_limited[self.env_idx],
            }
        has_arm_reach = np.isfinite(self.arm_reach) and self.arm_reach > 0.0

        record = {
            "timestep": int(timestep),
            "dt": float(getattr(self.env_unwrapped, "dt", float("nan"))),
            "actual": {
                "ee_pos": _float_list(actual_ee_pos),
                "ee_quat_wxyz": _float_list(actual_ee_quat),
                "base_pos": _float_list(actual_base_pos),
                "base_quat_wxyz": _float_list(actual_base_quat),
                "joint_pos": _float_list(actual_joint_pos),
                "gripper_width_m": actual_gripper_width,
            },
            "desired": {
                "ee_pos": _float_list(desired_ee_pos),
                "ee_quat_wxyz": _float_list(desired_ee_quat),
                "base_pos": _float_list(desired_base_pos),
                "base_quat_wxyz": _float_list(desired_base_quat),
                "joint_pos": _float_list(desired_joint_pos),
                "gripper_width_m": desired_gripper_width,
            },
            "error": {
                "ee_pos": _float_list(ee_pos_error),
                "ee_pos_norm_m": float(np.linalg.norm(ee_pos_error)) if ee_pos_error is not None else float("nan"),
                "ee_rot_rad": _quat_error_rad(desired_ee_quat, actual_ee_quat),
                "base_pos": _float_list(base_pos_error),
                "base_pos_norm_m": (
                    float(np.linalg.norm(base_pos_error)) if base_pos_error is not None else float("nan")
                ),
                "base_rot_rad": _quat_error_rad(desired_base_quat, actual_base_quat),
                "joint": _float_list(joint_error),
                "joint_l2_rad": (
                    float(np.linalg.norm(joint_error))
                    if joint_error is not None and joint_error.size > 0
                    else float("nan")
                ),
                "joint_abs_max_rad": (
                    float(np.max(np.abs(joint_error)))
                    if joint_error is not None and joint_error.size > 0
                    else float("nan")
                ),
                "joint_mean_abs_rad": (
                    float(np.mean(np.abs(joint_error)))
                    if joint_error is not None and joint_error.size > 0
                    else float("nan")
                ),
                "gripper_width_m": desired_gripper_width - actual_gripper_width,
                "normalized_ee_error": (
                    float(np.linalg.norm(ee_pos_error) / self.arm_reach)
                    if has_arm_reach and ee_pos_error is not None
                    else float("nan")
                ),
                "normalized_base_deviation": (
                    float(np.linalg.norm(base_pos_error) / self.arm_reach)
                    if has_arm_reach and base_pos_error is not None
                    else float("nan")
                ),
            },
            "control": {
                "base_tilt_rad": base_tilt,
                "base_cmd_step_pos_norm_m": base_cmd_step_pos_norm,
                "base_cmd_step_rot_rad": base_cmd_step_rot_rad,
                "joint_cmd_step_l2_rad": joint_cmd_step_l2,
                "joint_cmd_step_max_abs_rad": joint_cmd_step_max_abs,
                "commanded_motor_thrusts": _float_list(commanded_motor_thrusts),
                "motor_thrusts": _float_list(motor_thrusts),
                "motor_min_thrust": (
                    float(np.min(motor_thrusts))
                    if motor_thrusts is not None and motor_thrusts.size > 0
                    else float("nan")
                ),
                "motor_max_thrust": (
                    float(np.max(motor_thrusts))
                    if motor_thrusts is not None and motor_thrusts.size > 0
                    else float("nan")
                ),
                "motor_mean_thrust": (
                    float(np.mean(motor_thrusts))
                    if motor_thrusts is not None and motor_thrusts.size > 0
                    else float("nan")
                ),
                "motor_saturated": motor_saturated,
                "motor_thrust_limits": None if self.thrust_limits is None else list(self.thrust_limits),
                "actuator_thrust_commands": (
                    None if actuator_env is None else _float_list(actuator_env["thrust_commands"])
                ),
                "rotor_normalized_speed_commands": (
                    None if actuator_env is None else _float_list(actuator_env["normalized_speed_commands"])
                ),
                "rotor_normalized_accelerations_per_s": (
                    None if actuator_env is None else _float_list(actuator_env["normalized_accelerations"])
                ),
                "rotor_normalized_speeds": (
                    None if actuator_env is None else _float_list(actuator_env["normalized_speeds"])
                ),
                "actuator_thrust_saturated": (
                    None if actuator_env is None else _bool_list(actuator_env["thrust_saturated"])
                ),
                "actuator_acceleration_limited": (
                    None if actuator_env is None else _bool_list(actuator_env["acceleration_limited"])
                ),
                "body_wrench": _float_list(self._body_wrench()),
                "arm_reach_m": self.arm_reach,
            },
        }

        self.prev_desired_base_pos = None if desired_base_pos is None else desired_base_pos.copy()
        self.prev_desired_base_quat = None if desired_base_quat is None else desired_base_quat.copy()
        self.prev_desired_joint_pos = None if desired_joint_pos is None else desired_joint_pos.copy()
        return record
