# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

from ambench.robots.robot_cfg import RobotSpecCfg, RotorLayout
from ambench.robots.rotor_actuator import RotorActuator, RotorActuatorOutput


@configclass
class ControllerCfg:
    """Algorithm class and constructor parameters for a controller."""

    class_type: type | Any = MISSING
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerOutput:
    """Public output and telemetry produced by one controller step."""

    force_b: torch.Tensor
    torque_b: torch.Tensor
    arm_targets: torch.Tensor | None = None
    commanded_motor_thrusts: torch.Tensor | None = None
    motor_thrusts: torch.Tensor | None = None
    motor_arm_angles: torch.Tensor | None = None
    rotor_actuator_output: RotorActuatorOutput | None = None
    desired_wrench_b: torch.Tensor | None = None
    final_wrench_b: torch.Tensor | None = None
    ground_distances: torch.Tensor | None = None
    wall_distances: torch.Tensor | None = None
    ground_effect_wrench_b: torch.Tensor | None = None
    wall_effect_wrench_b: torch.Tensor | None = None
    drag_wrench_b: torch.Tensor | None = None
    wind_wrench_b: torch.Tensor | None = None


class BaseController(ABC):
    """Abstract base class for all controllers."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        robot_spec: RobotSpecCfg | None = None,
        rotor_layout: RotorLayout | None = None,
        rotor_actuator: RotorActuator | None = None,
        robot: Articulation | None = None,
        scene: InteractiveScene | None = None,
        enable_saturation: bool = False,
        enable_aerodynamic_effects: bool = True,
        enable_wind_effect: bool = False,
        **kwargs: Any,
    ):
        """Initialize base controller.

        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on
            dt: Time step in seconds
            robot_spec: Robot morphology and multirotor specification
            rotor_layout: Resolved rotor geometry shared with simulator-facing robot I/O
            rotor_actuator: Optional reduced-order actuator state owned by robot I/O
            robot: Robot articulation used for physical parameters and allocation
            scene: Scene object (optional, for aerodynamic effects)
            enable_saturation: Enable/disable thrust saturation (thrust limits). Required when rotor dynamics are used.
            enable_aerodynamic_effects: Enable/disable aerodynamic effects (ground effect, wall effect)
            enable_wind_effect: Enable/disable wind effect (constant wind force)
            **kwargs: Additional arguments passed to subclasses
        """
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dt = dt
        if robot_spec is None:
            raise ValueError("robot_spec is required.")
        self.robot_spec = robot_spec
        if robot_spec.multirotor is not None and rotor_layout is None:
            raise ValueError("rotor_layout is required for multirotor controllers.")
        if rotor_actuator is not None and robot_spec.multirotor is None:
            raise ValueError("rotor_actuator requires a multirotor specification.")
        if rotor_actuator is not None:
            if rotor_layout is None:
                raise ValueError("rotor_actuator requires a resolved rotor_layout.")
            if not enable_saturation:
                raise ValueError("Rotor actuator dynamics require enable_saturation=True.")
            if rotor_actuator.num_envs != num_envs or rotor_actuator.num_rotors != rotor_layout.num_rotors:
                raise ValueError("rotor_actuator batch dimensions must match the controller and rotor layout.")
            if rotor_actuator.device != self.device:
                raise ValueError("rotor_actuator and controller must use the same device.")
        self.rotor_layout = rotor_layout
        self.rotor_actuator = rotor_actuator
        self.robot = robot
        self.scene = scene

        self.enable_saturation = enable_saturation
        self.enable_aerodynamic_effects = enable_aerodynamic_effects
        self.enable_wind_effect = enable_wind_effect

        # Storage for pipeline intermediate values
        self._desired_wrench = None
        self._commanded_motor_thrust_magnitudes = None
        self._motor_thrust_magnitudes = None
        self._motor_arm_angles: torch.Tensor | None = None
        self._rotor_actuator_output: RotorActuatorOutput | None = None
        self._motor_arm_joint_ids: list[int] = []
        self._final_wrench = None
        self._d_ground = None
        self._d_wall = None
        self._ground_effect_wrench_b = None
        self._wall_effect_wrench_b = None
        self._drag_wrench_b = None
        self.last_output: ControllerOutput | None = None

    @abstractmethod
    def compute_desired_wrench(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute desired wrench from observations.

        This is the controller-specific method that should be implemented by subclasses.
        It computes the desired 6DOF wrench (force + torque) based on the current state.

        Args:
            obs: Current state tensor [num_envs, state_dim]

        Returns:
            desired_wrench: Desired wrench in body frame [num_envs, 6]
                          Format: [fx, fy, fz, tx, ty, tz]
        """

    def compute(self, obs: torch.Tensor) -> ControllerOutput:
        """Template method for control pipeline.

        This implements the control pipeline:
        1. Compute desired wrench (or force/torque for MPC)
        2. Inverse allocation + Saturation (wrench -> rotor thrusts)
        3. Optional rotor actuator dynamics
        4. Aerodynamic effects
        5. Allocation (rotor thrusts -> final wrench)
        6. Drag force calculation
        7. Wind effect

        For MPC controllers, if compute_mpc method exists, it will be called and return 3 values (force, torque, arm_targets).

        Args:
            obs: Current state tensor [num_envs, state_dim]

        Returns:
            Public controller output containing the applied wrench, optional
            joint/motor targets, and optional allocation/aerodynamic telemetry.
        """
        # Check if controller has compute_mpc method (MPC controllers)
        arm_targets = None
        if hasattr(self, "compute_mpc"):
            force_b, torque_b, arm_targets = self.compute_mpc(obs)
        else:
            # Step 1: Compute desired wrench (controller-specific)
            desired_wrench_b = self.compute_desired_wrench(obs)  # [num_envs, 6]
            self._desired_wrench = desired_wrench_b
            # Extract force and torque from wrench
            force_b = desired_wrench_b[:, 0:3]  # [num_envs, 3]
            torque_b = desired_wrench_b[:, 3:6]  # [num_envs, 3]

        # Convert force and torque to wrench format
        wrench_b = torch.cat([force_b, torque_b], dim=1)  # [num_envs, 6]
        self._desired_wrench = wrench_b

        # If no platform config, return directly
        if self.robot_spec.multirotor is None:
            output = ControllerOutput(
                force_b=force_b,
                torque_b=torque_b,
                arm_targets=arm_targets,
                desired_wrench_b=wrench_b,
                final_wrench_b=wrench_b,
            )
            self.last_output = output
            return output

        # Step 2: Inverse allocation (wrench -> commanded rotor thrust vectors)
        rotor_thrust_vectors, motor_thrust_magnitudes = self._apply_inverse_allocation(wrench_b)
        self._commanded_motor_thrust_magnitudes = motor_thrust_magnitudes
        rotor_axes_b = None
        self._rotor_actuator_output = None
        if self.rotor_actuator is not None:
            layout = self.rotor_layout
            if layout is None:
                raise RuntimeError("Rotor actuator execution requires a resolved rotor layout.")
            if layout.is_variable_tilt:
                if self._motor_arm_angles is None or layout.tilt_cos_axes_b is None or layout.tilt_sin_axes_b is None:
                    raise RuntimeError("Variable-tilt actuator execution requires allocated rotor angles and bases.")
                rotor_axes_b = torch.cos(self._motor_arm_angles).unsqueeze(-1) * layout.tilt_cos_axes_b.unsqueeze(
                    0
                ) + torch.sin(self._motor_arm_angles).unsqueeze(-1) * layout.tilt_sin_axes_b.unsqueeze(0)
            else:
                rotor_axes_b = layout.thrust_axes_b.unsqueeze(0).expand(self.num_envs, -1, -1)
            rotor_axis_norms = torch.linalg.vector_norm(rotor_axes_b, dim=-1, keepdim=True)
            rotor_axes_b = rotor_axes_b / rotor_axis_norms.clamp_min(torch.finfo(rotor_axes_b.dtype).eps)
            self._rotor_actuator_output = self.rotor_actuator.step(motor_thrust_magnitudes)
            motor_thrust_magnitudes = self._rotor_actuator_output.thrust
            rotor_thrust_vectors = motor_thrust_magnitudes.unsqueeze(-1) * rotor_axes_b
        self._motor_thrust_magnitudes = motor_thrust_magnitudes

        # Step 3: Aerodynamic effects (ground effect -> near wall effect)
        if self.enable_aerodynamic_effects:
            rotor_thrust_vectors = self._apply_aerodynamics_effects(
                rotor_thrust_vectors,
                rotor_axes_b=rotor_axes_b,
            )

        # Step 4: Forward allocation (rotor thrust vectors -> final wrench)
        final_wrench_b = self._apply_allocation(rotor_thrust_vectors)
        self._final_wrench = final_wrench_b

        # Extract force and torque from final wrench
        force_b = final_wrench_b[:, 0:3]
        torque_b = final_wrench_b[:, 3:6]

        # Step 5: Drag force calculation and Step 6: Wind effect
        base_link_indices, _ = self.robot.find_bodies(self.robot_spec.base_body_name)
        base_link_idx = base_link_indices[0]

        drag_force_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        if self.enable_aerodynamic_effects and self.robot_spec.multirotor.aerodynamics is not None:
            from ambench.disturbance.aerodynamic import compute_drag_force

            body_state = self.robot.data.body_link_state_w[:, base_link_idx, :]
            lin_vel_w = body_state[:, 7:10]
            quat_w = body_state[:, 3:7]
            lin_vel_b = math_utils.quat_apply_inverse(quat_w, lin_vel_w)
            drag_coefficients = torch.tensor(
                self.robot_spec.multirotor.aerodynamics.drag_coefficients,
                device=self.device,
            )
            drag_force_b = compute_drag_force(lin_vel_b, drag_coefficients)

            # Convert drag force to wrench format [fx, fy, fz, tx, ty, tz] (torque is zero for drag)
            drag_torque_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            self._drag_wrench_b = torch.cat([drag_force_b, drag_torque_b], dim=1)  # [num_envs, 6]

        # Step 6: Wind effect (constant wind force in world frame, only if enabled)
        wind_force_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        if self.enable_wind_effect and self.robot_spec.multirotor.aerodynamics is not None:
            # Additional wind effect (constant wind force in world frame)
            body_state = self.robot.data.body_link_state_w[:, base_link_idx, :]
            quat_w = body_state[:, 3:7]
            wind_force_w = torch.tensor(
                self.robot_spec.multirotor.aerodynamics.wind_force_w,
                device=self.device,
                dtype=torch.float32,
            )
            wind_force_w_expanded = wind_force_w.unsqueeze(0).expand(self.num_envs, -1)
            wind_force_b = math_utils.quat_apply_inverse(quat_w, wind_force_w_expanded)
        wind_torque_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        wind_wrench_b = torch.cat([wind_force_b, wind_torque_b], dim=1)

        final_force = force_b + drag_force_b + wind_force_b
        output = ControllerOutput(
            force_b=final_force,
            torque_b=torque_b,
            arm_targets=arm_targets,
            commanded_motor_thrusts=self._commanded_motor_thrust_magnitudes,
            motor_thrusts=self._motor_thrust_magnitudes,
            motor_arm_angles=self._motor_arm_angles,
            rotor_actuator_output=self._rotor_actuator_output,
            desired_wrench_b=self._desired_wrench,
            final_wrench_b=self._final_wrench,
            ground_distances=self._d_ground,
            wall_distances=self._d_wall,
            ground_effect_wrench_b=self._ground_effect_wrench_b,
            wall_effect_wrench_b=self._wall_effect_wrench_b,
            drag_wrench_b=self._drag_wrench_b,
            wind_wrench_b=wind_wrench_b,
        )
        self.last_output = output
        return output

    def _apply_inverse_allocation(self, wrench_b: torch.Tensor) -> torch.Tensor:
        """Apply inverse allocation: wrench -> rotor thrust vectors.

        Args:
            wrench_b: Desired wrench in body frame [num_envs, 6]

        Returns:
            rotor_thrust_vectors: Rotor thrust vectors in body frame [num_envs, num_motors, 3]
                                Each vector is magnitude * direction
        """
        from .utils.control_allocation import ctrl_alloc

        multirotor = self.robot_spec.multirotor
        if multirotor is None or self.rotor_layout is None:
            raise ValueError("A multirotor specification is required for inverse allocation.")
        layout = self.rotor_layout
        actuator = multirotor.actuator

        min_thrust = None
        max_thrust = None
        # Transient actuation owns saturation so it can compare the raw
        # allocation request against the physical thrust command.
        if self.enable_saturation and self.rotor_actuator is None:
            min_thrust, max_thrust = actuator.thrust_limits

        if layout.is_variable_tilt:
            motor_cos_dirs_b = layout.tilt_cos_axes_b
            motor_sin_dirs_b = layout.tilt_sin_axes_b
            if motor_cos_dirs_b is None or motor_sin_dirs_b is None:
                raise RuntimeError("Variable-tilt rotor layout is missing its allocation bases.")
            prev_motor_arm_angles = self._measured_motor_arm_angles()
            rotor_thrust_vectors, motor_thrust_magnitudes, motor_arm_angles = ctrl_alloc(
                target_wrench_b=wrench_b,
                motor_positions_b=layout.positions_b,
                motor_dirs_b=motor_cos_dirs_b,
                k_f=actuator.reaction_torque_ratio,
                spin_dirs=layout.spin_directions,
                min_thrust=min_thrust,
                max_thrust=max_thrust,
                fully_actuated=multirotor.fully_actuated,
                motor_cos_dirs_b=motor_cos_dirs_b,
                motor_sin_dirs_b=motor_sin_dirs_b,
                prev_motor_arm_angles=prev_motor_arm_angles,
            )
            self._motor_arm_angles = motor_arm_angles
            return rotor_thrust_vectors, motor_thrust_magnitudes

        self._motor_arm_angles = None

        rotor_thrust_vectors, motor_thrust_magnitudes, _ = ctrl_alloc(
            target_wrench_b=wrench_b,
            motor_positions_b=layout.positions_b,
            motor_dirs_b=layout.thrust_axes_b,
            k_f=actuator.reaction_torque_ratio,
            spin_dirs=layout.spin_directions,
            min_thrust=min_thrust,
            max_thrust=max_thrust,
            fully_actuated=multirotor.fully_actuated,
        )

        return rotor_thrust_vectors, motor_thrust_magnitudes

    def _apply_aerodynamics_effects(
        self,
        rotor_thrust_vectors: torch.Tensor,
        *,
        rotor_axes_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply aerodynamic effects: ground effect -> near wall effect.

        Args:
            rotor_thrust_vectors: Rotor thrust vectors in body frame [num_envs, num_motors, 3]
            rotor_axes_b: Optional current rotor axes [num_envs, num_motors, 3]

        Returns:
            rotor_thrust_vectors: Modified rotor thrust vectors after aerodynamic effects [num_envs, num_motors, 3]
        """
        multirotor = self.robot_spec.multirotor
        if multirotor is None or multirotor.aerodynamics is None or self.rotor_layout is None:
            return rotor_thrust_vectors

        if self.robot is None or self.scene is None:
            return rotor_thrust_vectors

        from ambench.disturbance.aerodynamic import (
            apply_ground_effect,
            apply_near_wall_effect,
        )

        layout = self.rotor_layout
        actuator = multirotor.actuator
        aerodynamic_coeffs = multirotor.aerodynamics.as_dict(propeller_radius=actuator.propeller_radius)
        motor_positions_b = layout.positions_b
        motor_dirs_b = layout.thrust_axes_b if rotor_axes_b is None else rotor_axes_b

        base_link_indices, _ = self.robot.find_bodies(self.robot_spec.base_body_name)
        base_link_idx = base_link_indices[0]

        body_state = self.robot.data.body_link_state_w[:, base_link_idx, :]
        base_pos_w = body_state[:, 0:3]
        quat_w = body_state[:, 3:7]
        base_rot_wb = math_utils.matrix_from_quat(quat_w)

        # Transform motor positions and directions to world frame
        # base_rot_wb: [num_envs, 3, 3]
        # motor_positions_b: [num_motors, 3] -> [num_envs, num_motors, 3]
        motor_positions_b_expanded = motor_positions_b.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )  # [num_envs, num_motors, 3]
        motor_positions_w = torch.einsum(
            "bij,bmj->bmi", base_rot_wb, motor_positions_b_expanded
        ) + base_pos_w.unsqueeze(1)

        # motor_dirs_b: [num_motors, 3] -> [num_envs, num_motors, 3]
        motor_dirs_b_expanded = (
            motor_dirs_b.unsqueeze(0).expand(self.num_envs, -1, -1) if motor_dirs_b.dim() == 2 else motor_dirs_b
        )
        motor_dirs_w = torch.einsum("bij,bmj->bmi", base_rot_wb, motor_dirs_b_expanded)  # [num_envs, num_motors, 3]

        # Apply ground effect
        rotor_thrust_vectors, d_ground, ground_effect_diff_b = apply_ground_effect(
            rotor_thrust_vectors=rotor_thrust_vectors,
            motor_positions_w=motor_positions_w,
            motor_axis_directions_w=motor_dirs_w,
            aerodynamic_coeffs=aerodynamic_coeffs,
        )

        # Apply near wall effect
        rotor_thrust_vectors, d_wall, wall_effect_diff_b = apply_near_wall_effect(
            rotor_thrust_vectors=rotor_thrust_vectors,
            motor_positions_w=motor_positions_w,
            drone_position_w=base_pos_w,
            base_rot_wb=base_rot_wb,
            aerodynamic_coeffs=aerodynamic_coeffs,
        )

        # Store intermediate values for logging/visualization
        self._d_ground = d_ground
        self._d_wall = d_wall

        # Convert thrust differences to wrenches for disturbance analysis
        from .utils.control_allocation import inv_ctrl_alloc

        if ground_effect_diff_b is not None:
            self._ground_effect_wrench_b = inv_ctrl_alloc(
                rotor_thrust_vectors=ground_effect_diff_b,
                motor_positions_b=motor_positions_b,
                k_f=actuator.reaction_torque_ratio,
                spin_dirs=layout.spin_directions,
            )

        if wall_effect_diff_b is not None:
            self._wall_effect_wrench_b = inv_ctrl_alloc(
                rotor_thrust_vectors=wall_effect_diff_b,
                motor_positions_b=motor_positions_b,
                k_f=actuator.reaction_torque_ratio,
                spin_dirs=layout.spin_directions,
            )

        return rotor_thrust_vectors

    def _apply_allocation(self, rotor_thrust_vectors: torch.Tensor) -> torch.Tensor:
        """Apply forward allocation: rotor thrust vectors -> final wrench.

        Args:
            rotor_thrust_vectors: Rotor thrust vectors in body frame [num_envs, num_motors, 3]

        Returns:
            final_wrench_b: Final wrench in body frame [num_envs, 6]
        """
        multirotor = self.robot_spec.multirotor
        if multirotor is None or self.rotor_layout is None:
            raise ValueError("A multirotor specification is required for forward allocation.")

        from .utils.control_allocation import inv_ctrl_alloc

        layout = self.rotor_layout

        # Compute final wrench from thrust vectors
        final_wrench_b = inv_ctrl_alloc(
            rotor_thrust_vectors=rotor_thrust_vectors,
            motor_positions_b=layout.positions_b,
            k_f=multirotor.actuator.reaction_torque_ratio,
            spin_dirs=layout.spin_directions,
        )

        return final_wrench_b

    def set_motor_arm_joint_ids(self, joint_ids: list[int]) -> None:
        """Register motor-arm joint indices for atan2 unwrap (measured ``joint_pos``)."""
        self._motor_arm_joint_ids = list(joint_ids)

    def _measured_motor_arm_angles(self) -> torch.Tensor | None:
        """Current simulated motor-arm joint positions (same source as onboard ``joint_msg.position``)."""
        if self.robot is None or not self._motor_arm_joint_ids:
            return None
        return self.robot.data.joint_pos[:, self._motor_arm_joint_ids]

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """Reset controller state."""
