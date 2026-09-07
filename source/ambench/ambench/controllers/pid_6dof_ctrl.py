# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""6DOF Geometric PID Controller for SE(3) tracking.

This module provides a universal 6DOF PID controller that can be used across different platforms.
The controller tracks position and orientation (SE(3)) using geometric control theory.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation

from ambench.controllers.controller_cfg import BaseController
from ambench.controllers.utils.control_utils import get_aggregate_physical_parameters


@dataclass
class PID6DOFGains:
    """PID gains for 6DOF control.

    Attributes:
        kp_pos: Position proportional gain (default: 100.0)
        kd_pos: Position derivative gain (default: 80.0)
        ki_pos: Position integral gain (default: 50.0)
        kp_rot: Rotation proportional gain (default: 300.0)
        kd_rot: Rotation derivative gain (default: 100.0)
        ki_rot: Rotation integral gain (default: 120.0)
    """

    kp_pos: float = 100.0
    kd_pos: float = 80.0
    ki_pos: float = 50.0
    kp_rot: float = 300.0
    kd_rot: float = 100.0
    ki_rot: float = 120.0


@dataclass
class ControlLimits:
    """Control output limits.

    Attributes:
        max_force: Maximum force in N (default: None, no limit)
        max_torque: Maximum torque in N⋅m (default: None, no limit)
        integral_windup_limit: Maximum integral term magnitude for anti-windup (default: None, no limit)
    """

    max_force: float | None = None
    max_torque: float | None = None
    integral_windup_limit: float | None = None


class PID6DOFController(BaseController):
    """6DOF Geometric PID Controller for SE(3) tracking.

    This controller implements a geometric PID controller for tracking position and orientation
    in SE(3). It uses quaternion-based attitude control and supports integral anti-windup.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        gains: PID6DOFGains | None = None,
        control_limits: ControlLimits | None = None,
        robot: Articulation | None = None,
        scene=None,
        robot_spec=None,
        **kwargs,
    ):
        """Initialize the 6DOF PID controller.

        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on ("cuda" or "cpu")
            gains: PID gains configuration. If None, uses default gains.
            control_limits: Control output limits. If None, no limits applied.
            robot: Articulation object. Required to compute physical parameters (mass, inertia) from simulator.
            scene: Scene object used to read gravity from the simulation.
            robot_spec: Robot morphology and multirotor specification.
            **kwargs: Additional arguments passed to BaseController
        """
        if robot is None:
            raise ValueError("robot parameter is required. Physical parameters are computed from the simulator.")

        # Initialize BaseController first
        super().__init__(
            num_envs=num_envs,
            device=device,
            dt=dt,
            robot_spec=robot_spec,
            robot=robot,
            scene=scene,
            **kwargs,
        )

        # Initialize gains
        self.gains = gains if gains is not None else PID6DOFGains()

        # Get mass, inertia, and gravity from simulator
        total_mass, total_inertia_matrix, gravity = get_aggregate_physical_parameters(robot, scene=scene)
        self.mass = total_mass
        self.inertia_matrix = total_inertia_matrix.to(self.device)
        # Extract diagonal inertia for compatibility
        self.inertia = torch.diag(self.inertia_matrix)
        self.gravity = gravity.to(self.device)

        # Initialize control limits
        self.control_limits = control_limits if control_limits is not None else ControlLimits()

        # Initialize integral state
        self.integral_state = {
            "pos": torch.zeros((num_envs, 3), device=self.device),
            "att": torch.zeros((num_envs, 3), device=self.device),
        }

        # Store latest computed control outputs and state
        self.force = torch.zeros((num_envs, 3), device=self.device)
        self.torque = torch.zeros((num_envs, 3), device=self.device)
        self.pos = torch.zeros((num_envs, 3), device=self.device)
        self.quat = torch.zeros((num_envs, 4), device=self.device)
        self.quat[:, 0] = 1.0  # Initialize to identity quaternion [1, 0, 0, 0]

        # Desired state for PID controller [pos(3), quat(4), lin_vel(3), ang_vel(3), acc(3), ang_acc(3)]
        self.x_d = torch.zeros((num_envs, 19), device=self.device)
        self.x_d[:, 3] = 1.0  # Initialize desired quaternion to [1, 0, 0, 0]

        # Previous desired state for numerical differentiation
        # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        self.x_d_prev = torch.zeros((num_envs, 13), device=self.device)
        self.x_d_prev[:, 3] = 1.0  # Initialize previous desired quaternion to [1, 0, 0, 0]

    def reset(self, default_root_state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state.

        Args:
            env_ids: Optional tensor of environment IDs to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            # Reset all environments
            self.integral_state["pos"].zero_()
            self.integral_state["att"].zero_()
        else:
            # Reset specific environments
            self.integral_state["pos"][env_ids] = 0.0
            self.integral_state["att"][env_ids] = 0.0

        # Reset control outputs and state
        self.force[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.pos[env_ids] = default_root_state[:, 0:3]
        self.quat[env_ids] = default_root_state[:, 3:7]

        # Initialize desired state to current root state
        self.x_d[env_ids, 0:3] = default_root_state[:, 0:3]  # position
        self.x_d[env_ids, 3:7] = default_root_state[:, 3:7]  # quaternion
        self.x_d[env_ids, 7:] = 0.0  # velocities and accelerations

        # Reset previous desired state for numerical differentiation
        self.x_d_prev[env_ids, 0:3] = default_root_state[:, 0:3]  # position
        self.x_d_prev[env_ids, 3:7] = default_root_state[:, 3:7]  # quaternion
        self.x_d_prev[env_ids, 7:10] = 0.0  # lin_vel_d
        self.x_d_prev[env_ids, 10:13] = 0.0  # ang_vel_d

    def set_gains(
        self,
        kp_pos: float | None = None,
        kd_pos: float | None = None,
        ki_pos: float | None = None,
        kp_rot: float | None = None,
        kd_rot: float | None = None,
        ki_rot: float | None = None,
    ) -> None:
        """Update PID gains.

        Args:
            kp_pos: Position proportional gain
            kd_pos: Position derivative gain
            ki_pos: Position integral gain
            kp_rot: Rotation proportional gain
            kd_rot: Rotation derivative gain
            ki_rot: Rotation integral gain
        """
        if kp_pos is not None:
            self.gains.kp_pos = kp_pos
        if kd_pos is not None:
            self.gains.kd_pos = kd_pos
        if ki_pos is not None:
            self.gains.ki_pos = ki_pos
        if kp_rot is not None:
            self.gains.kp_rot = kp_rot
        if kd_rot is not None:
            self.gains.kd_rot = kd_rot
        if ki_rot is not None:
            self.gains.ki_rot = ki_rot

    def set_physical_parameters(
        self,
        mass: float | None = None,
        inertia: torch.Tensor | list[float] | None = None,
        gravity: torch.Tensor | list[float] | None = None,
    ) -> None:
        """Update physical parameters.

        Args:
            mass: Total mass in kg
            inertia: Inertia tensor (3x3) or diagonal inertia [Ixx, Iyy, Izz] in kg⋅m²
            gravity: Gravity vector in world frame
        """
        if mass is not None:
            self.mass = mass
        if inertia is not None:
            if isinstance(inertia, list):
                inertia = torch.tensor(inertia, device=self.device)
            elif isinstance(inertia, torch.Tensor):
                inertia = inertia.to(self.device)
            self.inertia = inertia
            # Update inertia matrix
            if self.inertia.ndim == 1:
                self.inertia_matrix = torch.diag(self.inertia)
            else:
                self.inertia_matrix = self.inertia
        if gravity is not None:
            if isinstance(gravity, list):
                gravity = torch.tensor(gravity, device=self.device)
            elif isinstance(gravity, torch.Tensor):
                gravity = gravity.to(self.device)
            self.gravity = gravity

    def compute_desired_wrench(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute desired wrench from observations.

        This method implements the PID control law to compute the desired 6DOF wrench
        (force + torque) in body frame.

        Args:
            obs: Current state tensor of shape (num_envs, 13)
                 [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                 - pos: position in world frame
                 - quat: quaternion [w, x, y, z] in world frame
                 - lin_vel: linear velocity in world frame
                 - ang_vel: angular velocity in body frame

        Returns:
            desired_wrench_b: Desired wrench in body frame, shape (num_envs, 6)
                            Format: [fx, fy, fz, tx, ty, tz]
        """
        # Validate input shapes
        if obs.shape[0] != self.num_envs:
            raise ValueError(f"obs batch size {obs.shape[0]} does not match num_envs {self.num_envs}")

        # Unpack current state
        pos = obs[:, 0:3]  # [num_envs, 3]
        quat = obs[:, 3:7]  # [num_envs, 4]
        lin_vel = obs[:, 7:10]  # [num_envs, 3]
        ang_vel = obs[:, 10:13]  # [num_envs, 3]

        # Store current state
        self.pos = pos
        self.quat = quat

        # Unpack desired state
        pos_d = self.x_d[:, 0:3]  # [num_envs, 3]
        quat_d = self.x_d[:, 3:7]  # [num_envs, 4]
        lin_vel_d = self.x_d[:, 7:10]  # [num_envs, 3]
        ang_vel_d = self.x_d[:, 10:13]  # [num_envs, 3]
        acc_d = self.x_d[:, 13:16]  # [num_envs, 3]
        ang_acc_d = self.x_d[:, 16:19]  # [num_envs, 3]

        # Position loop: pos, lin_vel, lin_vel_d, and acc_d are all in world frame.
        pos_err = pos_d - pos
        lin_vel_err = lin_vel_d - lin_vel

        # Update integral term with anti-windup
        self.integral_state["pos"] += pos_err * self.dt
        if self.control_limits.integral_windup_limit is not None:
            integral_norm = torch.norm(self.integral_state["pos"], dim=1, keepdim=True)
            exceeds = integral_norm > self.control_limits.integral_windup_limit
            self.integral_state["pos"] *= torch.where(
                exceeds,
                self.control_limits.integral_windup_limit / (integral_norm + 1e-8),
                torch.ones_like(integral_norm),
            )

        # Position control: force = m * (acc_d + kp*pos_err + kd*lin_vel_err + ki*integral + gravity)
        # Note: force is computed in world frame (gravity is in world frame)
        force_w = self.mass * (
            acc_d
            + self.gains.kp_pos * pos_err
            + self.gains.kd_pos * lin_vel_err
            + self.gains.ki_pos * self.integral_state["pos"]
            + self.gravity.expand(self.num_envs, 3)
        )

        # Apply force limits (in world frame)
        if self.control_limits.max_force is not None:
            force_norm = torch.norm(force_w, dim=1, keepdim=True)
            exceeds = force_norm > self.control_limits.max_force
            force_w *= torch.where(
                exceeds, self.control_limits.max_force / (force_norm + 1e-8), torch.ones_like(force_norm)
            )

        # Convert force from world frame to body frame for wrench
        R_wb = math_utils.matrix_from_quat(quat)  # Rotation from body to world
        R_bw = R_wb.transpose(1, 2)  # Rotation from world to body
        force_b = torch.bmm(R_bw, force_w.unsqueeze(-1)).squeeze(-1)  # [num_envs, 3]

        # Attitude control
        R = math_utils.matrix_from_quat(quat)
        R_d = math_utils.matrix_from_quat(quat_d)

        R_d_T_R = torch.bmm(R_d.transpose(1, 2), R)  # R_d^T * R
        R_T_R_d = torch.bmm(R.transpose(1, 2), R_d)  # R^T * R_d
        R_err_diff = R_d_T_R - R_T_R_d  # R_d^T * R - R^T * R_d

        # Rotation error vector (vee map)
        e_R = 0.5 * torch.stack(
            [
                R_err_diff[:, 2, 1] - R_err_diff[:, 1, 2],
                R_err_diff[:, 0, 2] - R_err_diff[:, 2, 0],
                R_err_diff[:, 1, 0] - R_err_diff[:, 0, 1],
            ],
            dim=1,
        )

        # Angular velocity error: eOmega = Omega - R^T*R_d*Omegac
        # Compute R^T*R_d*ang_vel_d (transform desired angular velocity from R_d body frame to current body frame)
        R_T_R_d_ang_vel_d = torch.bmm(R.transpose(1, 2), torch.bmm(R_d, ang_vel_d.unsqueeze(-1))).squeeze(-1)
        ang_vel_err = ang_vel - R_T_R_d_ang_vel_d

        # Update integral term with anti-windup
        self.integral_state["att"] += e_R * self.dt
        if self.control_limits.integral_windup_limit is not None:
            integral_norm = torch.norm(self.integral_state["att"], dim=1, keepdim=True)
            exceeds = integral_norm > self.control_limits.integral_windup_limit
            self.integral_state["att"] *= torch.where(
                exceeds,
                self.control_limits.integral_windup_limit / (integral_norm + 1e-8),
                torch.ones_like(integral_norm),
            )

        # Inertia matrix
        J = self.inertia_matrix.expand(self.num_envs, 3, 3) if self.inertia_matrix.ndim == 2 else self.inertia_matrix

        # Torque: -kR*eR - kOmega*eOmega + Coriolis - J*feedforward
        pid_term = (
            -self.gains.kp_rot * e_R - self.gains.kd_rot * ang_vel_err - self.gains.ki_rot * self.integral_state["att"]
        )
        torque = torch.bmm(J, pid_term.unsqueeze(-1)).squeeze(-1)
        # Coriolis term: cross(Omega, J*Omega)
        torque += torch.cross(ang_vel, torch.bmm(J, ang_vel.unsqueeze(-1)).squeeze(-1), dim=1)
        # Feedforward term: hat(Omega)*R^T*R_d*Omegac - R^T*R_d*Omegac_1dot
        R_T_R_d_ang_acc_d = torch.bmm(R.transpose(1, 2), torch.bmm(R_d, ang_acc_d.unsqueeze(-1))).squeeze(-1)
        feedforward = (
            torch.bmm(math_utils.skew_symmetric_matrix(ang_vel), R_T_R_d_ang_vel_d.unsqueeze(-1)).squeeze(-1)
            - R_T_R_d_ang_acc_d
        )
        torque -= torch.bmm(J, feedforward.unsqueeze(-1)).squeeze(-1)

        # Apply torque limits
        if self.control_limits.max_torque is not None:
            torque_norm = torch.norm(torque, dim=1, keepdim=True)
            exceeds = torque_norm > self.control_limits.max_torque
            torque *= torch.where(
                exceeds, self.control_limits.max_torque / (torque_norm + 1e-8), torch.ones_like(torque_norm)
            )

        # Store computed control outputs (for backward compatibility)
        self.force = force_b  # Store in body frame
        self.torque = torque

        # Concatenate force and torque into wrench [fx, fy, fz, tx, ty, tz] in body frame
        desired_wrench_b = torch.cat([force_b, torque], dim=1)  # [num_envs, 6]

        return desired_wrench_b

    def get_integral_state(self) -> dict[str, torch.Tensor]:
        """Get current integral state.

        Returns:
            Dictionary with 'pos' and 'att' integral terms
        """
        return self.integral_state.copy()

    def set_integral_state(self, integral_state: dict[str, torch.Tensor]) -> None:
        """Set integral state (useful for warm-starting or loading saved state).

        Args:
            integral_state: Dictionary with 'pos' and 'att' integral terms
        """
        if "pos" in integral_state:
            self.integral_state["pos"] = integral_state["pos"].to(self.device)
        if "att" in integral_state:
            self.integral_state["att"] = integral_state["att"].to(self.device)

    def compute_desired_states(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        is_first_step: torch.Tensor,
    ) -> None:
        """Compute desired states.

        Args:
            target_pos: Desired position in world frame, shape (num_envs, 3).
            target_quat: Desired orientation quaternion (w, x, y, z) in world frame, shape (num_envs, 4).
            is_first_step: Boolean mask indicating first step after reset for each env.
        """

        target_quat = math_utils.normalize(target_quat)

        # Store previous desired state
        pos_d_prev = self.x_d_prev[:, 0:3]
        quat_prev = self.x_d_prev[:, 3:7]
        lin_vel_d_prev = self.x_d_prev[:, 7:10]
        ang_vel_d_prev = self.x_d_prev[:, 10:13]

        # Update current desired position and orientation
        self.x_d[:, 0:3] = target_pos
        self.x_d[:, 3:7] = target_quat

        # Compute velocities and accelerations numerically using finite differences
        # For first step, set velocities and accelerations to zero
        lin_vel_d = torch.zeros_like(target_pos)
        ang_vel_d = torch.zeros((self.num_envs, 3), device=self.device)
        acc_d = torch.zeros_like(target_pos)
        ang_acc_d = torch.zeros((self.num_envs, 3), device=self.device)

        # Compute velocities if not first step
        not_first = ~is_first_step
        if not_first.any():
            # Linear velocity: v = (pos_current - pos_d_previous) / dt
            lin_vel_d[not_first] = (target_pos[not_first] - pos_d_prev[not_first]) / self.dt

            # Angular velocity from quaternion difference using quaternion logarithm
            # Reference: https://arxiv.org/pdf/1711.02508
            # Compute relative quaternion: q_rel = q_curr * q_prev^-1
            quat_prev_inv = math_utils.quat_conjugate(quat_prev[not_first])
            quat_rel = math_utils.quat_mul(target_quat[not_first], quat_prev_inv)

            # Extract quaternion components: q = [w, x, y, z]
            quat_rel_w = quat_rel[:, 0:1]  # [num_envs, 1]
            quat_rel_xyz = quat_rel[:, 1:4]  # [num_envs, 3] - vector part
            quat_rel_xyz_norm = torch.norm(quat_rel_xyz, dim=1, keepdim=True)

            # Quaternion logarithm: log(q) = (θ/|v|) * v, where θ = 2*atan2(|v|, w), v = [x,y,z]
            # For small rotations: log(q) ≈ v (when |v| is small)
            # Angular velocity: ω = 2 * log(q_rel) / dt
            # Using atan2 is more numerically stable than acos
            small_angle_threshold = 1e-6

            # Compute rotation angle: θ = 2 * atan2(|v|, w)
            # This handles all cases including when w < 0 (rotation > 180°)
            angle = 2.0 * torch.atan2(quat_rel_xyz_norm, quat_rel_w + 1e-8)

            # Compute quaternion logarithm
            # For small angles: log(q) ≈ v (when |v| is very small, angle ≈ 2*|v|)
            # For larger angles: log(q) = (θ/|v|) * v
            log_quat = torch.where(
                quat_rel_xyz_norm > small_angle_threshold,
                (angle / (quat_rel_xyz_norm + 1e-8)) * quat_rel_xyz,
                quat_rel_xyz,  # Small angle approximation: log(q) ≈ v when |v| << 1
            )

            # Angular velocity: ω = 2 * log(q_rel) / dt
            ang_vel_computed = 2.0 * log_quat / self.dt
            ang_vel_d[not_first] = ang_vel_computed

            # Compute accelerations: a = (v_current - v_previous) / dt
            acc_d[not_first] = (lin_vel_d[not_first] - lin_vel_d_prev[not_first]) / self.dt
            ang_acc_d[not_first] = (ang_vel_d[not_first] - ang_vel_d_prev[not_first]) / self.dt

        # Bound numerically differentiated targets before updating desired state.
        self.x_d[:, 7:10] = lin_vel_d.clamp(-1.0, 1.0)
        self.x_d[:, 10:13] = ang_vel_d.clamp(-1.0, 1.0)
        self.x_d[:, 13:16] = acc_d.clamp(-1.0, 1.0)
        self.x_d[:, 16:19] = ang_acc_d.clamp(-1.0, 1.0)

        # Update previous state for next iteration
        self.x_d_prev[:, 0:3] = target_pos
        self.x_d_prev[:, 3:7] = target_quat
        self.x_d_prev[:, 7:10] = lin_vel_d
        self.x_d_prev[:, 10:13] = ang_vel_d
