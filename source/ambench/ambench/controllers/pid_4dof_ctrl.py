# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""4DOF Geometric PID Controller for SE(3) tracking.

This module provides a universal 4DOF PID controller that can be used across different platforms.
The controller tracks position and yaw using geometric control theory.

Reference: "Control of Complex Maneuvers for a Quadrotor UAV using Geometric Methods on SE(3)"
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation

from ambench.controllers.controller_cfg import BaseController
from ambench.controllers.utils.control_utils import get_aggregate_physical_parameters


@dataclass
class PID4DOFGains:
    """PID gains for position and attitude control."""

    kp_pos_xy: float = 10.0  # Position gain for x, y directions
    kp_pos_z: float = 30.0  # Position gain for z direction
    kd_pos_xy: float = 8.0  # Velocity gain for x, y directions
    kd_pos_z: float = 10.0  # Velocity gain for z direction
    ki_pos_xy: float = 6.0  # Integral gain for x, y directions
    ki_pos_z: float = 10.0  # Integral gain for z direction
    kp_rot: float = 200.0
    kd_rot: float = 120.0
    ki_rot: float = 120.0


@dataclass
class ControlLimits:
    """Control limits for force, torque, and integral windup."""

    max_force: float | None = None
    max_torque: float | None = None
    integral_windup_limit: float | None = None


class PID4DOFController(BaseController):
    """4DOF Geometric PID Controller for SE(3) tracking.

    Uses nested loop control:
    - Outer loop: Position control (updates every N steps)
    - Inner loop: Attitude control (runs every step)
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        gains: PID4DOFGains | None = None,
        control_limits: ControlLimits | None = None,
        robot: Articulation | None = None,
        scene=None,
        robot_spec=None,
        outer_loop_decimation: int = 3,
        **kwargs,
    ):
        """Initialize the 4DOF PID controller.

        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on ("cuda" or "cpu")
            dt: Time step in seconds
            gains: PID gains configuration. If None, uses default gains.
            control_limits: Control output limits. If None, no limits applied.
            robot: Articulation object. Required to compute physical parameters (mass, inertia) from simulator.
            scene: Scene object used to read gravity from the simulation.
            robot_spec: Robot morphology and multirotor specification.
            outer_loop_decimation: Decimation factor for outer loop (position control). Default: 3.
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
        self.gains = gains if gains is not None else PID4DOFGains()

        # Outer loop decimation: position control updates every N steps
        # This allows the rotational dynamics (inner loop) to run faster than position control (outer loop)
        self.outer_loop_decimation: int = max(1, outer_loop_decimation)
        self.outer_loop_dt: float = self.dt * self.outer_loop_decimation
        self._outer_loop_counter: int = 0

        # Get mass, inertia, and gravity from simulator
        total_mass, total_inertia_matrix, gravity = get_aggregate_physical_parameters(robot, scene=scene)
        self.mass = total_mass
        self.inertia_matrix = total_inertia_matrix.to(self.device)
        self.inertia = torch.diag(self.inertia_matrix)
        self.gravity = gravity.to(self.device)

        # Initialize control limits
        self.control_limits = control_limits if control_limits is not None else ControlLimits()

        # Initialize integral state
        self.integral_state = {
            "pos": torch.zeros((num_envs, 3), device=self.device),
            "att": torch.zeros((num_envs, 3), device=self.device),
        }

        # Desired state: [pos(3), quat(4), lin_vel(3), ang_vel(3), acc(3), ang_acc(3)]
        self.x_d = torch.zeros((num_envs, 19), device=self.device)
        self.x_d[:, 3] = 1.0  # Initialize quaternion to identity

        # Previous desired state for numerical differentiation (velocity/acceleration computation)
        self.pos_d_prev = torch.zeros((num_envs, 3), device=self.device)
        self.quat_d_prev = torch.zeros((num_envs, 4), device=self.device)
        self.quat_d_prev[:, 0] = 1.0  # Initialize to identity
        self.lin_vel_d_prev = torch.zeros((num_envs, 3), device=self.device)
        self.ang_vel_d_prev = torch.zeros((num_envs, 3), device=self.device)

        # Rotation matrix tracking (for attitude control)
        self.R_d_prev = torch.eye(3, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)
        self.R_d = torch.eye(3, device=self.device).unsqueeze(0).repeat(num_envs, 1, 1)

        # Control state tracking
        self.is_first_step = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._last_force_body = torch.zeros((num_envs, 3), device=self.device)
        self._target_quat_for_heading = torch.zeros((num_envs, 4), device=self.device)
        self._target_quat_for_heading[:, 0] = 1.0  # Initialize to identity

    def reset(self, default_root_state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state for specified environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.integral_state["pos"].zero_()
            self.integral_state["att"].zero_()
        else:
            self.integral_state["pos"][env_ids] = 0.0
            self.integral_state["att"][env_ids] = 0.0

        # Initialize desired state to current root state
        self.x_d[env_ids, 0:3] = default_root_state[:, 0:3]  # position
        self.x_d[env_ids, 3:7] = default_root_state[:, 3:7]  # quaternion
        self.x_d[env_ids, 7:] = 0.0  # velocities and accelerations

        # Reset previous desired state
        self.pos_d_prev[env_ids] = default_root_state[:, 0:3]
        self.quat_d_prev[env_ids] = default_root_state[:, 3:7]
        self.lin_vel_d_prev[env_ids] = 0.0
        self.ang_vel_d_prev[env_ids] = 0.0

        # Reset rotation matrices
        quat_reset = default_root_state[:, 3:7]
        R_d_reset = math_utils.matrix_from_quat(quat_reset)
        self.R_d_prev[env_ids] = R_d_reset
        self.R_d[env_ids] = R_d_reset

        # Reset first step flag
        if env_ids is None:
            self.is_first_step.fill_(True)
            self._outer_loop_counter = 0
        else:
            self.is_first_step[env_ids] = True
            if len(env_ids) == self.num_envs:
                self._outer_loop_counter = 0

    def set_gains(
        self,
        kp_pos_xy: float | None = None,
        kp_pos_z: float | None = None,
        kd_pos_xy: float | None = None,
        kd_pos_z: float | None = None,
        ki_pos_xy: float | None = None,
        ki_pos_z: float | None = None,
        kp_rot: float | None = None,
        kd_rot: float | None = None,
        ki_rot: float | None = None,
    ) -> None:
        """Update controller gains."""
        if kp_pos_xy is not None:
            self.gains.kp_pos_xy = kp_pos_xy
        if kp_pos_z is not None:
            self.gains.kp_pos_z = kp_pos_z
        if kd_pos_xy is not None:
            self.gains.kd_pos_xy = kd_pos_xy
        if kd_pos_z is not None:
            self.gains.kd_pos_z = kd_pos_z
        if ki_pos_xy is not None:
            self.gains.ki_pos_xy = ki_pos_xy
        if ki_pos_z is not None:
            self.gains.ki_pos_z = ki_pos_z
        if kp_rot is not None:
            self.gains.kp_rot = kp_rot
        if kd_rot is not None:
            self.gains.kd_rot = kd_rot
        if ki_rot is not None:
            self.gains.ki_rot = ki_rot

    def compute_desired_wrench(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute desired wrench from observations.

        This method is required by BaseController. It computes the desired 6DOF wrench
        (force + torque) based on the current state using nested loop control.

        Outer loop: Position control computes desired rotation matrix R_d
        - Updates every outer_loop_decimation steps (slower)
        - Uses outer_loop_dt for integral updates
        - Computes force in body frame (along body z-axis)

        Inner loop: Attitude control tracks R_d
        - Runs every step (faster)
        - Uses dt for derivative calculations

        Args:
            obs: Observation tensor [pos(3), quat(4), lin_vel(3), ang_vel(3)]

        Returns:
            wrench: Desired wrench tensor [num_envs, 6] in body frame
                   Format: [fx, fy, fz, tx, ty, tz]
        """
        # Check if outer loop should update
        self._outer_loop_counter += 1
        should_update_outer = (self._outer_loop_counter % self.outer_loop_decimation == 0) or self.is_first_step.any()

        # Outer loop: Compute position control and desired rotation matrix
        if should_update_outer:
            force_body, R_d = self.compute_position_control(obs)
            self._last_force_body = force_body.clone()
            self.R_d = R_d.clone()

            # Update x_d quaternion from R_d (ensures consistency with position control)
            quat_d_from_R_d = math_utils.quat_from_matrix(R_d)
            self.x_d[:, 3:7] = quat_d_from_R_d
            self.quat_d_prev = quat_d_from_R_d.clone()

        # Inner loop: Compute attitude control (runs every step)
        torque = self.compute_attitude_control(obs)

        # Mark that first step is complete
        self.is_first_step.fill_(False)

        # Stack into wrench: [fx, fy, fz, tx, ty, tz]
        wrench = torch.cat([self._last_force_body, torque], dim=-1)  # [num_envs, 6]
        return wrench

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

        # Store desired heading quaternion for use in compute_position_control
        # For 4DOF, we use the target_quat directly as the heading
        self._target_quat_for_heading = target_quat

        # Update desired position
        self.x_d[:, 0:3] = target_pos

        # Compute linear velocity and acceleration from position
        lin_vel_d = torch.zeros_like(target_pos)
        acc_d = torch.zeros_like(target_pos)

        not_first = ~is_first_step
        if not_first.any():
            lin_vel_d[not_first] = (target_pos[not_first] - self.pos_d_prev[not_first]) / self.dt
            acc_d[not_first] = (lin_vel_d[not_first] - self.lin_vel_d_prev[not_first]) / self.dt

        # Update desired velocities and accelerations
        self.x_d[:, 7:10] = lin_vel_d.clamp(-1.0, 1.0)
        self.x_d[:, 13:16] = acc_d.clamp(-1.0, 1.0)

        # Update previous state for next iteration
        self.pos_d_prev = target_pos.clone()
        self.lin_vel_d_prev = lin_vel_d.clone()

    def compute_position_control(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position control (outer loop) and desired rotation matrix.

        Computes the desired force magnitude and rotation matrix R_d based on position error.
        The force is returned in body frame (along body z-axis).

        Args:
            obs: Observation tensor [pos(3), quat(4), lin_vel(3), ang_vel(3)]

        Returns:
            force_body: Force vector in body frame [num_envs, 3] (only z-component is non-zero)
            R_d: Desired rotation matrix [num_envs, 3, 3]
        """
        if obs.shape[0] != self.num_envs:
            raise ValueError(f"obs batch size {obs.shape[0]} does not match num_envs {self.num_envs}")

        # Unpack current state
        pos = obs[:, 0:3]
        quat = obs[:, 3:7]
        lin_vel = obs[:, 7:10]
        R = math_utils.matrix_from_quat(quat)

        # Unpack desired state
        pos_d = self.x_d[:, 0:3]
        quat_d = self._target_quat_for_heading  # Desired heading from actions
        lin_vel_d = self.x_d[:, 7:10]
        acc_d = self.x_d[:, 13:16]

        # Compute position control errors
        pos_err = pos_d - pos
        lin_vel_err = lin_vel_d - lin_vel

        # Create direction-specific gain vectors [xy, xy, z]
        kp_pos_vec = torch.tensor(
            [self.gains.kp_pos_xy, self.gains.kp_pos_xy, self.gains.kp_pos_z],
            device=self.device,
        ).unsqueeze(0)
        kd_pos_vec = torch.tensor(
            [self.gains.kd_pos_xy, self.gains.kd_pos_xy, self.gains.kd_pos_z],
            device=self.device,
        ).unsqueeze(0)
        ki_pos_vec = torch.tensor(
            [self.gains.ki_pos_xy, self.gains.ki_pos_xy, self.gains.ki_pos_z],
            device=self.device,
        ).unsqueeze(0)

        # Update integral term with anti-windup
        self.integral_state["pos"] += pos_err * self.outer_loop_dt
        if self.control_limits.integral_windup_limit is not None:
            integral_norm = torch.norm(self.integral_state["pos"], dim=1, keepdim=True)
            exceeds = integral_norm > self.control_limits.integral_windup_limit
            self.integral_state["pos"] *= torch.where(
                exceeds,
                self.control_limits.integral_windup_limit / (integral_norm + 1e-8),
                torch.ones_like(integral_norm),
            )

        # Compute A vector: A = mass * (kp*pos_err + kd*lin_vel_err + ki*integral + gravity + acc_d)
        # Apply different gains for xy and z directions
        A = self.mass * (
            kp_pos_vec * pos_err
            + kd_pos_vec * lin_vel_err
            + ki_pos_vec * self.integral_state["pos"]
            + self.gravity.expand(self.num_envs, 3)
            + acc_d
        )

        # Compute thrust magnitude: f = dot(A, R*e3)
        body_z_axis = R[:, :, 2]  # Body z-axis in world frame
        thrust = torch.sum(A * body_z_axis, dim=1, keepdim=True)

        # Compute b3c: b3c = A/norm(A) - normalized feedback function
        A_norm = torch.norm(A, dim=1, keepdim=True)
        default_b3c = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        b_3d = torch.where(A_norm > 1e-6, A / (A_norm + 1e-8), default_b3c)

        # Compute desired body frame basis vectors
        R_d_from_quat = math_utils.matrix_from_quat(quat_d)
        b_1d = R_d_from_quat[:, :, 0]  # Desired x-axis from heading

        C = torch.cross(b_3d, b_1d, dim=1)
        C_norm = torch.norm(C, dim=1, keepdim=True)
        b_2d_from_quat = R_d_from_quat[:, :, 1]

        # Handle near-parallel case
        b_2d = torch.where(C_norm > 1e-6, C / (C_norm + 1e-8), b_2d_from_quat)
        b_1d = torch.where(
            C_norm > 1e-6,
            -torch.cross(b_3d, C, dim=1) / (C_norm + 1e-8),
            R_d_from_quat[:, :, 0],
        )

        # Construct R_d from orthonormal basis [b_1d, b_2d, b_3d]
        R_d = torch.stack([b_1d, b_2d, b_3d], dim=2)

        # Force vector in body frame: only along body z-axis (4DOF constraint)
        force_body = torch.zeros((self.num_envs, 3), device=self.device)
        force_body[:, 2] = thrust.squeeze(-1)

        # Apply force limits
        if self.control_limits.max_force is not None:
            force_body_norm = torch.abs(force_body[:, 2:3])
            exceeds = force_body_norm > self.control_limits.max_force
            force_body[:, 2:3] = torch.where(
                exceeds,
                torch.sign(force_body[:, 2:3]) * self.control_limits.max_force,
                force_body[:, 2:3],
            )

        return force_body, R_d

    def compute_attitude_control(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attitude control (inner loop) to track desired rotation matrix.

        Args:
            obs: Observation tensor [pos(3), quat(4), lin_vel(3), ang_vel(3)]

        Returns:
            torque: Torque vector [num_envs, 3]
        """
        if obs.shape[0] != self.num_envs:
            raise ValueError(f"obs batch size {obs.shape[0]} does not match num_envs {self.num_envs}")

        # Unpack current state
        quat = obs[:, 3:7]
        ang_vel = obs[:, 10:13]
        R = math_utils.matrix_from_quat(quat)

        # Compute R_d_dot for angular velocity computation
        R_d_dot = torch.zeros_like(self.R_d)
        not_first_mask = ~self.is_first_step
        if not_first_mask.any():
            R_d_dot[not_first_mask] = (self.R_d[not_first_mask] - self.R_d_prev[not_first_mask]) / self.dt

        # Compute angular velocity: ω = vee(R_d^T * R_d_dot)
        R_d_dot_body = torch.bmm(self.R_d.transpose(1, 2), R_d_dot)
        # Ensure R_d_dot_body is skew-symmetric by extracting the skew-symmetric part
        R_d_dot_body = 0.5 * (R_d_dot_body - R_d_dot_body.transpose(1, 2))

        ang_vel_d_computed = (
            torch.stack(
                [
                    R_d_dot_body[:, 2, 1] - R_d_dot_body[:, 1, 2],  # ω_x
                    R_d_dot_body[:, 0, 2] - R_d_dot_body[:, 2, 0],  # ω_y
                    R_d_dot_body[:, 1, 0] - R_d_dot_body[:, 0, 1],  # ω_z
                ],
                dim=1,
            )
            * 0.5
        )

        # Set ang_vel_d to 0.0 for the first step, otherwise use computed value
        ang_vel_d = torch.where(
            self.is_first_step.unsqueeze(-1).expand(-1, 3),
            torch.zeros_like(ang_vel_d_computed),
            ang_vel_d_computed,
        )
        ang_vel_d = ang_vel_d.clamp(-1.0, 1.0)

        # Compute angular acceleration: α = (ω_d - ω_d_prev) / dt
        ang_acc_d_computed = torch.zeros_like(ang_vel_d)
        if not_first_mask.any():
            ang_acc_d_computed[not_first_mask] = (
                ang_vel_d[not_first_mask] - self.ang_vel_d_prev[not_first_mask]
            ) / self.dt

        # Set ang_acc_d to 0.0 for the first step, otherwise use computed value
        ang_acc_d = torch.where(
            self.is_first_step.unsqueeze(-1).expand(-1, 3),
            torch.zeros_like(ang_acc_d_computed),
            ang_acc_d_computed,
        )
        ang_acc_d = ang_acc_d.clamp(-1.0, 1.0)

        # Compute rotation error
        R_d_T_R = torch.bmm(self.R_d.transpose(1, 2), R)
        R_T_R_d = torch.bmm(R.transpose(1, 2), self.R_d)
        R_err_diff = R_d_T_R - R_T_R_d

        # Rotation error vector (vee map)
        e_R = 0.5 * torch.stack(
            [
                R_err_diff[:, 2, 1] - R_err_diff[:, 1, 2],
                R_err_diff[:, 0, 2] - R_err_diff[:, 2, 0],
                R_err_diff[:, 1, 0] - R_err_diff[:, 0, 1],
            ],
            dim=1,
        )

        # Angular velocity error: eOmega = Omega - R^T*R_d*Omega_d
        R_T_R_d_ang_vel_d = torch.bmm(R.transpose(1, 2), torch.bmm(self.R_d, ang_vel_d.unsqueeze(-1))).squeeze(-1)
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

        # Compute torque
        J = self.inertia_matrix.expand(self.num_envs, 3, 3) if self.inertia_matrix.ndim == 2 else self.inertia_matrix

        # PID term: -kR*eR - kOmega*eOmega - ki*integral
        pid_term = (
            -self.gains.kp_rot * e_R - self.gains.kd_rot * ang_vel_err - self.gains.ki_rot * self.integral_state["att"]
        )
        torque = torch.bmm(J, pid_term.unsqueeze(-1)).squeeze(-1)

        # Coriolis term: cross(Omega, J*Omega)
        torque += torch.cross(ang_vel, torch.bmm(J, ang_vel.unsqueeze(-1)).squeeze(-1), dim=1)

        # Feedforward term
        R_T_R_d_ang_acc_d = torch.bmm(R.transpose(1, 2), torch.bmm(self.R_d, ang_acc_d.unsqueeze(-1))).squeeze(-1)
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

        # Update stored previous R_d and ang_vel_d for next iteration
        self.R_d_prev = self.R_d.clone()
        self.ang_vel_d_prev = ang_vel_d.clone()

        # update x_d for angular velocity and acceleration
        self.x_d[:, 10:13] = ang_vel_d.clone()
        self.x_d[:, 16:19] = ang_acc_d.clone()

        return torque

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
