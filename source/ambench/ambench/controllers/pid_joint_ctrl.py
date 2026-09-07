# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint Space PID Controller for manipulator control.

This module provides a PID controller for joint space control of manipulators.
The controller tracks joint position targets using PID control with velocity error
computed from numerical differentiation of position targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PIDJointGains:
    """PID gains for joint space control.

    Attributes:
        kp: Proportional gains for each joint, shape (num_joints,)
        ki: Integral gains for each joint, shape (num_joints,)
        kd: Derivative gains for each joint, shape (num_joints,)
    """

    kp: torch.Tensor | tuple[float, ...] | list[float]
    ki: torch.Tensor | tuple[float, ...] | list[float]
    kd: torch.Tensor | tuple[float, ...] | list[float]


@dataclass
class ControlLimits:
    """Control output limits.

    Attributes:
        max_torque: Maximum torque per joint in N⋅m (default: None, no limit)
        integral_windup_limit: Maximum integral term magnitude for anti-windup (default: None, no limit)
    """

    max_torque: float | None = None
    integral_windup_limit: float | None = None


class PIDJointController:
    """Joint Space PID Controller for manipulator control.

    This controller implements PID control for tracking joint position targets.
    It uses numerical differentiation to compute target velocity from position targets.

    The controller expects:
    - Current state: joint_pos (num_envs, num_joints), joint_vel (num_envs, num_joints)
    - Desired state: target_joint_pos (num_envs, num_joints)

    Example:
        >>> controller = PIDJointController(
        ...     num_envs=10,
        ...     num_joints=4,
        ...     device="cuda",
        ...     dt=0.008,
        ...     gains=PIDJointGains(kp=(5.0, 5.0, 5.0, 5.0), ki=(0.1, 0.1, 0.1, 0.1), kd=(2.0, 2.0, 2.0, 2.0))
        ... )
        >>> torques = controller.compute(current_pos, current_vel, target_pos)
    """

    def __init__(
        self,
        num_envs: int,
        num_joints: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        gains: PIDJointGains | None = None,
        control_limits: ControlLimits | None = None,
        root_physx_view=None,
        joint_ids: torch.Tensor | None = None,
        has_mobile_base: bool = False,
    ):
        """Initialize the joint space PID controller.

        Args:
            num_envs: Number of parallel environments
            num_joints: Number of joints to control
            device: Device to run computations on ("cuda" or "cpu")
            dt: Time step in seconds (required)
            gains: PID gains configuration. If None, uses default gains.
            control_limits: Control output limits. If None, no limits applied.
            root_physx_view: Optional root PhysX view. If provided and get_mass_matrices() is available,
                           enables dynamics-aware control using mass matrix and gravity vector.
            joint_ids: Optional joint IDs tensor. Shape: (num_joints,)
            has_mobile_base: If True, the robot has a mobile base (6DOF). Gravity compensation forces will
                           skip the first 6 DOFs (x, y, z, roll, pitch, yaw) when indexing. Default: False.
        """
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device(device) if isinstance(device, str) else device
        if not dt:
            raise ValueError("dt (time step) must be provided for the controller.")
        self.dt: float = dt

        # Initialize gains
        if gains is None:
            gains = PIDJointGains(
                kp=(5.0,) * num_joints,
                ki=(0.1,) * num_joints,
                kd=(2.0,) * num_joints,
            )
        self.gains = gains

        # Convert gains to tensors
        self.kp = (
            gains.kp.to(self.device)
            if isinstance(gains.kp, torch.Tensor)
            else torch.tensor(gains.kp, device=self.device, dtype=torch.float32)
        )
        self.ki = (
            gains.ki.to(self.device)
            if isinstance(gains.ki, torch.Tensor)
            else torch.tensor(gains.ki, device=self.device, dtype=torch.float32)
        )
        self.kd = (
            gains.kd.to(self.device)
            if isinstance(gains.kd, torch.Tensor)
            else torch.tensor(gains.kd, device=self.device, dtype=torch.float32)
        )

        # Ensure gains have correct shape
        if self.kp.shape[0] != num_joints:
            raise ValueError(f"kp shape {self.kp.shape[0]} does not match num_joints {num_joints}")
        if self.ki.shape[0] != num_joints:
            raise ValueError(f"ki shape {self.ki.shape[0]} does not match num_joints {num_joints}")
        if self.kd.shape[0] != num_joints:
            raise ValueError(f"kd shape {self.kd.shape[0]} does not match num_joints {num_joints}")

        # Initialize control limits
        self.control_limits = control_limits if control_limits is not None else ControlLimits()

        # Initialize integral error accumulator
        self.integral_error = torch.zeros((num_envs, num_joints), device=self.device)

        # Previous target positions for numerical differentiation
        self.target_pos_prev = torch.zeros((num_envs, num_joints), device=self.device)

        # Track first step for each environment (used to set target_vel to zero on first step)
        self.is_first_step = torch.ones(num_envs, dtype=torch.bool, device=self.device)

        # Dynamics-aware control setup
        # Enabled if root_physx_view is provided and get_mass_matrices() is available
        # Uses mass matrix and gravity vector for dynamics-aware PID control
        self.root_physx_view = root_physx_view
        self.joint_ids = joint_ids.to(self.device) if isinstance(joint_ids, torch.Tensor) else joint_ids
        self.has_mobile_base = has_mobile_base

        # Check if dynamics-aware control is available
        self.enable_dynamics_aware = root_physx_view is not None and joint_ids is not None

        if self.enable_dynamics_aware:
            # Initialize dynamics matrices (will be computed in compute() method)
            self.mass_matrix = torch.zeros((num_envs, num_joints, num_joints), device=self.device)
            self.gravity_vector = torch.zeros((num_envs, num_joints), device=self.device)
        else:
            self.mass_matrix = None
            self.gravity_vector = None

    def reset(self, env_ids: torch.Tensor | None = None, target_pos: torch.Tensor | None = None) -> None:
        """Reset controller state.

        Args:
            env_ids: Optional tensor of environment IDs to reset. If None, resets all environments.
            target_pos: Optional target positions to initialize target_pos_prev. If None, resets to zero.
                       Shape: (num_envs, num_joints) or (len(env_ids), num_joints) if env_ids is provided.
        """
        if env_ids is None:
            # Reset all environments
            self.integral_error.zero_()
            if target_pos is not None:
                self.target_pos_prev = target_pos.clone().to(self.device)
            else:
                self.target_pos_prev.zero_()
            self.is_first_step.fill_(True)
        else:
            # Reset specific environments
            self.integral_error[env_ids] = 0.0
            if target_pos is not None:
                self.target_pos_prev[env_ids] = target_pos.to(self.device)
            else:
                self.target_pos_prev[env_ids] = 0.0
            self.is_first_step[env_ids] = True

    def set_gains(
        self,
        kp: torch.Tensor | tuple[float, ...] | list[float] | None = None,
        ki: torch.Tensor | tuple[float, ...] | list[float] | None = None,
        kd: torch.Tensor | tuple[float, ...] | list[float] | None = None,
    ) -> None:
        """Update PID gains.

        Args:
            kp: Proportional gains for each joint
            ki: Integral gains for each joint
            kd: Derivative gains for each joint
        """
        if kp is not None:
            self.kp = (
                kp.to(self.device)
                if isinstance(kp, torch.Tensor)
                else torch.tensor(kp, device=self.device, dtype=torch.float32)
            )
        if ki is not None:
            self.ki = (
                ki.to(self.device)
                if isinstance(ki, torch.Tensor)
                else torch.tensor(ki, device=self.device, dtype=torch.float32)
            )
        if kd is not None:
            self.kd = (
                kd.to(self.device)
                if isinstance(kd, torch.Tensor)
                else torch.tensor(kd, device=self.device, dtype=torch.float32)
            )

    def compute(
        self,
        current_pos: torch.Tensor,
        current_vel: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control torques.

        Args:
            current_pos: Current joint positions, shape (num_envs, num_joints)
            current_vel: Current joint velocities, shape (num_envs, num_joints)
            target_pos: Target joint positions, shape (num_envs, num_joints)

        Returns:
            torques: Control torques, shape (num_envs, num_joints)
        """
        # Validate input shapes
        if current_pos.shape != (self.num_envs, self.num_joints):
            raise ValueError(
                f"current_pos shape {current_pos.shape} does not match expected ({self.num_envs}, {self.num_joints})"
            )
        if current_vel.shape != (self.num_envs, self.num_joints):
            raise ValueError(
                f"current_vel shape {current_vel.shape} does not match expected ({self.num_envs}, {self.num_joints})"
            )
        if target_pos.shape != (self.num_envs, self.num_joints):
            raise ValueError(
                f"target_pos shape {target_pos.shape} does not match expected ({self.num_envs}, {self.num_joints})"
            )

        # Ensure tensors are on correct device
        current_pos = current_pos.to(self.device)
        current_vel = current_vel.to(self.device)
        target_pos = target_pos.to(self.device)

        # Compute position error
        pos_error = target_pos - current_pos  # [num_envs, num_joints]

        # Compute target velocity by numerical differentiation
        # target_vel = (target_pos - target_pos_prev) / dt
        # For first step, use zero target velocity
        target_vel = torch.zeros_like(current_vel)  # [num_envs, num_joints]
        not_first = ~self.is_first_step
        if not_first.any():
            target_vel[not_first] = (target_pos[not_first] - self.target_pos_prev[not_first]) / self.dt

        # Mark that first step is complete for environments that were in first step
        # Only set to False for environments that were True (i.e., just completed their first step)
        self.is_first_step.fill_(False)

        # Velocity error: current_vel - target_vel
        vel_error = current_vel - target_vel  # [num_envs, num_joints]

        # Update integral error with anti-windup
        self.integral_error += pos_error * self.dt  # [num_envs, num_joints]
        if self.control_limits.integral_windup_limit is not None:
            integral_norm = torch.norm(self.integral_error, dim=1, keepdim=True)  # [num_envs, 1]
            exceeds = integral_norm > self.control_limits.integral_windup_limit
            self.integral_error *= torch.where(
                exceeds,
                self.control_limits.integral_windup_limit / (integral_norm + 1e-8),
                torch.ones_like(integral_norm),
            )

        # Compute control torques
        if self.enable_dynamics_aware:
            # Dynamics-aware control: τ = M(q) * q̈_d + G(q)
            # where M is mass matrix, G is gravity vector

            # Get gravity vector from Isaac Lab
            gravity_torques_all = self.root_physx_view.get_gravity_compensation_forces()
            # If mobile base exists, first 6 DOFs are base (x, y, z, roll, pitch, yaw), so offset joint_ids by 6
            if self.has_mobile_base:
                gravity_joint_ids = self.joint_ids + 6
            else:
                gravity_joint_ids = self.joint_ids
            self.gravity_vector = gravity_torques_all[:, gravity_joint_ids].to(self.device)  # [num_envs, num_joints]

            # Get mass matrix from Isaac Lab
            # Use get_generalized_mass_matrices() for full articulation mass matrix
            mass_matrices_all = self.root_physx_view.get_generalized_mass_matrices().to(self.device)

            # Extract arm joint mass matrix
            if self.has_mobile_base:
                arm_start_idx = 6  # Base has 6 DOFs
                arm_end_idx = arm_start_idx + self.num_joints  # arm_start_idx + 4 = 10

                # Extract only arm joint part (skip base and gripper)
                mass_matrix_arm = mass_matrices_all[:, arm_start_idx:arm_end_idx, arm_start_idx:arm_end_idx]

                # joint_ids are already relative to arm joints (0, 1, 2, 3)
                # So we can use them directly
                mass_matrix_selected = mass_matrix_arm[:, self.joint_ids, :][:, :, self.joint_ids]
            else:
                # Fixed base: mass matrix is directly for joints
                mass_matrix_selected = mass_matrices_all[:, self.joint_ids, :][:, :, self.joint_ids]
            self.mass_matrix = mass_matrix_selected.to(self.device)  # [num_envs, num_joints, num_joints]

            # Compute desired acceleration from PID control
            # q̈_d = Kp * pos_error + Ki * integral_error - Kd * vel_error
            desired_acc = (
                self.kp.unsqueeze(0) * pos_error
                + self.ki.unsqueeze(0) * self.integral_error
                - self.kd.unsqueeze(0) * vel_error
            )  # [num_envs, num_joints]

            # M(q) * q̈_d
            mass_acc = torch.bmm(self.mass_matrix, desired_acc.unsqueeze(-1)).squeeze(-1)  # [num_envs, num_joints]

            torques = mass_acc + self.gravity_vector

        else:
            # Standard PID control only: torque = Kp * pos_error + Ki * integral_error - Kd * vel_error
            # No gravity compensation or mass matrix multiplication
            torques = (
                self.kp.unsqueeze(0) * pos_error
                + self.ki.unsqueeze(0) * self.integral_error
                - self.kd.unsqueeze(0) * vel_error
            )  # [num_envs, num_joints]

        # Apply torque limits
        if self.control_limits.max_torque is not None:
            torque_norm = torch.norm(torques, dim=1, keepdim=True)  # [num_envs, 1]
            exceeds = torque_norm > self.control_limits.max_torque
            torques *= torch.where(
                exceeds, self.control_limits.max_torque / (torque_norm + 1e-8), torch.ones_like(torque_norm)
            )

        # Store current targets for next timestep's numerical differentiation
        self.target_pos_prev = target_pos.clone()

        return torques

    def get_integral_state(self) -> torch.Tensor:
        """Get current integral error state.

        Returns:
            Integral error tensor, shape (num_envs, num_joints)
        """
        return self.integral_error.clone()

    def set_integral_state(self, integral_error: torch.Tensor) -> None:
        """Set integral error state (useful for warm-starting or loading saved state).

        Args:
            integral_error: Integral error tensor, shape (num_envs, num_joints)
        """
        self.integral_error = integral_error.to(self.device)
