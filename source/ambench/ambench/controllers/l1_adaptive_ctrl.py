# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""L1 Adaptive Controller for SE(3) tracking.

This module provides an L1 adaptive controller implementation for 6DOF control.
L1 adaptive control provides fast adaptation and guaranteed transient performance
with bounded control signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation

from ambench.controllers.controller_cfg import BaseController
from ambench.controllers.utils.control_utils import get_aggregate_physical_parameters


@dataclass
class L1AdaptiveGains:
    """L1 adaptive control gains.

    Attributes:
        k_pos: Position gain (default: [8.0, 8.0, 8.0] from launch file)
        k_vel: Linear velocity gain (default: [5.0, 5.0, 5.0] from launch file)
        k_ori: Orientation gain (default: [15.0, 20.0, 10.0] from launch file)
        k_ang_vel: Angular velocity gain (default: [10.0, 9.0, 5.0] from launch file)
        A_lin: Linear adaptation A matrix diagonal (default: [-1.0, -1.0, -1.0])
        A_ang: Angular adaptation A matrix diagonal (default: [-0.65, -0.65, -0.65])
        filter_bandwidth_lin: L1 filter bandwidth for linear adaptation (default: 0.75 Hz)
        filter_bandwidth_ang: L1 filter bandwidth for angular adaptation (default: 0.75 Hz)
        sigma_max_lin: Optional norm bound for linear adaptive estimate.
        sigma_max_ang: Optional norm bound for angular adaptive estimate.
        adaptive_mix_lin: Linear adaptive term mixing ratio [0.0-1.0] (default: 1.0)
        adaptive_mix_ang: Angular adaptive term mixing ratio [0.0-1.0] (default: 1.0)
    """

    k_pos: float | list[float] | torch.Tensor = None
    k_vel: float | list[float] | torch.Tensor = None
    k_ori: float | list[float] | torch.Tensor = None
    k_ang_vel: float | list[float] | torch.Tensor = None
    A_lin: list[float] | torch.Tensor = None
    A_ang: list[float] | torch.Tensor = None
    filter_bandwidth_lin: float = 0.75
    filter_bandwidth_ang: float = 0.75
    sigma_max_lin: float | None = None
    sigma_max_ang: float | None = None
    adaptive_mix_lin: float = 1.0
    adaptive_mix_ang: float = 1.0

    def __post_init__(self):
        # Default values from launch file: omnihexa_l1ctrller_planner.launch
        if self.k_pos is None:
            self.k_pos = [8.0, 8.0, 8.0]  # positoin_P_gain
        if self.k_vel is None:
            self.k_vel = [5.0, 5.0, 5.0]  # positoin_D_gain
        if self.k_ori is None:
            self.k_ori = [10.0, 10.0, 10.0]  # orientation_P_gain
        if self.k_ang_vel is None:
            self.k_ang_vel = [10.0, 20.0, 5.0]  # orienation_D_gain
        if self.A_lin is None:
            self.A_lin = [-1.0, -1.0, -1.0]  # L1_Adaptive_position_A_Matrix
        if self.A_ang is None:
            self.A_ang = [-0.65, -0.65, -0.65]  # L1_Adaptive_orienation_A_Matrix


class L1AdaptiveController(BaseController):
    """L1 Adaptive Controller for 6DOF SE(3) tracking.

    This controller implements L1 adaptive control for tracking position and orientation
    in SE(3). The controller consists of four key components:
    1. State Predictor: Predicts state using reference model + adaptive estimate
    2. Adaptation Law: Updates adaptive estimate based on prediction error
    3. L1 Low-Pass Filter: Filters adaptive signal before injecting into control
    4. Control Law: Baseline control + filtered adaptive compensation

    Example:
        >>> controller = L1AdaptiveController(
        ...     num_envs=10,
        ...     device="cuda",
        ...     dt=0.01,
        ...     gains=L1AdaptiveGains(...),
        ...     robot=robot,
        ...     scene=scene
        ... )
        >>> wrench = controller.compute_desired_wrench(obs)
    """

    REQUIRES_ROBOT = True
    REQUIRES_SCENE = True

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        gains: Any | None = None,
        l1_params: dict[str, Any] | None = None,
        robot: Articulation | None = None,
        scene=None,
        robot_spec=None,
        **kwargs,
    ):
        """Initialize the L1 adaptive controller.

        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on ("cuda" or "cpu")
            dt: Time step in seconds. Required for the controller.
            gains: Gains. Accepts either `L1AdaptiveGains` or `PID6DOFGains` (from `pid_6dof_ctrl.py`).
            l1_params: Optional dict to override L1 parameters. Keys:
                - "A_lin", "A_ang": Adaptation A matrix diagonals
                - "low_pass_filter_bandwidth": Set both filter bandwidths (convenience)
                - "filter_bandwidth_lin", "filter_bandwidth_ang": Individual L1 filter bandwidths (Hz)
                - "sigma_max_lin", "sigma_max_ang": Optional norm bounds for adaptive estimates
                - "adaptive_mix": Set both adaptive_mix_lin and adaptive_mix_ang (convenience)
                - "adaptive_mix_lin", "adaptive_mix_ang": Individual adaptive term mixing ratios [0.0-1.0]
            robot: Articulation object. Required to compute physical parameters from simulator.
            scene: Scene object used to read gravity from the simulation.
            robot_spec: Robot morphology and multirotor specification.
            **kwargs: Additional arguments passed to BaseController
        """
        if robot is None:
            raise ValueError("robot parameter is required. Physical parameters are computed from the simulator.")
        if not dt:
            raise ValueError("dt (time step) must be provided for the controller.")

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
        self.gains = self._initialize_gains(gains)
        self._apply_l1_params(l1_params)

        # Get physical parameters from simulator
        total_mass, total_inertia_matrix, gravity = get_aggregate_physical_parameters(robot, scene=scene)
        self.mass = total_mass
        self.inertia_matrix = total_inertia_matrix.to(self.device)
        self.inertia = torch.diag(self.inertia_matrix)
        self.gravity = gravity.to(self.device)
        self.inertia_matrix_inv = torch.inverse(self.inertia_matrix)

        # Initialize controller state variables
        self._initialize_state_variables()

        # Convert gains to tensors
        self._initialize_gain_matrices()

        # Initialize L1 adaptive matrices
        self._initialize_l1_matrices()

    def _initialize_gains(self, gains: Any | None) -> L1AdaptiveGains:
        """Initialize gains from input (supports L1AdaptiveGains or PID6DOFGains)."""
        if gains is None:
            return L1AdaptiveGains()
        elif isinstance(gains, L1AdaptiveGains):
            return gains
        else:
            # Convert PID6DOFGains to L1AdaptiveGains
            # Default values from launch file: omnihexa_l1ctrller_planner.launch
            return L1AdaptiveGains(
                k_pos=getattr(gains, "kp_pos", [8.0, 8.0, 8.0]),
                k_vel=getattr(gains, "kd_pos", [5.0, 5.0, 5.0]),
                k_ori=getattr(gains, "kp_rot", [15.0, 20.0, 10.0]),
                k_ang_vel=getattr(gains, "kd_rot", [10.0, 9.0, 5.0]),
                A_lin=getattr(gains, "A_lin", [-1.0, -1.0, -1.0]),
                A_ang=getattr(gains, "A_ang", [-0.65, -0.65, -0.65]),
                filter_bandwidth_lin=getattr(gains, "filter_bandwidth_lin", 0.75),
                filter_bandwidth_ang=getattr(gains, "filter_bandwidth_ang", 0.75),
            )

    def _apply_l1_params(self, l1_params: dict[str, Any] | None) -> None:
        """Apply L1 parameter overrides from l1_params dict."""
        if not l1_params:
            return

        # Adaptation A matrices
        if "A_lin" in l1_params:
            self.gains.A_lin = l1_params["A_lin"]
        if "A_ang" in l1_params:
            self.gains.A_ang = l1_params["A_ang"]

        # Filter bandwidths (support both individual and combined keys)
        if "low_pass_filter_bandwidth" in l1_params:
            bw = float(l1_params["low_pass_filter_bandwidth"])
            self.gains.filter_bandwidth_lin = bw
            self.gains.filter_bandwidth_ang = bw
        if "filter_bandwidth_lin" in l1_params:
            self.gains.filter_bandwidth_lin = float(l1_params["filter_bandwidth_lin"])
        if "filter_bandwidth_ang" in l1_params:
            self.gains.filter_bandwidth_ang = float(l1_params["filter_bandwidth_ang"])

        # Adaptive estimate norm bounds
        if "sigma_max_lin" in l1_params:
            self.gains.sigma_max_lin = self._optional_positive_float(l1_params["sigma_max_lin"])
        if "sigma_max_ang" in l1_params:
            self.gains.sigma_max_ang = self._optional_positive_float(l1_params["sigma_max_ang"])

        # Adaptive mixing ratios (support both individual and combined keys)
        if "adaptive_mix" in l1_params:
            mix = float(l1_params["adaptive_mix"])
            self.gains.adaptive_mix_lin = mix
            self.gains.adaptive_mix_ang = mix
        if "adaptive_mix_lin" in l1_params:
            self.gains.adaptive_mix_lin = float(l1_params["adaptive_mix_lin"])
        if "adaptive_mix_ang" in l1_params:
            self.gains.adaptive_mix_ang = float(l1_params["adaptive_mix_ang"])

    def _optional_positive_float(self, value: Any) -> float | None:
        """Return a positive float bound, or None for disabled/nonpositive values."""
        if value is None:
            return None
        value_f = float(value)
        if value_f <= 0.0:
            return None
        return value_f

    def _initialize_state_variables(self) -> None:
        """Initialize all controller state variables."""
        # L1 adaptive states (following C++ implementation)
        self.v_hat = torch.zeros((self.num_envs, 3), device=self.device)
        self.w_hat = torch.zeros((self.num_envs, 3), device=self.device)
        self.v_tilda = torch.zeros((self.num_envs, 3), device=self.device)
        self.w_tilda = torch.zeros((self.num_envs, 3), device=self.device)
        self.h_v = torch.zeros((self.num_envs, 3), device=self.device)
        self.h_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.sigma_v_raw = torch.zeros((self.num_envs, 3), device=self.device)
        self.sigma_w_raw = torch.zeros((self.num_envs, 3), device=self.device)
        self.sigma_v = torch.zeros((self.num_envs, 3), device=self.device)
        self.sigma_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_F_l1 = torch.zeros((self.num_envs, 3), device=self.device)
        self.pre_tau_l1 = torch.zeros((self.num_envs, 3), device=self.device)
        self.F_l1 = torch.zeros((self.num_envs, 3), device=self.device)
        self.tau_l1 = torch.zeros((self.num_envs, 3), device=self.device)

        # Desired state [pos(3), quat(4), lin_vel(3), ang_vel(3), acc(3), ang_acc(3)]
        self.x_d = torch.zeros((self.num_envs, 19), device=self.device)
        self.x_d[:, 3] = 1.0  # Initialize quaternion to [1, 0, 0, 0]

        # Previous desired state for numerical differentiation
        self.x_d_prev = torch.zeros((self.num_envs, 13), device=self.device)
        self.x_d_prev[:, 3] = 1.0

        # Store latest computed control outputs (for backward compatibility)
        self.force = torch.zeros((self.num_envs, 3), device=self.device)
        self.torque = torch.zeros((self.num_envs, 3), device=self.device)
        self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.quat[:, 0] = 1.0  # Initialize to identity quaternion [1, 0, 0, 0]

        # e3 vector [0, 0, 1]
        self.e3 = torch.zeros((self.num_envs, 3), device=self.device)
        self.e3[:, 2] = 1.0

    def _initialize_gain_matrices(self) -> None:
        """Convert scalar/vector gains to diagonal matrices."""

        def to_diag_tensor(gain, num_envs, device):
            if isinstance(gain, (int, float)):
                # Scalar: create diagonal matrix with same value
                return torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * gain
            elif isinstance(gain, list):
                # List: convert to tensor and create diagonal matrix
                gain_tensor = torch.tensor(gain, device=device)
                return torch.diag(gain_tensor).unsqueeze(0).repeat(num_envs, 1, 1)
            elif isinstance(gain, torch.Tensor):
                if gain.ndim == 0:
                    # Scalar tensor
                    return torch.eye(3, device=device).unsqueeze(0).repeat(num_envs, 1, 1) * gain
                elif gain.ndim == 1:
                    # Vector tensor: create diagonal matrix
                    return torch.diag(gain).unsqueeze(0).repeat(num_envs, 1, 1)
                else:
                    # Matrix tensor: expand to batch
                    return gain.unsqueeze(0).repeat(num_envs, 1, 1)
            return gain

        self.K_pos = to_diag_tensor(self.gains.k_pos, self.num_envs, self.device)
        self.K_vel = to_diag_tensor(self.gains.k_vel, self.num_envs, self.device)
        self.K_ori = to_diag_tensor(self.gains.k_ori, self.num_envs, self.device)
        self.K_ang_vel = to_diag_tensor(self.gains.k_ang_vel, self.num_envs, self.device)

    def _initialize_l1_matrices(self) -> None:
        """Initialize L1 adaptive matrices (A_v, A_w, exp_Adt_v, exp_Adt_w, exp_w_dt)."""
        # Convert A_lin and A_ang to tensors
        if isinstance(self.gains.A_lin, list):
            A_lin_vec = torch.tensor(self.gains.A_lin, device=self.device)
        else:
            A_lin_vec = (
                self.gains.A_lin.to(self.device)
                if isinstance(self.gains.A_lin, torch.Tensor)
                else torch.tensor([self.gains.A_lin] * 3, device=self.device)
            )

        if isinstance(self.gains.A_ang, list):
            A_ang_vec = torch.tensor(self.gains.A_ang, device=self.device)
        else:
            A_ang_vec = (
                self.gains.A_ang.to(self.device)
                if isinstance(self.gains.A_ang, torch.Tensor)
                else torch.tensor([self.gains.A_ang] * 3, device=self.device)
            )

        # Create diagonal matrices
        self.A_v = torch.diag(A_lin_vec).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.A_w = torch.diag(A_ang_vec).unsqueeze(0).repeat(self.num_envs, 1, 1)

        # Compute exp(Adt) for each environment
        self.exp_Adt_v = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.exp_Adt_w = torch.zeros((self.num_envs, 3, 3), device=self.device)
        eye3 = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

        for i in range(3):
            self.exp_Adt_v[:, i, i] = torch.exp(self.dt * A_lin_vec[i])
            self.exp_Adt_w[:, i, i] = torch.exp(self.dt * A_ang_vec[i])

        # Compute exp(-cutoff_freq * 2 * pi * dt) for L1 filter
        self.exp_w_dt_lin = torch.exp(
            torch.tensor(-self.gains.filter_bandwidth_lin * 2.0 * math.pi * self.dt, device=self.device)
        )
        self.exp_w_dt_ang = torch.exp(
            torch.tensor(-self.gains.filter_bandwidth_ang * 2.0 * math.pi * self.dt, device=self.device)
        )

        # Pre-compute (exp_Adt - I)^(-1) for efficiency
        self.exp_Adt_v_minus_I = self.exp_Adt_v - eye3
        self.exp_Adt_w_minus_I = self.exp_Adt_w - eye3

    def reset(self, default_root_state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state.

        Args:
            default_root_state: Default root state tensor [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            env_ids: Optional tensor of environment IDs to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if default_root_state.shape[0] != env_ids.numel():
            raise ValueError(
                "default_root_state must contain one row per reset env id. "
                f"Got default_root_state.shape={tuple(default_root_state.shape)} "
                f"and env_ids.shape={tuple(env_ids.shape)}."
            )

        # Reset L1 adaptive states
        self.v_hat[env_ids] = default_root_state[:, 7:10]  # Initialize to current velocity
        self.w_hat[env_ids] = default_root_state[:, 10:13]  # Initialize to current angular velocity
        self.v_tilda[env_ids] = 0.0
        self.w_tilda[env_ids] = 0.0
        self.h_v[env_ids] = 0.0
        self.h_w[env_ids] = 0.0
        self.sigma_v_raw[env_ids] = 0.0
        self.sigma_w_raw[env_ids] = 0.0
        self.sigma_v[env_ids] = 0.0
        self.sigma_w[env_ids] = 0.0
        self.pre_F_l1[env_ids] = 0.0
        self.pre_tau_l1[env_ids] = 0.0
        self.F_l1[env_ids] = 0.0
        self.tau_l1[env_ids] = 0.0

        # Initialize desired state to current root state
        self.x_d[env_ids, 0:3] = default_root_state[:, 0:3]
        self.x_d[env_ids, 3:7] = default_root_state[:, 3:7]
        self.x_d[env_ids, 7:] = 0.0

        # Reset previous desired state
        self.x_d_prev[env_ids, 0:3] = default_root_state[:, 0:3]
        self.x_d_prev[env_ids, 3:7] = default_root_state[:, 3:7]
        self.x_d_prev[env_ids, 7:10] = 0.0
        self.x_d_prev[env_ids, 10:13] = 0.0

        # Reset stored control outputs
        self.force[env_ids] = 0.0
        self.torque[env_ids] = 0.0
        self.pos[env_ids] = default_root_state[:, 0:3]
        self.quat[env_ids] = default_root_state[:, 3:7]

    def compute_desired_wrench(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute desired wrench from observations.

        This method implements the L1 adaptive control law to compute the desired 6DOF wrench
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

        # 1. Baseline Control Law (compute first, needed for L1 prediction)
        force_pd_w, torque_pd, _, R = self._compute_baseline_control(
            pos, quat, lin_vel, ang_vel, pos_d, quat_d, lin_vel_d, ang_vel_d, acc_d, ang_acc_d
        )

        # 2. L1 Prediction (following C++ L1Prediction() logic)
        # Note: L1 prediction uses force in body frame, so convert force_pd_w to body frame
        R_bw = R.transpose(1, 2)  # Rotation from world to body
        force_pd_b = torch.bmm(R_bw, force_pd_w.unsqueeze(-1)).squeeze(-1)
        self._l1_prediction(obs, force_pd_b, torque_pd, R)

        # 3. Total Control (PD + L1 adaptive compensation)
        # Force: world frame PD + L1 adaptive (convert L1 to world frame first)
        force_w = force_pd_w + float(self.gains.adaptive_mix_lin) * torch.bmm(R, self.F_l1.unsqueeze(-1)).squeeze(-1)
        # Torque: body frame PD + L1 adaptive
        torque = torque_pd + float(self.gains.adaptive_mix_ang) * self.tau_l1

        # Convert force from world frame to body frame for wrench
        force_b = torch.bmm(R_bw, force_w.unsqueeze(-1)).squeeze(-1)

        # Store computed control outputs (for backward compatibility)
        self.force = force_b  # Store in body frame (matching PID6DOFController)
        self.torque = torque

        # Concatenate force and torque into wrench [fx, fy, fz, tx, ty, tz] in body frame
        desired_wrench_b = torch.cat([force_b, torque], dim=1)  # [num_envs, 6]

        return desired_wrench_b

    def _l1_prediction(
        self, obs: torch.Tensor, force_pd: torch.Tensor, torque_pd: torch.Tensor, R: torch.Tensor
    ) -> None:
        """L1 adaptive prediction step (following C++ L1Prediction() function).

        This implements the exact logic from the C++ code:
        1. Update state predictor (v_hat, w_hat)
        2. Compute prediction error (v_tilda, w_tilda)
        3. Compute adaptation signals (h_v, h_w)
        4. Compute sigma (sigma_v, sigma_w)
        5. Apply L1 filter (pre_F_l1, pre_tau_l1)
        6. Compute final adaptive compensation (F_l1, tau_l1)
        """
        # Extract current states
        lin_vel = obs[:, 7:10]
        ang_vel = obs[:, 10:13]

        # Compute reference model terms (f_v, f_w)
        # f_v = -gravity * e3
        f_v = -self.gravity.unsqueeze(0) * self.e3

        # f_w = -MoI^-1 * hat(w) * MoI * w (Coriolis term)
        J = (
            self.inertia_matrix.unsqueeze(0).repeat(self.num_envs, 1, 1)
            if self.inertia_matrix.ndim == 2
            else self.inertia_matrix
        )
        Jw = torch.bmm(J, ang_vel.unsqueeze(-1)).squeeze(-1)
        hat_w = self._hat_map(ang_vel)
        hat_w_Jw = torch.bmm(hat_w, Jw.unsqueeze(-1)).squeeze(-1)
        f_w = -torch.bmm(
            self.inertia_matrix_inv.unsqueeze(0).repeat(self.num_envs, 1, 1), hat_w_Jw.unsqueeze(-1)
        ).squeeze(-1)

        # Update state predictor
        # v_hat += dt * (f_v + R*(F_pd + F_l1)/mass + h_v + A_v*v_tilda)
        R_F_pd_F_l1 = torch.bmm(R, (force_pd + self.F_l1).unsqueeze(-1)).squeeze(-1)
        v_hat_dot = (
            f_v
            + R_F_pd_F_l1 / float(self.mass)
            + self.h_v
            + torch.bmm(self.A_v, self.v_tilda.unsqueeze(-1)).squeeze(-1)
        )
        self.v_hat += v_hat_dot * self.dt

        # w_hat += dt * (f_w + MoI^-1*(tau_pd + tau_l1) + h_w + A_w*w_tilda)
        tau_pd_tau_l1 = torque_pd + self.tau_l1
        w_hat_dot = (
            f_w
            + torch.bmm(
                self.inertia_matrix_inv.unsqueeze(0).repeat(self.num_envs, 1, 1), tau_pd_tau_l1.unsqueeze(-1)
            ).squeeze(-1)
            + self.h_w
            + torch.bmm(self.A_w, self.w_tilda.unsqueeze(-1)).squeeze(-1)
        )
        self.w_hat += w_hat_dot * self.dt

        # Compute prediction error
        self.v_tilda = self.v_hat - lin_vel
        self.w_tilda = self.w_hat - ang_vel

        # Compute adaptation signals
        # h_v = -(exp_Adt_v - I)^(-1) * A_v * exp_Adt_v * v_tilda
        # Note: (exp_Adt - I) is diagonal, so we can compute element-wise efficiently
        exp_Adt_v_v_tilda = torch.bmm(self.exp_Adt_v, self.v_tilda.unsqueeze(-1)).squeeze(-1)
        A_v_exp_Adt_v_v_tilda = torch.bmm(self.A_v, exp_Adt_v_v_tilda.unsqueeze(-1)).squeeze(-1)

        # Extract diagonal elements and compute inverse (element-wise for diagonal matrix)
        exp_Adt_v_minus_I_diag = torch.diagonal(self.exp_Adt_v_minus_I, dim1=1, dim2=2)  # (num_envs, 3)
        # Avoid division by zero
        valid_mask_v = torch.abs(exp_Adt_v_minus_I_diag) > 1e-10
        self.h_v = torch.where(
            valid_mask_v, -A_v_exp_Adt_v_v_tilda / exp_Adt_v_minus_I_diag, torch.zeros_like(A_v_exp_Adt_v_v_tilda)
        )

        # h_w = -(exp_Adt_w - I)^(-1) * A_w * exp_Adt_w * w_tilda
        exp_Adt_w_w_tilda = torch.bmm(self.exp_Adt_w, self.w_tilda.unsqueeze(-1)).squeeze(-1)
        A_w_exp_Adt_w_w_tilda = torch.bmm(self.A_w, exp_Adt_w_w_tilda.unsqueeze(-1)).squeeze(-1)

        exp_Adt_w_minus_I_diag = torch.diagonal(self.exp_Adt_w_minus_I, dim1=1, dim2=2)  # (num_envs, 3)
        valid_mask_w = torch.abs(exp_Adt_w_minus_I_diag) > 1e-10
        self.h_w = torch.where(
            valid_mask_w, -A_w_exp_Adt_w_w_tilda / exp_Adt_w_minus_I_diag, torch.zeros_like(A_w_exp_Adt_w_w_tilda)
        )

        # Compute sigma (adaptive estimate)
        # sigma_v = mass * R^T * h_v
        R_T = R.transpose(1, 2)
        self.sigma_v_raw = float(self.mass) * torch.bmm(R_T, self.h_v.unsqueeze(-1)).squeeze(-1)

        # sigma_w = MoI * h_w
        self.sigma_w_raw = torch.bmm(J, self.h_w.unsqueeze(-1)).squeeze(-1)
        self.sigma_v = self._project_vector_norm(self.sigma_v_raw, self.gains.sigma_max_lin)
        self.sigma_w = self._project_vector_norm(self.sigma_w_raw, self.gains.sigma_max_ang)

        # Apply L1 filter
        # pre_F_l1 = exp_w_dt * pre_F_l1 + (1 - exp_w_dt) * sigma_v
        self.pre_F_l1 = self.exp_w_dt_lin * self.pre_F_l1 + (1.0 - self.exp_w_dt_lin) * self.sigma_v

        # pre_tau_l1 = exp_w_dt * pre_tau_l1 + (1 - exp_w_dt) * sigma_w
        self.pre_tau_l1 = self.exp_w_dt_ang * self.pre_tau_l1 + (1.0 - self.exp_w_dt_ang) * self.sigma_w

        # Final adaptive compensation (negative of filtered signal)
        self.F_l1 = -self.pre_F_l1
        self.tau_l1 = -self.pre_tau_l1

    def _project_vector_norm(self, values: torch.Tensor, max_norm: float | None) -> torch.Tensor:
        """Project batched vectors onto an L2 ball when a positive bound is configured."""
        if max_norm is None:
            return values
        norms = torch.linalg.norm(values, dim=1, keepdim=True)
        scale = torch.clamp(float(max_norm) / norms.clamp_min(1e-8), max=1.0)
        return values * scale

    def _hat_map(self, v: torch.Tensor) -> torch.Tensor:
        """Hat map: R^3 -> so(3) (skew-symmetric matrix)."""
        hat = torch.zeros((v.shape[0], 3, 3), device=v.device)
        hat[:, 0, 1] = -v[:, 2]
        hat[:, 0, 2] = v[:, 1]
        hat[:, 1, 0] = v[:, 2]
        hat[:, 1, 2] = -v[:, 0]
        hat[:, 2, 0] = -v[:, 1]
        hat[:, 2, 1] = v[:, 0]
        return hat

    def _compute_rotation_error(
        self, quat: torch.Tensor, quat_d: torch.Tensor, ang_vel: torch.Tensor, ang_vel_d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotation error vector and angular velocity error.

        Returns:
            e_R: Rotation error vector (vee map of R_d^T*R - R^T*R_d)
            ang_vel_err: Angular velocity error
        """
        R = math_utils.matrix_from_quat(quat)
        R_d = math_utils.matrix_from_quat(quat_d)

        R_d_T_R = torch.bmm(R_d.transpose(1, 2), R)
        R_T_R_d = torch.bmm(R.transpose(1, 2), R_d)
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

        # Angular velocity error
        R_T_R_d_ang_vel_d = torch.bmm(R.transpose(1, 2), torch.bmm(R_d, ang_vel_d.unsqueeze(-1))).squeeze(-1)
        ang_vel_err = ang_vel - R_T_R_d_ang_vel_d

        return e_R, ang_vel_err

    def _compute_baseline_control(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        pos_d: torch.Tensor,
        quat_d: torch.Tensor,
        lin_vel_d: torch.Tensor,
        ang_vel_d: torch.Tensor,
        acc_d: torch.Tensor,
        ang_acc_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute baseline PD control with feedforward (following C++ CalculateControlInput()).

        Returns:
            force_pd_w: Baseline force in world frame
            torque_pd: Baseline torque in body frame
            e_R: Rotation error vector
            R: Rotation matrix (body to world)
        """
        # Position control (following C++ line 408, but compute in world frame like PID)
        # F_pd = mass * (acc_d + KP_p * e_p + KP_d * e_v + gravity)
        pos_error = pos_d - pos
        lin_vel_error = lin_vel_d - lin_vel
        R = math_utils.matrix_from_quat(quat)

        # Compute force in world frame (like PID controller)
        pd_term = (
            acc_d
            + torch.bmm(self.K_pos, pos_error.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(self.K_vel, lin_vel_error.unsqueeze(-1)).squeeze(-1)
            + self.gravity.unsqueeze(0)
        )
        force_pd_w = float(self.mass) * pd_term

        # Orientation control (following C++ line 409)
        # tau_pd = MoI * (-KR_p * e_R - KR_d * e_w) + hat(w) * MoI * w - MoI * hat(w) * R_d^T * R * w_d
        e_R, ang_vel_err = self._compute_rotation_error(quat, quat_d, ang_vel, ang_vel_d)
        J = (
            self.inertia_matrix.unsqueeze(0).repeat(self.num_envs, 1, 1)
            if self.inertia_matrix.ndim == 2
            else self.inertia_matrix
        )

        pd_term_rot = -torch.bmm(self.K_ori, e_R.unsqueeze(-1)).squeeze(-1) - torch.bmm(
            self.K_ang_vel, ang_vel_err.unsqueeze(-1)
        ).squeeze(-1)
        torque_pd = torch.bmm(J, pd_term_rot.unsqueeze(-1)).squeeze(-1)

        # Add Coriolis term: hat(w) * MoI * w
        Jw = torch.bmm(J, ang_vel.unsqueeze(-1)).squeeze(-1)
        hat_w = self._hat_map(ang_vel)
        coriolis = torch.bmm(hat_w, Jw.unsqueeze(-1)).squeeze(-1)
        torque_pd += coriolis

        # Add feedforward term: -MoI * hat(w) * R_d^T * R * w_d
        R_d = math_utils.matrix_from_quat(quat_d)
        R_d_T_R = torch.bmm(R_d.transpose(1, 2), R)
        R_d_T_R_w_d = torch.bmm(R_d_T_R, ang_vel_d.unsqueeze(-1)).squeeze(-1)
        hat_w_R_d_T_R_w_d = torch.bmm(hat_w, R_d_T_R_w_d.unsqueeze(-1)).squeeze(-1)
        feedforward = -torch.bmm(J, hat_w_R_d_T_R_w_d.unsqueeze(-1)).squeeze(-1)
        torque_pd += feedforward

        return force_pd_w, torque_pd, e_R, R

    def get_adaptation_state(self) -> dict[str, torch.Tensor]:
        """Get current adaptation state.

        Returns:
            Dictionary with adaptation state variables.
        """
        return {
            "v_hat": self.v_hat.clone(),
            "w_hat": self.w_hat.clone(),
            "v_tilda": self.v_tilda.clone(),
            "w_tilda": self.w_tilda.clone(),
            "h_v": self.h_v.clone(),
            "h_w": self.h_w.clone(),
            "sigma_v_raw": self.sigma_v_raw.clone(),
            "sigma_w_raw": self.sigma_w_raw.clone(),
            "sigma_v": self.sigma_v.clone(),
            "sigma_w": self.sigma_w.clone(),
            "pre_F_l1": self.pre_F_l1.clone(),
            "pre_tau_l1": self.pre_tau_l1.clone(),
            "F_l1": self.F_l1.clone(),
            "tau_l1": self.tau_l1.clone(),
        }

    def set_adaptation_state(self, adaptation_state: dict[str, torch.Tensor]) -> None:
        """Set adaptation state (useful for warm-starting or loading saved state).

        Args:
            adaptation_state: Dictionary with adaptation state variables.
        """
        if "v_hat" in adaptation_state:
            self.v_hat = adaptation_state["v_hat"].to(self.device)
        if "w_hat" in adaptation_state:
            self.w_hat = adaptation_state["w_hat"].to(self.device)
        if "v_tilda" in adaptation_state:
            self.v_tilda = adaptation_state["v_tilda"].to(self.device)
        if "w_tilda" in adaptation_state:
            self.w_tilda = adaptation_state["w_tilda"].to(self.device)
        if "h_v" in adaptation_state:
            self.h_v = adaptation_state["h_v"].to(self.device)
        if "h_w" in adaptation_state:
            self.h_w = adaptation_state["h_w"].to(self.device)
        if "sigma_v_raw" in adaptation_state:
            self.sigma_v_raw = adaptation_state["sigma_v_raw"].to(self.device)
        if "sigma_w_raw" in adaptation_state:
            self.sigma_w_raw = adaptation_state["sigma_w_raw"].to(self.device)
        if "sigma_v" in adaptation_state:
            self.sigma_v = adaptation_state["sigma_v"].to(self.device)
        if "sigma_w" in adaptation_state:
            self.sigma_w = adaptation_state["sigma_w"].to(self.device)
        if "pre_F_l1" in adaptation_state:
            self.pre_F_l1 = adaptation_state["pre_F_l1"].to(self.device)
        if "pre_tau_l1" in adaptation_state:
            self.pre_tau_l1 = adaptation_state["pre_tau_l1"].to(self.device)
        if "F_l1" in adaptation_state:
            self.F_l1 = adaptation_state["F_l1"].to(self.device)
        if "tau_l1" in adaptation_state:
            self.tau_l1 = adaptation_state["tau_l1"].to(self.device)

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

        # Update desired position and orientation
        self.x_d[:, 0:3] = target_pos
        self.x_d[:, 3:7] = target_quat

        # Compute velocities and accelerations numerically
        lin_vel_d = torch.zeros_like(target_pos)
        ang_vel_d = torch.zeros((self.num_envs, 3), device=self.device)
        acc_d = torch.zeros_like(target_pos)
        ang_acc_d = torch.zeros((self.num_envs, 3), device=self.device)

        not_first = ~is_first_step
        if not_first.any():
            # Linear velocity
            lin_vel_d[not_first] = (target_pos[not_first] - pos_d_prev[not_first]) / self.dt

            # Angular velocity from quaternion logarithm
            quat_prev_inv = math_utils.quat_conjugate(quat_prev[not_first])
            quat_rel = math_utils.quat_mul(target_quat[not_first], quat_prev_inv)
            quat_rel_w = quat_rel[:, 0:1]
            quat_rel_xyz = quat_rel[:, 1:4]
            quat_rel_xyz_norm = torch.norm(quat_rel_xyz, dim=1, keepdim=True)

            angle = 2.0 * torch.atan2(quat_rel_xyz_norm, quat_rel_w + 1e-8)
            log_quat = torch.where(
                quat_rel_xyz_norm > 1e-6,
                (angle / (quat_rel_xyz_norm + 1e-8)) * quat_rel_xyz,
                quat_rel_xyz,
            )
            ang_vel_d[not_first] = 2.0 * log_quat / self.dt

            # Accelerations
            acc_d[not_first] = (lin_vel_d[not_first] - lin_vel_d_prev[not_first]) / self.dt
            ang_acc_d[not_first] = (ang_vel_d[not_first] - ang_vel_d_prev[not_first]) / self.dt

        # Update desired state
        self.x_d[:, 7:10] = lin_vel_d.clamp(-1.0, 1.0)
        self.x_d[:, 10:13] = ang_vel_d.clamp(-1.0, 1.0)
        self.x_d[:, 13:16] = acc_d.clamp(-1.0, 1.0)
        self.x_d[:, 16:19] = ang_acc_d.clamp(-1.0, 1.0)

        # Update previous state
        self.x_d_prev[:, 0:3] = target_pos
        self.x_d_prev[:, 3:7] = target_quat
        self.x_d_prev[:, 7:10] = lin_vel_d
        self.x_d_prev[:, 10:13] = ang_vel_d
