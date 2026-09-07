# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Literal

import isaaclab.utils.math as math_utils
import torch

from ambench.tasks.base.base_env import BaseEnv

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Waypoint:
    """Structured waypoint description used by scripted policies."""

    t: int
    xyz: torch.Tensor
    quat: torch.Tensor
    gripper: float
    interpolation_type: Literal["linear", "circle"] = "linear"


class BasePolicy:
    """Base class for scripted policies compatible with IsaacLab environments.

    IsaacLab expects policies to have:
    - reset() method to reset internal state
    - advance() method to generate actions from observations
    """

    def __init__(
        self,
        env,
        inject_noise: bool = False,
        noise_pos_std_m: float = 0.01,
        noise_rot_std_rad: float = 0.05,
    ):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.waypoints: list[Waypoint] = []
        self.env: BaseEnv = env.unwrapped
        self.device = self.env.device if env is not None else "cpu"
        self.trajectory_generated = False

        self._initial_orientation_offset_quat: torch.Tensor | None = None
        self._refresh_initial_ee_state()

        # For delta action mode, we can compute increments purely from the planned trajectory.

        # Noise parameters (used when inject_noise=True).
        # Interpreted as Gaussian standard deviations.
        # - Position: meters
        # - Rotation: radians
        self.noise_pos_std_m: float = noise_pos_std_m
        self.noise_rot_std_rad: float = noise_rot_std_rad

    def _log_trajectory_summary(self, *extra: str) -> None:
        """Log a trajectory summary and the per-waypoint plan in a consistent format."""

        def _format_summary_value(value: object) -> object:
            """Convert tensors and simple containers into print-friendly values."""
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            if isinstance(value, tuple):
                return [_format_summary_value(item) for item in value]
            return value

        logger.debug("[%s] Generated %d waypoints.", self.__class__.__name__, len(self.waypoints))
        for line in extra:
            logger.debug("[%s] %s", self.__class__.__name__, _format_summary_value(line))
        for index, waypoint in enumerate(self.waypoints):
            logger.debug(
                "[%s] waypoint %d: t=%d, xyz=%s, quat=%s, gripper=%s, interp=%s",
                self.__class__.__name__,
                index,
                waypoint.t,
                _format_summary_value(waypoint.xyz),
                _format_summary_value(waypoint.quat),
                waypoint.gripper,
                waypoint.interpolation_type,
            )

    def identity_quat(self) -> torch.Tensor:
        """Return the identity quaternion on the policy device."""
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def _apply_initial_orientation_offset(self, quat: torch.Tensor) -> torch.Tensor:
        """Compose in the cached reset orientation when the robot/controller stack requires it.

        Args:
            quat: Quaternion from policy (4,) [w, x, y, z]

        Returns:
            Quaternion with the reset-orientation offset applied when needed, otherwise the original quaternion.
        """
        if self._initial_orientation_offset_quat is not None:
            return math_utils.quat_mul(quat.unsqueeze(0), self._initial_orientation_offset_quat.unsqueeze(0)).squeeze(0)
        return quat

    def _refresh_initial_ee_state(self) -> None:
        """Refresh the cached reset pose used by scripted trajectories."""
        self._ee_state_init_w = self.env.robot.data.body_link_state_w[:, self.env.ee_link_idx, :]
        self._ee_pos_init_w = self._ee_state_init_w[:, 0:3]
        self._ee_pos_init_local = self._ee_pos_init_w - self.env.scene.env_origins
        self._ee_quat_init_w = self._ee_state_init_w[:, 3:7]

        self._initial_orientation_offset_quat = None

    def reset(self):
        """Reset policy state. Called when environment resets."""
        self.step_count = 0
        self.waypoints = []
        self.trajectory_generated = False
        self._refresh_initial_ee_state()

    def generate_trajectory(self, obs, env_id=0):
        """Generate trajectory from initial observation.

        Waypoints should use env-origin-relative positions. The environment converts
        absolute action targets back to world coordinates before control.

        Must set self.waypoints to a list of `Waypoint` objects with fields:
        - `t`: timestep
        - `xyz`: position (3,)
        - `quat`: orientation (4,) [w, x, y, z]
        - `gripper`: gripper command float

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts.
                 Each environment dict has keys like: ee_pos, ee_quat, goal_pos, etc.
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(
        curr_waypoint: Waypoint, next_waypoint: Waypoint, t: int
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Interpolate between two waypoints.

        Returns:
            xyz: torch.Tensor of shape (3,)
            quat: torch.Tensor of shape (4,) - [w, x, y, z]
            gripper: float
        """
        t_frac: float = (t - curr_waypoint.t) / (next_waypoint.t - curr_waypoint.t)
        t_frac: float = max(0.0, min(1.0, t_frac))  # Clamp to [0, 1]

        curr_xyz: torch.Tensor = curr_waypoint.xyz
        curr_quat: torch.Tensor = curr_waypoint.quat
        curr_grip: float = curr_waypoint.gripper
        next_xyz: torch.Tensor = next_waypoint.xyz
        next_quat: torch.Tensor = next_waypoint.quat
        next_grip: float = next_waypoint.gripper

        # Smootherstep interpolation for position (C2 continuous - smooth position, velocity, acceleration)
        t_smooth = t_frac * t_frac * t_frac * (t_frac * (t_frac * 6.0 - 15.0) + 10.0)

        xyz: torch.Tensor = curr_xyz + (next_xyz - curr_xyz) * t_smooth
        quat: torch.Tensor = math_utils.quat_slerp(curr_quat, next_quat, t_frac)
        gripper: float = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def _get_target_pose(self, t: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get target pose at timestep t by interpolating waypoints.

        Args:
            t: Current timestep

        Returns:
            Tuple of (xyz, quat, gripper)
        """
        # Find waypoint segment
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].t <= t < self.waypoints[i + 1].t:
                if self.waypoints[i].interpolation_type == "circle":
                    return self.interpolate_circle(self.waypoints[i], self.waypoints[i + 1], t)
                else:
                    return self.interpolate(self.waypoints[i], self.waypoints[i + 1], t)

        # If past last waypoint, return last waypoint pose with closed gripper
        last_wp = self.waypoints[-1]
        return last_wp.xyz, last_wp.quat, last_wp.gripper

    def advance(self, obs, env_id) -> torch.Tensor:
        """Generate action from observation.

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts.
                 Each environment dict has keys like: ee_pos, ee_quat, goal_pos, etc.
            env_id: The environment index.

        Returns:
            action: Action for the environment.
        """
        # Generate trajectory once the first observation after reset is available.
        if not self.trajectory_generated:
            self.generate_trajectory(obs, env_id)
            self.trajectory_generated = True

        # Get target pose for the current trajectory timestep.
        target_pos, target_quat, gripper_cmd = self._get_target_pose(self.step_count)
        target_quat = math_utils.quat_unique(target_quat)

        # Some robot/controller stacks define policy quaternions relative to the reset EE orientation.
        target_quat = self._apply_initial_orientation_offset(target_quat)
        target_quat = math_utils.quat_unique(target_quat)

        # Inject noise
        if self.inject_noise:
            # Position noise: Gaussian N(0, std^2)
            pos_noise = math_utils.sample_gaussian(
                0.0,
                self.noise_pos_std_m,
                size=target_pos.shape,
                device=self.device,
            ).to(dtype=target_pos.dtype)
            target_pos = target_pos + pos_noise

            # Orientation noise: apply a small random rotation delta to the target quaternion.
            axis = torch.randn(1, 3, device=self.device, dtype=torch.float32)
            angle = math_utils.sample_gaussian(
                0.0,
                self.noise_rot_std_rad,
                size=1,
                device=self.device,
            ).to(dtype=torch.float32)
            axis = axis / torch.norm(axis, dim=1, keepdim=True)
            delta_quat = math_utils.quat_from_angle_axis(angle, axis)  # (1, 4)
            target_quat = math_utils.quat_mul(delta_quat, target_quat.unsqueeze(0)).squeeze(0)
            target_quat = math_utils.quat_unique(target_quat)

        action_gripper = torch.tensor([gripper_cmd], dtype=torch.float32, device=self.device)

        action = torch.cat([target_pos, target_quat, action_gripper])
        self.step_count += 1

        # Add batch dimension
        return action.unsqueeze(0)
