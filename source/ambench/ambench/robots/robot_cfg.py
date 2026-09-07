# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Data-only robot specifications shared by environments and controllers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass
from math import isfinite
from typing import Any

import torch
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ambench.robots.visuals.propeller_cfg import PropellerVizCfg

GripperWidthConverter = Callable[[float], float]


@dataclass(frozen=True)
class RotorLayout:
    """Resolved rotor geometry in the robot body frame.

    ``spin_directions`` follows the established allocation convention: its
    sign selects the direction of positive airframe reaction torque about each
    thrust axis. The alternating signs correspond to the robot's CW/CCW rotor
    arrangement.
    """

    positions_b: torch.Tensor
    thrust_axes_b: torch.Tensor
    spin_directions: torch.Tensor
    tilt_cos_axes_b: torch.Tensor | None = None
    tilt_sin_axes_b: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected_vector_shape = self.positions_b.shape
        if self.positions_b.ndim != 2 or self.positions_b.shape[1] != 3:
            raise ValueError(f"Expected rotor positions with shape (num_rotors, 3), got {self.positions_b.shape}.")
        if self.positions_b.shape[0] == 0:
            raise ValueError("Rotor layouts must contain at least one rotor.")
        if self.thrust_axes_b.shape != expected_vector_shape:
            raise ValueError(
                "Rotor thrust axes must have the same shape as rotor positions, "
                f"got {self.thrust_axes_b.shape} and {expected_vector_shape}."
            )
        expected_spin_shape = (self.positions_b.shape[0],)
        if self.spin_directions.shape != expected_spin_shape:
            raise ValueError(
                f"Expected rotor spin directions with shape {expected_spin_shape}, got {self.spin_directions.shape}."
            )
        if not self.positions_b.is_floating_point() or not self.thrust_axes_b.is_floating_point():
            raise ValueError("Rotor positions and thrust axes must use floating-point tensors.")
        if (
            self.thrust_axes_b.device != self.positions_b.device
            or self.spin_directions.device != self.positions_b.device
        ):
            raise ValueError("Rotor layout tensors must use the same device.")
        for label, values in (
            ("positions", self.positions_b),
            ("thrust axes", self.thrust_axes_b),
            ("spin directions", self.spin_directions),
        ):
            if not torch.all(torch.isfinite(values)):
                raise ValueError(f"Rotor {label} must contain only finite values.")
        if not torch.all((self.spin_directions == 1.0) | (self.spin_directions == -1.0)):
            raise ValueError("Rotor spin directions must contain only -1 or +1.")
        thrust_axis_norms = torch.linalg.vector_norm(self.thrust_axes_b, dim=-1)
        if not torch.allclose(thrust_axis_norms, torch.ones_like(thrust_axis_norms), rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("Rotor thrust axes must be unit vectors.")

        has_cos_basis = self.tilt_cos_axes_b is not None
        has_sin_basis = self.tilt_sin_axes_b is not None
        if has_cos_basis != has_sin_basis:
            raise ValueError("Variable-tilt layouts require both cosine and sine thrust-axis bases.")
        if self.tilt_cos_axes_b is not None and self.tilt_cos_axes_b.shape != expected_vector_shape:
            raise ValueError(
                f"Expected tilt cosine axes with shape {expected_vector_shape}, got {self.tilt_cos_axes_b.shape}."
            )
        if self.tilt_sin_axes_b is not None and self.tilt_sin_axes_b.shape != expected_vector_shape:
            raise ValueError(
                f"Expected tilt sine axes with shape {expected_vector_shape}, got {self.tilt_sin_axes_b.shape}."
            )
        for label, values in (
            ("tilt cosine axes", self.tilt_cos_axes_b),
            ("tilt sine axes", self.tilt_sin_axes_b),
        ):
            if values is None:
                continue
            if not values.is_floating_point():
                raise ValueError(f"Rotor {label} must use a floating-point tensor.")
            if values.device != self.positions_b.device:
                raise ValueError("Rotor layout tensors must use the same device.")
            if not torch.all(torch.isfinite(values)):
                raise ValueError(f"Rotor {label} must contain only finite values.")
            axis_norms = torch.linalg.vector_norm(values, dim=-1)
            if not torch.allclose(axis_norms, torch.ones_like(axis_norms), rtol=1.0e-5, atol=1.0e-6):
                raise ValueError(f"Rotor {label} must contain unit vectors.")
        if self.tilt_cos_axes_b is not None and self.tilt_sin_axes_b is not None:
            basis_dot_products = torch.sum(self.tilt_cos_axes_b * self.tilt_sin_axes_b, dim=-1)
            if not torch.allclose(
                basis_dot_products,
                torch.zeros_like(basis_dot_products),
                rtol=0.0,
                atol=1.0e-5,
            ):
                raise ValueError("Rotor tilt cosine and sine axes must be orthogonal.")

    @property
    def num_rotors(self) -> int:
        """Number of rotors in the resolved layout."""
        return self.positions_b.shape[0]

    @property
    def is_variable_tilt(self) -> bool:
        """Whether the layout provides variable-tilt allocation bases."""
        return self.tilt_cos_axes_b is not None


RotorLayoutProvider = Callable[..., RotorLayout]


@configclass
class EndEffectorFrameCfg:
    """Relationship between the controlled link and the task command frame."""

    tool_tip_offset_local: tuple[float, float, float] = (0.18, 0.0, 0.0)
    command_to_link_quat_wxyz: tuple[float, float, float, float] | None = None


@configclass
class GripperSpecCfg:
    """Ordered gripper joints and their semantic open/closed calibration."""

    joint_names: tuple[str, ...] = ()
    open_joint_positions: tuple[float, ...] | None = None
    closed_joint_positions: tuple[float, ...] | None = None
    joint_position_from_object_width: GripperWidthConverter | None = None

    def __post_init__(self) -> None:
        for label, positions in (
            ("open_joint_positions", self.open_joint_positions),
            ("closed_joint_positions", self.closed_joint_positions),
        ):
            if positions is not None and len(positions) != len(self.joint_names):
                raise ValueError(f"{label} must contain one value per gripper joint.")


@configclass
class AerodynamicCfg:
    """Coefficients for the drag, rotor-proximity, wall-effect, and wind models.

    Ground effect follows
    ``T_commanded / T_actual = b - k * (R / (4 z))**2``.
    """

    max_raycast_distance: float = 5.0
    ground_effect_b: float = 1.0
    ground_effect_k: float = 0.171
    wall_effect_a1: float = 0.05
    wall_effect_b1: float = 0.34
    wall_effect_a2: float = 0.02
    wall_effect_b2: float = 0.25
    drag_coefficients: tuple[float, float, float] = (0.1, 0.1, 0.1)
    # Constant world-frame force, applied when the environment enables wind.
    wind_force_w: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self, *, propeller_radius: float) -> dict[str, Any]:
        """Return the mapping consumed by the established aerodynamic functions."""
        return {
            "propeller_radius": propeller_radius,
            "max_raycast_distance": self.max_raycast_distance,
            "ground_effect_b": self.ground_effect_b,
            "ground_effect_k": self.ground_effect_k,
            "wall_effect_a1": self.wall_effect_a1,
            "wall_effect_b1": self.wall_effect_b1,
            "wall_effect_a2": self.wall_effect_a2,
            "wall_effect_b2": self.wall_effect_b2,
            "drag_coefficients": torch.tensor(self.drag_coefficients),
        }


@configclass
class RotorActuatorCfg:
    """Steady-state rotor properties and optional normalized-speed dynamics.

    ``response_time_constant_s`` is an effective first-order rotor response.
    ``normalized_acceleration_limit_per_s`` bounds the rate of
    ``sqrt(thrust / maximum_thrust)`` and therefore does not require a rotor
    speed sensor or a calibrated thrust coefficient. The two effects are
    independently optional; actuation remains instantaneous when both are
    disabled. ``reaction_torque_ratio`` is the reaction moment per unit thrust
    in metres.
    """

    thrust_limits: tuple[float, float] = MISSING
    reaction_torque_ratio: float = 0.02
    propeller_radius: float = 0.152

    response_time_constant_s: float | None = None
    normalized_acceleration_limit_per_s: float | None = None

    def __post_init__(self) -> None:
        min_thrust, max_thrust = self.thrust_limits
        scalar_fields = {
            "minimum thrust": min_thrust,
            "maximum thrust": max_thrust,
            "reaction_torque_ratio": self.reaction_torque_ratio,
            "propeller_radius": self.propeller_radius,
        }
        optional_fields = {
            "response_time_constant_s": self.response_time_constant_s,
            "normalized_acceleration_limit_per_s": self.normalized_acceleration_limit_per_s,
        }
        scalar_fields.update({name: value for name, value in optional_fields.items() if value is not None})
        nonfinite_fields = [name for name, value in scalar_fields.items() if not isfinite(value)]
        if nonfinite_fields:
            raise ValueError(f"Rotor actuator parameters must be finite: {', '.join(nonfinite_fields)}.")
        if min_thrust < 0.0 or max_thrust <= min_thrust:
            raise ValueError(f"Expected nonnegative increasing thrust limits, got {self.thrust_limits}.")
        if self.reaction_torque_ratio < 0.0:
            raise ValueError("reaction_torque_ratio must be nonnegative.")
        if self.propeller_radius <= 0.0:
            raise ValueError("propeller_radius must be positive.")
        if self.response_time_constant_s is not None and self.response_time_constant_s <= 0.0:
            raise ValueError("response_time_constant_s must be positive when configured.")
        if self.normalized_acceleration_limit_per_s is not None and self.normalized_acceleration_limit_per_s <= 0.0:
            raise ValueError("normalized_acceleration_limit_per_s must be positive when configured.")

    @property
    def has_dynamics(self) -> bool:
        """Whether either optional transient effect is enabled."""
        return self.response_time_constant_s is not None or self.normalized_acceleration_limit_per_s is not None


@configclass
class MultirotorSpecCfg:
    """Optional multirotor components attached to a robot."""

    layout_provider: RotorLayoutProvider = MISSING
    actuator: RotorActuatorCfg = MISSING
    fully_actuated: bool = True
    motor_arm_joint_names: tuple[str, ...] = ()
    aerodynamics: AerodynamicCfg | None = None
    propeller_viz: PropellerVizCfg | None = None


@configclass
class RobotSpecCfg:
    """Authoritative robot asset, morphology, frames, and optional capabilities."""

    robot_id: str = MISSING
    asset: ArticulationCfg = MISSING
    control_body_name: str = MISSING
    base_body_name: str = MISSING
    ee_body_name: str | None = None
    arm_joint_names: tuple[str, ...] = ()
    end_effector: EndEffectorFrameCfg | None = None
    gripper: GripperSpecCfg | None = None
    max_arm_reach: float | None = None
    multirotor: MultirotorSpecCfg | None = None

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise ValueError("robot_id must not be empty.")
        if len(set(self.arm_joint_names)) != len(self.arm_joint_names):
            raise ValueError("arm_joint_names must be unique.")
        if self.gripper is not None and len(set(self.gripper.joint_names)) != len(self.gripper.joint_names):
            raise ValueError("gripper joint names must be unique.")
