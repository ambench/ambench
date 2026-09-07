# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Batched reduced-order rotor dynamics independent of the simulator runtime."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ambench.robots.robot_cfg import RotorActuatorCfg


@dataclass(frozen=True)
class RotorActuatorOutput:
    """Realized normalized rotor state and per-step actuator telemetry."""

    commanded_thrust: torch.Tensor
    commanded_normalized_speed: torch.Tensor
    normalized_speed: torch.Tensor
    normalized_acceleration: torch.Tensor
    thrust: torch.Tensor
    thrust_saturated: torch.Tensor
    acceleration_limited: torch.Tensor


class RotorActuator:
    """Advance a reduced-order normalized rotor-speed state.

    Allocated thrust is converted to ``q_command = sqrt(T_command / T_max)``.
    An optional exact first-order update models the identified effective rotor
    response, and an optional symmetric rate cap bounds ``dq/dt``. Realized
    thrust is then ``T_max * q**2``. This representation avoids unmeasured
    motor torque, rotor inertia, and thrust-coefficient parameters while still
    exposing the transient effects under study.

    A reset environment initializes at its first valid command. This prevents
    an artificial spin-up transient when Isaac Lab resets a vehicle in flight;
    subsequent commands advance the state normally. All state tensors have
    shape ``(num_envs, num_rotors)``.
    """

    def __init__(
        self,
        cfg: RotorActuatorCfg,
        *,
        num_envs: int,
        num_rotors: int,
        dt: float,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if num_rotors <= 0:
            raise ValueError("num_rotors must be positive.")
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point.")

        min_thrust, max_thrust = cfg.thrust_limits
        scalar_fields = {
            "minimum thrust": min_thrust,
            "maximum thrust": max_thrust,
        }
        if cfg.response_time_constant_s is not None:
            scalar_fields["response_time_constant_s"] = cfg.response_time_constant_s
        if cfg.normalized_acceleration_limit_per_s is not None:
            scalar_fields["normalized_acceleration_limit_per_s"] = cfg.normalized_acceleration_limit_per_s
        nonfinite_fields = [name for name, value in scalar_fields.items() if not isfinite(value)]
        if nonfinite_fields:
            raise ValueError(f"Rotor actuator parameters must be finite: {', '.join(nonfinite_fields)}.")
        if min_thrust < 0.0 or max_thrust <= min_thrust:
            raise ValueError(f"Expected nonnegative increasing thrust limits, got {cfg.thrust_limits}.")
        if cfg.response_time_constant_s is not None and cfg.response_time_constant_s <= 0.0:
            raise ValueError("response_time_constant_s must be positive when configured.")
        if cfg.normalized_acceleration_limit_per_s is not None and cfg.normalized_acceleration_limit_per_s <= 0.0:
            raise ValueError("normalized_acceleration_limit_per_s must be positive when configured.")
        if cfg.response_time_constant_s is None and cfg.normalized_acceleration_limit_per_s is None:
            raise ValueError("RotorActuator requires a response time constant, an acceleration limit, or both.")

        self.num_envs = num_envs
        self.num_rotors = num_rotors
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.thrust_limits = float(min_thrust), float(max_thrust)
        self.response_time_constant_s = (
            None if cfg.response_time_constant_s is None else float(cfg.response_time_constant_s)
        )
        self.normalized_acceleration_limit_per_s = (
            None if cfg.normalized_acceleration_limit_per_s is None else float(cfg.normalized_acceleration_limit_per_s)
        )
        self._response_fraction = (
            1.0 if self.response_time_constant_s is None else 1.0 - exp(-self.dt / self.response_time_constant_s)
        )
        self._minimum_normalized_speed = sqrt(min_thrust / max_thrust)

        state_shape = (num_envs, num_rotors)
        self.normalized_speed = torch.full(
            state_shape,
            self._minimum_normalized_speed,
            device=self.device,
            dtype=self.dtype,
        )
        self._initialized = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    def step(self, thrust_command: torch.Tensor) -> RotorActuatorOutput:
        """Advance the rotor state by one integration step.

        ``thrust_command`` is the finite, unsaturated allocation result. The
        actuator applies its physical thrust limits before transient dynamics.
        """
        thrust_command = torch.as_tensor(thrust_command, device=self.device, dtype=self.dtype)
        expected_shape = (self.num_envs, self.num_rotors)
        if thrust_command.shape != expected_shape:
            raise ValueError(f"Expected thrust_command with shape {expected_shape}, got {thrust_command.shape}.")

        min_thrust, max_thrust = self.thrust_limits
        commanded_thrust = torch.clamp(thrust_command, min=min_thrust, max=max_thrust)
        thrust_saturated = commanded_thrust != thrust_command
        commanded_normalized_speed = torch.sqrt(commanded_thrust / max_thrust)

        initialized = self._initialized.unsqueeze(-1)
        previous_speed = torch.where(initialized, self.normalized_speed, commanded_normalized_speed)
        response_speed = previous_speed + self._response_fraction * (commanded_normalized_speed - previous_speed)
        unconstrained_delta = response_speed - previous_speed
        if self.normalized_acceleration_limit_per_s is None:
            realized_delta = unconstrained_delta
            acceleration_limited = torch.zeros_like(thrust_saturated)
        else:
            maximum_delta = self.normalized_acceleration_limit_per_s * self.dt
            realized_delta = torch.clamp(unconstrained_delta, min=-maximum_delta, max=maximum_delta)
            acceleration_limited = initialized & (realized_delta != unconstrained_delta)

        next_speed = torch.clamp(
            previous_speed + realized_delta,
            min=self._minimum_normalized_speed,
            max=1.0,
        )
        normalized_acceleration = torch.where(
            initialized,
            (next_speed - previous_speed) / self.dt,
            torch.zeros_like(next_speed),
        )
        self.normalized_speed.copy_(next_speed)
        self._initialized.fill_(True)
        thrust = max_thrust * self.normalized_speed.square()

        return RotorActuatorOutput(
            commanded_thrust=commanded_thrust,
            commanded_normalized_speed=commanded_normalized_speed,
            normalized_speed=self.normalized_speed.clone(),
            normalized_acceleration=normalized_acceleration,
            thrust=thrust,
            thrust_saturated=thrust_saturated,
            acceleration_limited=acceleration_limited,
        )

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        """Reset selected environments to initialize from their next command."""
        if env_ids is None:
            self.normalized_speed.fill_(self._minimum_normalized_speed)
            self._initialized.fill_(False)
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_tensor.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional sequence of environment indices.")
        if env_ids_tensor.numel() == 0:
            return
        if torch.any(env_ids_tensor < 0) or torch.any(env_ids_tensor >= self.num_envs):
            raise IndexError("env_ids contains an environment index outside the actuator batch.")
        self.normalized_speed[env_ids_tensor] = self._minimum_normalized_speed
        self._initialized[env_ids_tensor] = False
