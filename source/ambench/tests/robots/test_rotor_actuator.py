# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ambench.robots.rotor_actuator import RotorActuator


def _actuator_cfg(**overrides: object) -> SimpleNamespace:
    values = {
        "thrust_limits": (0.0, 100.0),
        "response_time_constant_s": 0.1,
        "normalized_acceleration_limit_per_s": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _actuator(
    *,
    cfg: SimpleNamespace | None = None,
    num_envs: int = 1,
    num_rotors: int = 1,
    dt: float = 0.01,
    device: str = "cpu",
) -> RotorActuator:
    return RotorActuator(
        _actuator_cfg() if cfg is None else cfg,
        num_envs=num_envs,
        num_rotors=num_rotors,
        dt=dt,
        device=device,
    )


def test_zero_command_preserves_zero_state() -> None:
    actuator = _actuator(num_envs=2, num_rotors=3)

    output = actuator.step(torch.zeros((2, 3)))

    assert torch.count_nonzero(output.commanded_normalized_speed) == 0
    assert torch.count_nonzero(output.normalized_speed) == 0
    assert torch.count_nonzero(output.normalized_acceleration) == 0
    assert torch.count_nonzero(output.thrust) == 0
    assert not torch.any(output.thrust_saturated)
    assert not torch.any(output.acceleration_limited)


def test_first_command_initializes_without_artificial_spinup() -> None:
    actuator = _actuator()

    output = actuator.step(torch.tensor([[36.0]]))

    assert output.commanded_normalized_speed.item() == pytest.approx(0.6)
    assert output.normalized_speed.item() == pytest.approx(0.6)
    assert output.normalized_acceleration.item() == pytest.approx(0.0)
    assert output.thrust.item() == pytest.approx(36.0)


def test_response_uses_exact_first_order_update() -> None:
    actuator = _actuator(cfg=_actuator_cfg(response_time_constant_s=0.2), dt=0.05)
    actuator.step(torch.zeros((1, 1)))

    output = actuator.step(torch.tensor([[100.0]]))

    expected_speed = 1.0 - torch.exp(torch.tensor(-0.05 / 0.2)).item()
    assert output.normalized_speed.item() == pytest.approx(expected_speed)
    assert output.normalized_acceleration.item() == pytest.approx(expected_speed / 0.05)
    assert output.thrust.item() == pytest.approx(100.0 * expected_speed**2)
    assert not output.acceleration_limited.item()


def test_acceleration_limit_can_run_without_first_order_response() -> None:
    actuator = _actuator(
        cfg=_actuator_cfg(
            response_time_constant_s=None,
            normalized_acceleration_limit_per_s=2.5,
        ),
        dt=0.01,
    )
    actuator.step(torch.zeros((1, 1)))

    output = actuator.step(torch.tensor([[100.0]]))

    assert output.normalized_speed.item() == pytest.approx(0.025)
    assert output.normalized_acceleration.item() == pytest.approx(2.5)
    assert output.thrust.item() == pytest.approx(0.0625)
    assert output.acceleration_limited.item()


def test_response_and_acceleration_limit_compose() -> None:
    actuator = _actuator(
        cfg=_actuator_cfg(normalized_acceleration_limit_per_s=0.5),
        dt=0.1,
    )
    actuator.step(torch.zeros((1, 1)))

    output = actuator.step(torch.tensor([[100.0]]))

    assert output.normalized_speed.item() == pytest.approx(0.05)
    assert output.normalized_acceleration.item() == pytest.approx(0.5)
    assert output.acceleration_limited.item()


def test_acceleration_limit_is_symmetric_for_braking() -> None:
    actuator = _actuator(
        cfg=_actuator_cfg(
            response_time_constant_s=None,
            normalized_acceleration_limit_per_s=2.5,
        ),
        dt=0.1,
    )
    actuator.step(torch.tensor([[100.0]]))

    output = actuator.step(torch.zeros((1, 1)))

    assert output.normalized_speed.item() == pytest.approx(0.75)
    assert output.normalized_acceleration.item() == pytest.approx(-2.5)
    assert output.thrust.item() == pytest.approx(56.25)
    assert output.acceleration_limited.item()


def test_thrust_limits_are_applied_before_normalization() -> None:
    actuator = _actuator(num_rotors=2)

    output = actuator.step(torch.tensor([[-1.0, 400.0]]))

    assert torch.equal(output.commanded_thrust, torch.tensor([[0.0, 100.0]]))
    assert torch.equal(output.commanded_normalized_speed, torch.tensor([[0.0, 1.0]]))
    assert torch.equal(output.thrust, torch.tensor([[0.0, 100.0]]))
    assert torch.equal(output.thrust_saturated, torch.tensor([[True, True]]))


def test_nonzero_minimum_thrust_defines_normalized_speed_floor() -> None:
    actuator = _actuator(
        cfg=_actuator_cfg(
            thrust_limits=(25.0, 100.0),
            response_time_constant_s=None,
            normalized_acceleration_limit_per_s=1.0,
        )
    )

    output = actuator.step(torch.zeros((1, 1)))

    assert output.commanded_thrust.item() == pytest.approx(25.0)
    assert output.normalized_speed.item() == pytest.approx(0.5)
    assert output.thrust.item() == pytest.approx(25.0)


def test_selective_reset_reinitializes_from_next_command() -> None:
    actuator = _actuator(
        cfg=_actuator_cfg(
            response_time_constant_s=None,
            normalized_acceleration_limit_per_s=1.0,
        ),
        num_envs=2,
        dt=0.1,
    )
    actuator.step(torch.tensor([[0.0], [100.0]]))
    actuator.step(torch.tensor([[100.0], [0.0]]))
    actuator.reset(torch.tensor([0]))

    output = actuator.step(torch.tensor([[25.0], [0.0]]))

    assert output.normalized_speed[0].item() == pytest.approx(0.5)
    assert output.normalized_acceleration[0].item() == pytest.approx(0.0)
    assert output.normalized_speed[1].item() == pytest.approx(0.8)
    assert output.normalized_acceleration[1].item() == pytest.approx(-1.0)


def test_invalid_shape_and_configuration_are_rejected() -> None:
    actuator = _actuator(num_rotors=2)

    with pytest.raises(ValueError, match="shape"):
        actuator.step(torch.zeros((1, 1)))
    with pytest.raises(IndexError, match="outside"):
        actuator.reset([1])
    with pytest.raises(ValueError, match="requires"):
        _actuator(
            cfg=_actuator_cfg(
                response_time_constant_s=None,
                normalized_acceleration_limit_per_s=None,
            )
        )
    with pytest.raises(ValueError, match="positive"):
        _actuator(cfg=_actuator_cfg(response_time_constant_s=0.0))
    with pytest.raises(ValueError, match="finite"):
        _actuator(cfg=_actuator_cfg(normalized_acceleration_limit_per_s=torch.nan))
    with pytest.raises(ValueError, match="finite and positive"):
        _actuator(dt=float("nan"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_state_and_output_remain_on_device() -> None:
    actuator = _actuator(num_envs=2, num_rotors=3, device="cuda")

    output = actuator.step(torch.ones((2, 3), device="cuda"))

    assert actuator.normalized_speed.device.type == "cuda"
    assert output.thrust.device.type == "cuda"
    assert output.acceleration_limited.device.type == "cuda"
