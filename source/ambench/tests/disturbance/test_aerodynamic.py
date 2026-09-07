# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from ambench.disturbance.aerodynamic import apply_ground_effect


def _ground_effect_at_distance(distance: float) -> tuple[torch.Tensor, torch.Tensor]:
    rotor_thrust_vectors = torch.tensor([[[0.0, 0.0, 8.0], [0.0, 0.0, 16.0]]])
    modified, _, difference = apply_ground_effect(
        rotor_thrust_vectors=rotor_thrust_vectors,
        motor_positions_w=torch.zeros((1, 2, 3)),
        motor_axis_directions_w=torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
        aerodynamic_coeffs={
            "propeller_radius": 0.2,
            "ground_effect_b": 1.0,
            "ground_effect_k": 0.2,
            "max_raycast_distance": 5.0,
        },
        d_ground=torch.full((1, 2), distance),
    )
    return modified, difference


def test_ground_effect_matches_analytical_ratio_at_minimum_distance() -> None:
    modified, difference = _ground_effect_at_distance(0.05)

    expected = torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]]])
    torch.testing.assert_close(modified, expected)
    torch.testing.assert_close(difference, expected - torch.tensor([[[0.0, 0.0, 8.0], [0.0, 0.0, 16.0]]]))


def test_ground_effect_clamps_distances_below_quarter_radius() -> None:
    at_minimum, _ = _ground_effect_at_distance(0.05)
    below_minimum, _ = _ground_effect_at_distance(0.01)

    torch.testing.assert_close(below_minimum, at_minimum)


def test_ground_effect_approaches_unmodified_thrust_in_far_field() -> None:
    modified, difference = _ground_effect_at_distance(5.0)
    original = torch.tensor([[[0.0, 0.0, 8.0], [0.0, 0.0, 16.0]]])

    assert torch.all(torch.linalg.vector_norm(modified, dim=-1) >= torch.linalg.vector_norm(original, dim=-1))
    torch.testing.assert_close(modified, original, rtol=3.0e-5, atol=0.0)
    torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0.0, atol=4.0e-4)


@pytest.mark.parametrize(
    "coefficients",
    [
        {"ground_effect_b": 0.0},
        {"ground_effect_b": float("nan")},
        {"ground_effect_k": -0.1},
        {"ground_effect_k": float("inf")},
    ],
)
def test_ground_effect_rejects_invalid_coefficients(coefficients: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        apply_ground_effect(
            rotor_thrust_vectors=torch.ones((1, 1, 3)),
            motor_positions_w=torch.zeros((1, 1, 3)),
            motor_axis_directions_w=torch.tensor([[[0.0, 0.0, 1.0]]]),
            aerodynamic_coeffs={"propeller_radius": 0.2, **coefficients},
            d_ground=torch.ones((1, 1)),
        )
