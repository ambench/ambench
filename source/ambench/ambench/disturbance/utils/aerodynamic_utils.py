# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""PhysX raycast helpers for aerodynamic distance models."""

from __future__ import annotations

import math
from typing import Any

import carb
import torch
from omni.physx import get_physx_scene_query_interface

_NUM_WALL_RAYS = 10
_WALL_RAY_HALF_ANGLE = math.radians(45.0)


def _validate_raycast_distance(max_raycast_distance: float) -> None:
    if max_raycast_distance <= 0.0:
        raise ValueError(f"Expected a positive raycast distance, got {max_raycast_distance}.")


def _validate_vector_batch(name: str, values: torch.Tensor) -> None:
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected {name} with shape (num_envs, num_motors, 3), got {values.shape}.")


def _closest_non_robot_hit(
    scene_query: Any,
    origin: carb.Float3,
    direction: carb.Float3,
    max_raycast_distance: float,
    robot_prim_prefix: str,
) -> tuple[float, bool]:
    closest_distance = max_raycast_distance
    hit_found = False

    def report_raycast(raycast_hit: Any) -> bool:
        nonlocal closest_distance, hit_found
        if raycast_hit.rigid_body.startswith(robot_prim_prefix):
            return True
        if raycast_hit.distance < closest_distance:
            closest_distance = float(raycast_hit.distance)
            hit_found = True
        return True

    scene_query.raycast_all(origin, direction, max_raycast_distance, report_raycast)
    return closest_distance, hit_found


def compute_ground_effect_distance(
    motor_positions_w: torch.Tensor,
    motor_axis_directions_w: torch.Tensor,
    max_raycast_distance: float = 5.0,
) -> torch.Tensor:
    """Return the nearest non-robot hit along each motor's negative thrust axis."""

    _validate_vector_batch("motor positions", motor_positions_w)
    _validate_vector_batch("motor axis directions", motor_axis_directions_w)
    if motor_axis_directions_w.shape != motor_positions_w.shape:
        raise ValueError(
            "Expected motor positions and axis directions to have identical shapes, "
            f"got {motor_positions_w.shape} and {motor_axis_directions_w.shape}."
        )
    _validate_raycast_distance(max_raycast_distance)

    num_envs, num_motors, _ = motor_positions_w.shape
    output_device = motor_positions_w.device
    motor_positions_cpu = motor_positions_w.detach().to(device="cpu", dtype=torch.float32)
    motor_directions_cpu = motor_axis_directions_w.detach().to(device="cpu", dtype=torch.float32)
    distances = torch.full((num_envs, num_motors), max_raycast_distance, dtype=torch.float32)
    scene_query = get_physx_scene_query_interface()

    for env_idx in range(num_envs):
        robot_prim_prefix = f"/World/envs/env_{env_idx}/Robot/"
        for motor_idx in range(num_motors):
            position = motor_positions_cpu[env_idx, motor_idx]
            direction = -motor_directions_cpu[env_idx, motor_idx]
            direction_norm = float(torch.linalg.vector_norm(direction))
            if direction_norm <= 1e-8:
                raise ValueError(f"Motor axis direction is zero for env {env_idx}, motor {motor_idx}.")
            direction /= direction_norm

            origin = carb.Float3(*position.tolist())
            ray_direction = carb.Float3(*direction.tolist())
            distance, _ = _closest_non_robot_hit(
                scene_query,
                origin,
                ray_direction,
                max_raycast_distance,
                robot_prim_prefix,
            )
            distances[env_idx, motor_idx] = distance

    return distances.to(output_device)


def compute_near_wall_effect_distance(
    motor_positions_w: torch.Tensor,
    drone_position_w: torch.Tensor,
    propeller_radius: float,
    max_raycast_distance: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return wall clearance, normalized clearance, and nearest-wall direction."""

    _validate_vector_batch("motor positions", motor_positions_w)
    num_envs, num_motors, _ = motor_positions_w.shape
    if drone_position_w.shape != (num_envs, 3):
        raise ValueError(f"Expected drone positions with shape {(num_envs, 3)}, got {drone_position_w.shape}.")
    if propeller_radius <= 0.0:
        raise ValueError(f"Expected a positive propeller radius, got {propeller_radius}.")
    _validate_raycast_distance(max_raycast_distance)

    output_device = motor_positions_w.device
    motor_positions_cpu = motor_positions_w.detach().to(device="cpu", dtype=torch.float32)
    drone_positions_cpu = drone_position_w.detach().to(device="cpu", dtype=torch.float32)
    wall_distances = torch.empty((num_envs, num_motors), dtype=torch.float32)
    wall_directions = torch.zeros((num_envs, num_motors, 3), dtype=torch.float32)
    scene_query = get_physx_scene_query_interface()
    angle_step = 2.0 * _WALL_RAY_HALF_ANGLE / (_NUM_WALL_RAYS - 1)

    for env_idx in range(num_envs):
        robot_prim_prefix = f"/World/envs/env_{env_idx}/Robot/"
        drone_position = drone_positions_cpu[env_idx]
        for motor_idx in range(num_motors):
            motor_position = motor_positions_cpu[env_idx, motor_idx]
            radial_direction = motor_position[:2] - drone_position[:2]
            if float(torch.linalg.vector_norm(radial_direction)) <= 1e-6:
                base_angle = 0.0
            else:
                base_angle = math.atan2(float(radial_direction[1]), float(radial_direction[0]))

            origin = carb.Float3(*motor_position.tolist())
            nearest_distance = max_raycast_distance
            nearest_direction = None
            for ray_idx in range(_NUM_WALL_RAYS):
                angle = base_angle - _WALL_RAY_HALF_ANGLE + ray_idx * angle_step
                direction_values = (math.cos(angle), math.sin(angle), 0.0)
                ray_direction = carb.Float3(*direction_values)
                distance, hit_found = _closest_non_robot_hit(
                    scene_query,
                    origin,
                    ray_direction,
                    max_raycast_distance,
                    robot_prim_prefix,
                )
                if hit_found and distance < nearest_distance:
                    nearest_distance = distance
                    nearest_direction = direction_values

            wall_distances[env_idx, motor_idx] = max(nearest_distance - propeller_radius, 0.0)
            if nearest_direction is not None:
                wall_directions[env_idx, motor_idx] = torch.tensor(nearest_direction)

    normalized_distances = wall_distances / propeller_radius
    return (
        wall_distances.to(output_device),
        normalized_distances.to(output_device),
        wall_directions.to(output_device),
    )
