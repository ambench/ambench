# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Internal frame helpers shared by dataset and video recording."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_depth_image, prepare_rgb_image

logger = logging.getLogger(__name__)

DEPTH_SUFFIX = "_depth"


def resolve_camera_names(
    env: Any,
    requested_camera_names: list[str] | None = None,
    *,
    require_any: bool = False,
) -> list[str]:
    """Resolve requested camera names against the environment sensor registry."""
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    requested_names = list(requested_camera_names or [])
    sensors = env_unwrapped.scene.sensors
    available_sensor_names = sorted(sensors.keys())

    resolved_names = [name for name in requested_names if name in sensors]
    missing_names = [name for name in requested_names if name not in sensors]
    if missing_names:
        logger.warning(
            "Requested camera sensor(s) %s were not found. Available environment sensors: %s",
            missing_names,
            available_sensor_names or "<none>",
        )

    if require_any and not resolved_names:
        raise ValueError(
            "Expected at least one valid camera sensor, but none were resolved."
            f" Requested camera names: {requested_names}."
            f" Available environment sensors: {available_sensor_names or '<none>'}."
        )

    return resolved_names


def capture_camera_frames(
    env: Any,
    env_id: int,
    camera_names: list[str] | None = None,
    *,
    data_types: tuple[str, ...] = ("rgb",),
) -> dict[str, np.ndarray]:
    """Capture requested camera frame streams for one environment."""
    unsupported_types = [data_type for data_type in data_types if data_type not in {"rgb", "depth"}]
    if unsupported_types:
        raise ValueError(
            f"Unsupported camera frame data_types: {unsupported_types}. Supported values are ['rgb', 'depth']."
        )

    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    sensors = env_unwrapped.scene.sensors
    frames: dict[str, np.ndarray] = {}

    for camera_name in list(camera_names or []):
        if camera_name not in sensors:
            continue

        camera = sensors[camera_name]
        if "rgb" in data_types and "rgb" in camera.data.output:
            frames[camera_name] = prepare_rgb_image(get_camera_rgb(env_unwrapped, camera_name, env_id))

        depth_output = None
        if "depth" in data_types:
            depth_output = camera.data.output.get("distance_to_image_plane", camera.data.output.get("depth"))

        if depth_output is not None:
            depth_key = f"{camera_name}{DEPTH_SUFFIX}"
            frames[depth_key] = prepare_depth_image(depth_output[env_id])

    return frames
