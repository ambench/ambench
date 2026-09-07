# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera utility functions."""

from typing import Any, Literal

import torch


def get_camera_rgb(
    env: Any,
    camera_name: str,
    env_indices: int | list[int] | None = None,
) -> torch.Tensor:
    """Return raw HWC or BHWC RGB data from an environment camera sensor."""

    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    sensor = getattr(env_unwrapped, camera_name, None)
    if sensor is None:
        try:
            sensor = env_unwrapped.scene.sensors[camera_name]
        except (AttributeError, KeyError) as error:
            raise KeyError(f"Camera '{camera_name}' was not found in the environment sensors.") from error

    rgb = sensor.data.output.get("rgb")
    if rgb is None:
        raise KeyError(f"Camera '{camera_name}' has no RGB output.")
    if not torch.is_tensor(rgb):
        rgb = torch.as_tensor(rgb)
    if env_indices is not None:
        rgb = rgb[env_indices]
    if rgb.ndim not in (3, 4) or rgb.shape[-1] < 3:
        raise ValueError(f"Camera '{camera_name}' must produce HWC or BHWC RGB images, got {tuple(rgb.shape)}.")
    return rgb[..., :3]


def compute_camera_quat_from_lookat(
    camera_pos: tuple[float, float, float],
    lookat_target: tuple[float, float, float],
    up_axis: Literal["Y", "Z"] = "Z",
) -> tuple[float, float, float, float]:
    """Compute camera quaternion (w, x, y, z) from position and lookat target for ROS convention.

    This function computes the quaternion orientation needed for a camera at `camera_pos`
    to look at `lookat_target`, using the ROS camera convention (+Z forward, -Y up).

    Args:
        camera_pos: Camera position (x, y, z) in world coordinates.
        lookat_target: Target point to look at (x, y, z) in world coordinates.
        up_axis: World up axis, either "Y" or "Z". Defaults to "Z".

    Returns:
        Quaternion (w, x, y, z) for the camera orientation in ROS convention.

    Example:
        >>> from ambench.utils.camera_utils import compute_camera_quat_from_lookat
        >>> quat = compute_camera_quat_from_lookat(
        ...     camera_pos=(-1.5, -1.5, 2.5),
        ...     lookat_target=(2.0, 0.0, 1.0),
        ... )
        >>> # Use with CameraCfg.OffsetCfg(pos=camera_pos, rot=quat, convention="ros")
    """
    # Keep dependency-light camera data helpers importable before AppLauncher initializes Isaac Sim.
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        create_rotation_matrix_from_view,
        quat_from_matrix,
    )

    # Convert to tensors
    eyes = torch.tensor([camera_pos], dtype=torch.float32)
    targets = torch.tensor([lookat_target], dtype=torch.float32)

    # Create rotation matrix from view (returns OpenGL convention)
    rot_matrix = create_rotation_matrix_from_view(eyes, targets, up_axis=up_axis, device="cpu")

    # Convert rotation matrix to quaternion
    quat_opengl = quat_from_matrix(rot_matrix)

    # Convert from OpenGL to ROS convention
    quat_ros = convert_camera_frame_orientation_convention(quat_opengl, origin="opengl", target="ros")

    # Return as tuple (w, x, y, z)
    q = quat_ros[0]
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
