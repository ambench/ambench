# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_numpy_array(value: Any) -> np.ndarray:
    """Return a detached CPU NumPy view when possible."""

    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_rgb_image(image: Any) -> np.ndarray:
    """Convert an HWC RGB(A) image to contiguous HWC RGB ``uint8``.

    Floating-point inputs are expected to use the normalized ``[0, 1]`` range.
    Extra channels, such as alpha, are discarded.
    """

    image_array = to_numpy_array(image)

    if image_array.ndim != 3 or image_array.shape[-1] < 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image_array.shape}")

    if image_array.shape[-1] > 3:
        image_array = image_array[..., :3]

    if np.issubdtype(image_array.dtype, np.floating):
        image_array = np.clip(image_array, 0.0, 1.0) * 255.0

    return np.ascontiguousarray(image_array, dtype=np.uint8)


def prepare_rgb_tensor(image: torch.Tensor) -> torch.Tensor:
    """Convert HWC or BHWC ``uint8`` RGB to normalized CHW or BCHW ``float32``."""

    if image.dtype != torch.uint8:
        raise TypeError(f"Expected a uint8 RGB image, got {image.dtype}.")
    if image.ndim not in (3, 4) or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC or BHWC RGB image, got shape {tuple(image.shape)}.")
    return image.movedim(-1, -3).contiguous().to(dtype=torch.float32).div_(255.0)


def prepare_depth_image(image: Any) -> np.ndarray:
    """Convert a depth image to contiguous HWC1 ``float32``."""

    image_array = to_numpy_array(image)

    if image_array.ndim == 2:
        image_array = image_array[..., None]
    elif image_array.ndim != 3:
        raise ValueError(f"Expected an HWC depth image, got shape {image_array.shape}")

    if image_array.shape[-1] != 1:
        raise ValueError(f"Expected a single-channel depth image, got shape {image_array.shape}")

    depth = np.ascontiguousarray(image_array, dtype=np.float32)
    return np.nan_to_num(depth, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def depth_to_preview_rgb(depth_image: Any) -> np.ndarray:
    """Convert positive depth values to a frame-normalized RGB preview."""

    depth = prepare_depth_image(depth_image)[..., 0]
    valid_mask = depth > 0.0
    normalized = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid_mask):
        valid_depth = depth[valid_mask]
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())
        if depth_max > depth_min:
            normalized[valid_mask] = (valid_depth - depth_min) / (depth_max - depth_min)
        else:
            normalized[valid_mask] = 1.0
    depth_u8 = (normalized * 255.0).astype(np.uint8, copy=False)
    return np.repeat(depth_u8[..., None], 3, axis=-1)
