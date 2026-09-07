# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ambench.recording.utils import capture_camera_frames
from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_image, prepare_rgb_tensor


def _env(output: dict[str, torch.Tensor]):
    sensor = SimpleNamespace(data=SimpleNamespace(output=output))
    return SimpleNamespace(scene=SimpleNamespace(sensors={"ee_camera": sensor}))


def test_camera_rgb_capture_and_conversion() -> None:
    rgb = torch.tensor([[[[0, 127, 255], [255, 0, 127]]]], dtype=torch.uint8)
    env = _env({"rgb": rgb})

    frame = get_camera_rgb(env, "ee_camera", 0)
    batch = get_camera_rgb(env, "ee_camera", [0])
    array = prepare_rgb_image(frame)
    tensor = prepare_rgb_tensor(batch)

    assert frame.shape == (1, 2, 3)
    assert array.shape == (1, 2, 3)
    assert array.dtype == np.uint8 and array.flags.c_contiguous
    assert tensor.shape == (1, 3, 1, 2)
    assert tensor.dtype == torch.float32
    assert tensor[0, 2, 0, 0].item() == pytest.approx(1.0)


def test_camera_rgb_capture_validates_sensor_output() -> None:
    with pytest.raises(KeyError, match="missing"):
        get_camera_rgb(_env({"rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)}), "missing")
    with pytest.raises(KeyError, match="has no RGB output"):
        get_camera_rgb(_env({}), "ee_camera")
    with pytest.raises(ValueError, match="HWC or BHWC"):
        get_camera_rgb(_env({"rgb": torch.zeros((1, 3, 2, 2), dtype=torch.uint8)}), "ee_camera")


def test_recording_capture_uses_canonical_rgb_conversion() -> None:
    rgba = torch.tensor([[[[0.0, 0.5, 1.0, 0.25]]]], dtype=torch.float32)

    frames = capture_camera_frames(_env({"rgb": rgba}), 0, ["ee_camera"])

    np.testing.assert_array_equal(frames["ee_camera"], [[[0, 127, 255]]])


def test_prepare_rgb_tensor_validates_dtype_and_shape() -> None:
    with pytest.raises(TypeError, match="uint8"):
        prepare_rgb_tensor(torch.zeros((1, 2, 2, 3)))
    with pytest.raises(ValueError, match="HWC or BHWC"):
        prepare_rgb_tensor(torch.zeros((1, 3, 2, 2), dtype=torch.uint8))
