# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ambench_learn.data.action_semantics import BASE_JOINT_ABSOLUTE, EE_ABSOLUTE

UMI_ROOT = (
    Path(__file__).resolve().parents[2] / "ambench_learn" / "policies" / "dp" / "universal_manipulation_interface"
)
sys.path.insert(0, str(UMI_ROOT))

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "scripts" / "data" / "dp" / "lerobot_to_zarr.py"
SPEC = importlib.util.spec_from_file_location("ambench_lerobot_to_zarr", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
LEROBOT_TO_ZARR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEROBOT_TO_ZARR)
rows_to_episode_data = LEROBOT_TO_ZARR.rows_to_episode_data


def _image(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


def test_dp_zarr_ee_layout_preserves_measured_state_and_absolute_targets(tmp_path: Path) -> None:
    info = {
        "ambench": {
            "action_semantics": EE_ABSOLUTE,
            "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
        }
    }
    rows = [
        {
            "observation.state": [1.0 + index, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.04],
            "action": [4.0 + index, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, -0.5],
            "observation.images.ee_camera": _image(index),
        }
        for index in range(2)
    ]

    episode = rows_to_episode_data(
        rows,
        dataset_root=tmp_path,
        info=info,
        image_size=3,
        ee_image_key="observation.images.ee_camera",
        base_image_key="observation.images.base_camera",
        state_key="observation.state",
        action_key="action",
        include_base_image=False,
        fill_missing_base_state=True,
    )

    assert episode["camera0_rgb"].shape == (2, 3, 3, 3)
    assert episode["robot0_eef_pos"].shape == (2, 3)
    assert episode["robot0_eef_rot_axis_angle"].shape == (2, 3)
    assert episode["action"].shape == (2, 7)
    np.testing.assert_array_equal(episode["action"][:, :3], [[4.0, 5.0, 6.0], [5.0, 5.0, 6.0]])
    np.testing.assert_array_equal(episode["robot0_base_pos"], np.zeros((2, 3), dtype=np.float32))


def test_dp_zarr_base_joint_layout_preserves_joint_state_and_targets(tmp_path: Path) -> None:
    info = {
        "ambench": {
            "action_semantics": BASE_JOINT_ABSOLUTE,
            "state_keys": ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"],
        }
    }
    rows = [
        {
            "observation.state": [
                1.0 + index,
                2.0,
                3.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.04,
            ],
            "action": [
                4.0 + index,
                5.0,
                6.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.6,
                0.7,
                0.8,
                -0.5,
            ],
            "observation.images.ee_camera": _image(index),
        }
        for index in range(2)
    ]

    episode = rows_to_episode_data(
        rows,
        dataset_root=tmp_path,
        info=info,
        image_size=3,
        ee_image_key="observation.images.ee_camera",
        base_image_key="observation.images.base_camera",
        state_key="observation.state",
        action_key="action",
        include_base_image=False,
        fill_missing_base_state=False,
    )

    assert episode["camera0_rgb"].shape == (2, 3, 3, 3)
    assert episode["robot0_base_pos"].shape == (2, 3)
    assert episode["robot0_base_rot_axis_angle"].shape == (2, 3)
    assert episode["robot0_joint_pos"].shape == (2, 4)
    assert episode["action"].shape == (2, 11)
    np.testing.assert_allclose(episode["action"][:, 6:10], [[0.5, 0.6, 0.7, 0.8]] * 2)
