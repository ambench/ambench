# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate a converted UMI zarr dataset for Diffusion Policy training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

EE_REQUIRED_SHAPES = {
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (3,),
    "robot0_gripper_width": (1,),
    "robot0_demo_start_pose": (6,),
    "robot0_demo_end_pose": (6,),
    "camera0_rgb": None,
    "action": (7,),
}
BASE_JOINT_REQUIRED_SHAPES = {
    "robot0_base_pos": (3,),
    "robot0_base_rot_axis_angle": (3,),
    "robot0_joint_pos": (4,),
    "robot0_gripper_width": (1,),
    "robot0_base_demo_start_pose": (6,),
    "robot0_base_demo_end_pose": (6,),
    "camera0_rgb": None,
    "action": (11,),
}
OPTIONAL_SHAPES = {
    "camera1_rgb": None,
    "robot0_joint_pos": (4,),
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (3,),
    "robot0_demo_start_pose": (6,),
    "robot0_demo_end_pose": (6,),
    "robot0_base_pos": (3,),
    "robot0_base_rot_axis_angle": (3,),
    "robot0_base_demo_start_pose": (6,),
    "robot0_base_demo_end_pose": (6,),
}


def _open_store(path: Path):
    if path.is_file() and path.suffix == ".zip":
        return zarr.ZipStore(str(path), mode="r")
    if path.is_dir():
        return zarr.DirectoryStore(str(path))
    raise FileNotFoundError(f"Expected a .zarr.zip file or zarr directory: {path}")


def _shape_error(
    key: str,
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...] | None,
    image_size: int | None,
) -> str | None:
    if key.endswith("_rgb"):
        if len(actual_shape) != 3 or actual_shape[2] != 3 or min(actual_shape[:2]) < 1:
            return f"{key} must have image shape (height, width, 3), got {actual_shape}."
        if image_size is not None and actual_shape != (image_size, image_size, 3):
            return f"{key} must have shape ({image_size}, {image_size}, 3), got {actual_shape}."
        return None
    if actual_shape != expected_shape:
        return f"{key} must have shape {expected_shape}, got {actual_shape}."
    return None


def validate_dataset(
    zarr_path: str | Path,
    *,
    image_size: int | None = None,
    verbose: bool = True,
) -> bool:
    """Return whether a zipped or directory UMI dataset satisfies the supported schema."""
    path = Path(zarr_path).expanduser().resolve()
    errors: list[str] = []
    dataset_mode = "unknown"
    num_steps = 0
    num_episodes = 0
    store = None

    try:
        store = _open_store(path)
        root = zarr.open_group(store=store, mode="r")
        if "data" not in root or "meta" not in root:
            errors.append("Root must contain both data and meta groups.")
        else:
            data = root["data"]
            meta = root["meta"]
            if "episode_ends" not in meta:
                errors.append("meta/episode_ends is missing.")
            else:
                episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
                num_episodes = len(episode_ends)
                if num_episodes == 0:
                    errors.append("Dataset contains no episodes.")
                elif np.any(np.diff(episode_ends) <= 0):
                    errors.append("meta/episode_ends must be strictly increasing.")

            if "action" not in data:
                errors.append("data/action is missing.")
                required_shapes = {}
            else:
                action_shape = tuple(data["action"].shape[1:])
                num_steps = int(data["action"].shape[0])
                if action_shape == (7,):
                    dataset_mode = "ee"
                    required_shapes = EE_REQUIRED_SHAPES
                elif action_shape == (11,):
                    dataset_mode = "base_joint"
                    required_shapes = BASE_JOINT_REQUIRED_SHAPES
                else:
                    errors.append(f"Unsupported action shape {action_shape}; expected (7,) or (11,).")
                    required_shapes = {}

            if num_episodes and num_steps and episode_ends[-1] != num_steps:
                errors.append(
                    f"Last episode end is {int(episode_ends[-1])}, but data/action contains {num_steps} steps."
                )

            for key, array in data.items():
                if array.shape[0] != num_steps:
                    errors.append(f"data/{key} has {array.shape[0]} steps; expected {num_steps}.")

            for key, expected_shape in required_shapes.items():
                if key not in data:
                    errors.append(f"Required data/{key} is missing.")
                    continue
                shape_error = _shape_error(key, tuple(data[key].shape[1:]), expected_shape, image_size)
                if shape_error is not None:
                    errors.append(shape_error)

            for key, expected_shape in OPTIONAL_SHAPES.items():
                if key not in data or key in required_shapes:
                    continue
                shape_error = _shape_error(key, tuple(data[key].shape[1:]), expected_shape, image_size)
                if shape_error is not None:
                    errors.append(shape_error)

            if num_steps:
                for key in required_shapes:
                    if key not in data:
                        continue
                    try:
                        samples = (np.asarray(data[key][0]), np.asarray(data[key][-1]))
                    except Exception as exc:
                        errors.append(f"Could not read boundary samples from data/{key}: {exc}")
                        continue
                    if not key.endswith("_rgb") and any(
                        np.issubdtype(sample.dtype, np.number) and not np.all(np.isfinite(sample)) for sample in samples
                    ):
                        errors.append(f"data/{key} contains non-finite boundary samples.")
    except Exception as exc:
        errors.append(f"Could not load dataset: {type(exc).__name__}: {exc}")
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()

    if verbose:
        print(f"Dataset: {path}")
        print(f"Mode: {dataset_mode}")
        print(f"Episodes: {num_episodes}")
        print(f"Steps: {num_steps}")
        if errors:
            print("Validation failed:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("Validation passed. Dataset is ready for UMI Diffusion Policy training.")
    return not errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to a .zarr.zip file or zarr directory.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Optionally require square camera images of this size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.image_size is not None and args.image_size < 1:
        raise ValueError("--image_size must be positive when provided.")
    return 0 if validate_dataset(args.zarr_path, image_size=args.image_size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
