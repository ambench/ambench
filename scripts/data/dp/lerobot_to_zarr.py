# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert canonical AM-Bench LeRobot recordings to UMI zarr format."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ambench_learn.data.action_semantics import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
    dataset_metadata,
)

# Datasets recorded before delta actions were removed still carry this marker.
LEGACY_EE_DELTA = "ee_delta"

register_codecs()

DEFAULT_EE_IMAGE_KEY = "observation.images.ee_camera"
DEFAULT_BASE_IMAGE_KEY = "observation.images.base_camera"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"

STATE_KEY_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "gripper_width": 1,
    "arm_joint_pos": 4,
    "base_pos": 3,
    "base_quat": 4,
}


def resolve_dataset_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser().resolve()
    if (dataset_root / "meta" / "info.json").is_file():
        return dataset_root

    canonical_root = dataset_root / "lerobot"
    if (canonical_root / "meta" / "info.json").is_file():
        return canonical_root

    raise FileNotFoundError(
        f"Could not find a LeRobot dataset under '{dataset_root}'. "
        "Expected either 'meta/info.json' directly or a 'lerobot/meta/info.json' child."
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No canonical LeRobot episode tables under {dataset_root / 'meta' / 'episodes'}")

    episodes: list[dict[str, Any]] = []
    for episode_file in episode_files:
        episodes.extend(pq.read_table(episode_file).to_pylist())
    return sorted(episodes, key=lambda episode: int(episode["episode_index"]))


def source_data_file(dataset_root: Path, info: dict[str, Any], episode: dict[str, Any]) -> Path:
    data_path = str(info.get("data_path", ""))
    if not data_path:
        raise ValueError(f"Source dataset info.json is missing data_path: {dataset_root}")

    chunk_index = int(episode.get("data/chunk_index", episode.get("meta/episodes/chunk_index", 0)))
    file_index = int(episode.get("data/file_index", episode.get("meta/episodes/file_index", episode["episode_index"])))
    episode_index = int(episode["episode_index"])
    relative_path = data_path.format(
        chunk_index=chunk_index,
        file_index=file_index,
        episode_chunk=chunk_index,
        episode_index=episode_index,
    )
    file_path = dataset_root / relative_path
    if not file_path.is_file():
        raise FileNotFoundError(f"Episode parquet file does not exist: {file_path}")
    return file_path


def load_episode_rows(
    dataset_root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    *,
    columns: list[str],
) -> list[dict[str, Any]]:
    """Load only the frames belonging to one episode from its Parquet shard."""
    episode_index = int(episode["episode_index"])
    table = pq.read_table(
        source_data_file(dataset_root, info, episode),
        columns=columns,
        filters=[("episode_index", "=", episode_index)],
    )
    if table.num_rows == 0:
        raise ValueError(f"Episode {episode_index} has no frames in its referenced Parquet file.")

    expected_length = episode.get("length")
    if expected_length is None:
        start_index = episode.get("dataset_from_index")
        end_index = episode.get("dataset_to_index")
        if start_index is not None and end_index is not None:
            expected_length = int(end_index) - int(start_index)
    if expected_length is not None and table.num_rows != int(expected_length):
        raise ValueError(
            f"Episode {episode_index} metadata reports {int(expected_length)} frames, "
            f"but its Parquet rows contain {table.num_rows}."
        )
    return table.to_pylist()


def validate_source_info(
    dataset_root: Path,
    info: dict[str, Any],
    *,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    include_base_image: bool,
) -> None:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Source info.json has no feature dictionary: {dataset_root}")

    for key in (ee_image_key, state_key, action_key):
        if key not in features:
            raise KeyError(f"Source dataset {dataset_root} is missing required feature '{key}'.")

    if include_base_image and base_image_key not in features:
        raise KeyError(
            f"Source dataset {dataset_root} is missing requested base image feature '{base_image_key}'. "
            "Pass --omit_base_image to export EE camera only."
        )

    action_semantics = dataset_metadata(info).get("action_semantics")
    if action_semantics not in (None, LEGACY_EE_DELTA, EE_ABSOLUTE, BASE_JOINT_ABSOLUTE):
        raise ValueError(
            "The UMI DP bridge currently supports ee_absolute and base_joint_absolute datasets. "
            f"Dataset {dataset_root} reports action_semantics={action_semantics!r}."
        )
    if action_semantics == LEGACY_EE_DELTA:
        raise ValueError(
            "Refusing to export an ee_delta dataset for the standard UMI DP path. "
            "Delta actions are no longer produced; re-record this dataset against an absolute environment."
        )
    action_shape = tuple(features[action_key].get("shape", ()))
    if action_semantics in (None, EE_ABSOLUTE) and action_shape not in {(7,), (8,)}:
        raise ValueError(f"Expected 7D rotvec or 8D quaternion EE actions. Got {action_shape}.")
    if action_semantics == BASE_JOINT_ABSOLUTE and action_shape != (12,):
        raise ValueError(f"Expected 12D Base+joints actions. Got {action_shape}.")


def resolve_state_keys(info: dict[str, Any], state_dim: int) -> list[str]:
    state_keys = dataset_metadata(info).get("state_keys")
    if state_keys is None:
        if state_dim == 8:
            return ["ee_pos", "ee_quat", "gripper_width"]
        if state_dim == 19:
            return ["ee_pos", "ee_quat", "gripper_width", "arm_joint_pos", "base_pos", "base_quat"]
        raise ValueError(
            "Cannot infer observation.state layout because meta/info.json lacks "
            f"ambench.state_keys and state_dim={state_dim} is not a known default."
        )

    if not isinstance(state_keys, list) or not all(isinstance(key, str) for key in state_keys):
        raise ValueError(f"Expected ambench.state_keys to be a list of strings. Got {state_keys!r}.")

    expected_dim = sum(STATE_KEY_DIMS.get(key, 0) for key in state_keys)
    unknown_keys = [key for key in state_keys if key not in STATE_KEY_DIMS]
    if unknown_keys:
        raise ValueError(f"Unsupported observation.state keys for DP zarr export: {unknown_keys}")
    if expected_dim != state_dim:
        raise ValueError(f"state_keys imply dimension {expected_dim}, but observation.state has dimension {state_dim}.")
    return state_keys


def split_state(state: np.ndarray, state_keys: list[str]) -> dict[str, np.ndarray]:
    parts: dict[str, np.ndarray] = {}
    offset = 0
    for key in state_keys:
        width = STATE_KEY_DIMS[key]
        parts[key] = state[:, offset : offset + width]
        offset += width
    return parts


def quaternion_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    quat_wxyz = quat_wxyz / np.maximum(norm, 1.0e-9)
    return R.from_quat(quat_wxyz, scalar_first=True).as_rotvec().astype(np.float32, copy=False)


def decode_image(value: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            return np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
        image_path = value.get("path")
        if image_path is not None:
            return np.asarray(Image.open(dataset_root / image_path).convert("RGB"))
    if isinstance(value, (bytes, bytearray)):
        return np.asarray(Image.open(BytesIO(value)).convert("RGB"))
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return value.astype(np.uint8, copy=False)
        if value.ndim == 1:
            return np.asarray(Image.open(BytesIO(value.tobytes())).convert("RGB"))
    raise TypeError(f"Unsupported image value type: {type(value)}")


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.shape[:2] == (image_size, image_size):
        return image.astype(np.uint8, copy=False)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


def rows_to_episode_data(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    info: dict[str, Any],
    image_size: int,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    include_base_image: bool,
    fill_missing_base_state: bool,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Cannot convert an empty episode.")

    state = np.asarray([row[state_key] for row in rows], dtype=np.float32)
    if state.ndim != 2:
        raise ValueError(f"Expected {state_key} to have shape (T, D). Got {state.shape}.")
    state_parts = split_state(state, resolve_state_keys(info, state.shape[1]))
    action_semantics = dataset_metadata(info).get("action_semantics")

    if action_semantics == BASE_JOINT_ABSOLUTE:
        required_state_keys = {"base_pos", "base_quat", "arm_joint_pos", "gripper_width"}
        if not required_state_keys.issubset(state_parts):
            raise ValueError(f"{state_key} must contain {sorted(required_state_keys)} for Base+joints UMI DP export.")

        raw_action = np.asarray([row[action_key] for row in rows], dtype=np.float32)
        if raw_action.ndim != 2 or raw_action.shape[1] != 12:
            raise ValueError(f"Expected Base+joints action shape (T, 12). Got {raw_action.shape}.")
        action = np.concatenate(
            [
                raw_action[:, :3],
                quaternion_to_axis_angle(raw_action[:, 3:7]),
                raw_action[:, 7:11],
                raw_action[:, 11:12],
            ],
            axis=1,
        )

        episode_data = {
            "robot0_base_pos": state_parts["base_pos"].astype(np.float32, copy=False),
            "robot0_base_rot_axis_angle": quaternion_to_axis_angle(state_parts["base_quat"]),
            "robot0_joint_pos": state_parts["arm_joint_pos"].astype(np.float32, copy=False),
            "robot0_gripper_width": state_parts["gripper_width"].astype(np.float32, copy=False),
            "camera0_rgb": np.stack(
                [resize_image(decode_image(row[ee_image_key], dataset_root), image_size) for row in rows],
                axis=0,
            ),
            "action": action.astype(np.float32, copy=False),
        }
        if include_base_image:
            episode_data["camera1_rgb"] = np.stack(
                [resize_image(decode_image(row[base_image_key], dataset_root), image_size) for row in rows],
                axis=0,
            )
        return episode_data

    if not {"ee_pos", "ee_quat", "gripper_width"}.issubset(state_parts):
        raise ValueError(f"{state_key} must contain ee_pos, ee_quat, and gripper_width for UMI DP export.")

    raw_action = np.asarray([row[action_key] for row in rows], dtype=np.float32)
    if raw_action.ndim != 2 or raw_action.shape[1] not in (7, 8):
        raise ValueError(f"Expected action shape (T, 7) or (T, 8). Got {raw_action.shape}.")
    if raw_action.shape[1] == 8:
        action = np.concatenate(
            [raw_action[:, :3], quaternion_to_axis_angle(raw_action[:, 3:7]), raw_action[:, 7:8]],
            axis=1,
        )
    else:
        action = raw_action

    episode_data = {
        "robot0_eef_pos": state_parts["ee_pos"].astype(np.float32, copy=False),
        "robot0_eef_rot_axis_angle": quaternion_to_axis_angle(state_parts["ee_quat"]),
        "robot0_gripper_width": state_parts["gripper_width"].astype(np.float32, copy=False),
        "camera0_rgb": np.stack(
            [resize_image(decode_image(row[ee_image_key], dataset_root), image_size) for row in rows],
            axis=0,
        ),
        "action": action.astype(np.float32, copy=False),
    }

    if include_base_image:
        episode_data["camera1_rgb"] = np.stack(
            [resize_image(decode_image(row[base_image_key], dataset_root), image_size) for row in rows],
            axis=0,
        )
    if "arm_joint_pos" in state_parts:
        episode_data["robot0_joint_pos"] = state_parts["arm_joint_pos"].astype(np.float32, copy=False)
    if "base_pos" in state_parts and "base_quat" in state_parts:
        episode_data["robot0_base_pos"] = state_parts["base_pos"].astype(np.float32, copy=False)
        episode_data["robot0_base_rot_axis_angle"] = quaternion_to_axis_angle(state_parts["base_quat"])
    elif fill_missing_base_state:
        episode_length = state.shape[0]
        episode_data["robot0_base_pos"] = np.zeros((episode_length, 3), dtype=np.float32)
        episode_data["robot0_base_rot_axis_angle"] = np.zeros((episode_length, 3), dtype=np.float32)
    return episode_data


def add_demo_pose_fields(episode_data: dict[str, np.ndarray]) -> None:
    if "robot0_eef_pos" in episode_data:
        episode_length = episode_data["robot0_eef_pos"].shape[0]
        start_pose = np.concatenate(
            [episode_data["robot0_eef_pos"][0:1], episode_data["robot0_eef_rot_axis_angle"][0:1]], axis=1
        )
        end_pose = np.concatenate(
            [episode_data["robot0_eef_pos"][-1:], episode_data["robot0_eef_rot_axis_angle"][-1:]], axis=1
        )
        episode_data["robot0_demo_start_pose"] = np.tile(start_pose, (episode_length, 1)).astype(np.float32, copy=False)
        episode_data["robot0_demo_end_pose"] = np.tile(end_pose, (episode_length, 1)).astype(np.float32, copy=False)

    if "robot0_base_pos" not in episode_data:
        return

    episode_length = episode_data["robot0_base_pos"].shape[0]
    base_start_pose = np.concatenate(
        [episode_data["robot0_base_pos"][0:1], episode_data["robot0_base_rot_axis_angle"][0:1]], axis=1
    )
    base_end_pose = np.concatenate(
        [episode_data["robot0_base_pos"][-1:], episode_data["robot0_base_rot_axis_angle"][-1:]], axis=1
    )
    episode_data["robot0_base_demo_start_pose"] = np.tile(base_start_pose, (episode_length, 1)).astype(
        np.float32, copy=False
    )
    episode_data["robot0_base_demo_end_pose"] = np.tile(base_end_pose, (episode_length, 1)).astype(
        np.float32, copy=False
    )


def zarr_chunks(episode_data: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    chunks = {key: value.shape for key, value in episode_data.items() if not key.endswith("_rgb")}
    for key, value in episode_data.items():
        if key.endswith("_rgb"):
            chunks[key] = (1,) + value.shape[1:]
    return chunks


def zarr_compressors(episode_data: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: Jpeg2k(level=50) if key.endswith("_rgb") else None for key in episode_data}


def convert_dataset(
    *,
    input_path: Path,
    output_path: Path,
    image_size: int,
    ee_image_key: str,
    base_image_key: str,
    state_key: str,
    action_key: str,
    include_base_image: bool,
    fill_missing_base_state: bool,
    overwrite: bool,
    max_episodes: int | None,
) -> None:
    dataset_root = resolve_dataset_root(input_path)
    info = load_json(dataset_root / "meta" / "info.json")
    validate_source_info(
        dataset_root,
        info,
        ee_image_key=ee_image_key,
        base_image_key=base_image_key,
        state_key=state_key,
        action_key=action_key,
        include_base_image=include_base_image,
    )

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(dataset_root)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    columns = [state_key, action_key, ee_image_key]
    if include_base_image:
        columns.append(base_image_key)

    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}-", dir=output_path.parent) as temp_dir:
        temporary_root = Path(temp_dir)
        working_zarr_path = temporary_root / "dataset.zarr"
        working_store = zarr.DirectoryStore(str(working_zarr_path))
        try:
            replay_buffer = ReplayBuffer.create_empty_zarr(storage=working_store)
            for episode in tqdm(episodes, desc="converting episodes"):
                rows = load_episode_rows(dataset_root, info, episode, columns=columns)
                episode_data = rows_to_episode_data(
                    rows,
                    dataset_root=dataset_root,
                    info=info,
                    image_size=image_size,
                    ee_image_key=ee_image_key,
                    base_image_key=base_image_key,
                    state_key=state_key,
                    action_key=action_key,
                    include_base_image=include_base_image,
                    fill_missing_base_state=fill_missing_base_state,
                )
                add_demo_pose_fields(episode_data)
                replay_buffer.add_episode(
                    data=episode_data,
                    chunks=zarr_chunks(episode_data),
                    compressors=zarr_compressors(episode_data),
                )

            if replay_buffer.n_episodes == 0:
                raise ValueError("No episodes were converted.")
            action_dim = replay_buffer["action"].shape[-1]
            if action_dim not in (7, 11):
                raise ValueError(f"Expected 7D EE or 11D Base+joints UMI action, got {replay_buffer['action'].shape}.")
            converted_episodes = replay_buffer.n_episodes
            converted_steps = replay_buffer.n_steps
        finally:
            working_store.close()

        if output_path.suffix == ".zip":
            temporary_output = temporary_root / output_path.name
            source_store = zarr.DirectoryStore(str(working_zarr_path))
            try:
                with zarr.ZipStore(str(temporary_output), mode="w") as zip_store:
                    zarr.copy_store(source_store, zip_store, if_exists="replace")
            finally:
                source_store.close()
        else:
            temporary_output = working_zarr_path

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        temporary_output.replace(output_path)

    print(f"Converted {converted_episodes} episodes / {converted_steps} steps.")
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert canonical AM-Bench LeRobot recordings to UMI zarr.")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="LeRobot root or session root containing lerobot/.",
    )
    parser.add_argument("--output_path", type=Path, default=None, help="Output .zarr.zip path.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--ee_image_key", type=str, default=DEFAULT_EE_IMAGE_KEY)
    parser.add_argument("--base_image_key", type=str, default=DEFAULT_BASE_IMAGE_KEY)
    parser.add_argument("--state_key", type=str, default=DEFAULT_STATE_KEY)
    parser.add_argument("--action_key", type=str, default=DEFAULT_ACTION_KEY)
    parser.add_argument("--omit_base_image", action="store_true", help="Export only camera0_rgb from the EE camera.")
    parser.add_argument(
        "--no_fill_missing_base_state",
        action="store_true",
        help=(
            "Do not synthesize zero-valued base pose arrays when observation.state lacks base_pos/base_quat. "
            "The default keeps converted EE datasets compatible with umi_drone_ee_pos, where base pose is ignored."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size < 1:
        raise ValueError("--image_size must be positive.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max_episodes must be >= 1 when set.")

    output_path = args.output_path
    if output_path is None:
        output_path = args.input_path.with_suffix(".zarr.zip")

    convert_dataset(
        input_path=args.input_path,
        output_path=output_path,
        image_size=args.image_size,
        ee_image_key=args.ee_image_key,
        base_image_key=args.base_image_key,
        state_key=args.state_key,
        action_key=args.action_key,
        include_base_image=not args.omit_base_image,
        fill_missing_base_state=not args.no_fill_missing_base_state,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
