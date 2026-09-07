# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Export canonical AM-Bench data into the pinned OpenPI LeRobot v2.1 schema."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from ambench_learn.data.action_resampling import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
    compute_stride,
    resolve_dataset_action_semantics,
)
from ambench_learn.data.action_semantics import dataset_metadata
from ambench_learn.data.resampled_lerobot_dataset import build_resampled_episodes

DEFAULT_REPO_ID = "am_bench/openpi_export"
DEFAULT_EE_IMAGE_KEY = "observation.images.ee_camera"
DEFAULT_BASE_IMAGE_KEY = "observation.images.base_camera"
OPENPI_LEROBOT_VERSION = "v2.1"
OPENPI_CHUNK_SIZE = 1000
OPENPI_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
TASK_PROMPT_MAP_KEYS = ("task_prompt_map", "task_prompts", "prompts")
STATE_KEY_DIMS = {
    "ee_pos": 3,
    "ee_quat": 4,
    "gripper_width": 1,
    "arm_joint_pos": 4,
    "base_pos": 3,
    "base_quat": 4,
}


def resolve_dataset_root(dataset_root: Path) -> Path:
    """Resolve either a direct LeRobot root or a recorder session root."""
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


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_task_prompt_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task prompt map must be a JSON object: {path}")
    for key in TASK_PROMPT_MAP_KEYS:
        nested = raw.get(key)
        if isinstance(nested, dict):
            raw = nested
            break

    prompt_map: dict[str, str] = {}
    for source_task, prompt in raw.items():
        if not isinstance(source_task, str) or not isinstance(prompt, str):
            raise ValueError(f"Task prompt map entries must be strings. Got {source_task!r}: {prompt!r} in {path}.")
        source_task = source_task.strip()
        prompt = prompt.strip()
        if not source_task or not prompt:
            raise ValueError(f"Task prompt map contains an empty key or prompt in {path}.")
        prompt_map[source_task] = prompt
    return prompt_map


def task_from_index(dataset: LeRobotDataset, task_index: int) -> str | None:
    tasks = getattr(dataset.meta, "tasks", None)
    if tasks is None:
        return None
    if hasattr(tasks, "columns") and "task_index" in tasks.columns:
        matches = tasks[tasks["task_index"] == task_index]
        if len(matches) > 0 and "task" in matches.columns:
            return str(matches.iloc[0]["task"])
    values = getattr(tasks, "values", None)
    if callable(values):
        values = values()
    if values is not None:
        values = list(values)
        if 0 <= task_index < len(values):
            return str(values[task_index])
    try:
        return str(tasks[task_index])
    except (IndexError, KeyError, TypeError):
        return None


def resolve_task_prompt(
    item: dict[str, Any],
    dataset: LeRobotDataset,
    fallback: str,
    prompt_map: dict[str, str],
    require_prompt_map: bool,
) -> str:
    if fallback:
        return fallback

    # Recover the source task from either frame text or LeRobot metadata.
    source_task = None
    if "task" in item:
        value = item["task"]
        array = to_numpy(value)
        if array.size == 1:
            value = array.reshape(()).item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        source_task = str(value)
    elif "task_index" in item:
        task_index = int(to_numpy(item["task_index"]).reshape(()).item())
        source_task = task_from_index(dataset, task_index)

    if source_task is not None:
        if source_task in prompt_map:
            return prompt_map[source_task]
        if require_prompt_map:
            raise KeyError(
                f"Missing language instruction for source task '{source_task}'. "
                "Add it to --task_prompt_map or remove --require_task_prompt_map."
            )
        return source_task
    if require_prompt_map:
        raise KeyError("Cannot require task prompt mapping because this frame has no task or task_index.")
    return "am_bench task"


def image_from_item(item: dict[str, Any], key: str) -> np.ndarray:
    if key not in item:
        raise KeyError(f"Missing required image key '{key}'. Available keys: {sorted(item)}")
    image = to_numpy(item[key])
    if image.ndim != 3:
        raise ValueError(f"Expected image '{key}' to have three dimensions. Got {image.shape}.")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = image * scale
    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def resolve_state_keys(info: dict[str, Any], state_dim: int, action_semantics: str) -> list[str]:
    state_keys = dataset_metadata(info).get("state_keys")
    if state_keys is None:
        if action_semantics == EE_ABSOLUTE and state_dim == 8:
            return ["ee_pos", "ee_quat", "gripper_width"]
        if action_semantics == BASE_JOINT_ABSOLUTE and state_dim == 12:
            return ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]
        if state_dim == 19:
            return ["ee_pos", "ee_quat", "gripper_width", "arm_joint_pos", "base_pos", "base_quat"]
        raise ValueError(
            "Cannot infer observation.state layout because metadata lacks ambench.state_keys "
            f"and state_dim={state_dim} is not a known layout."
        )
    if not isinstance(state_keys, list) or not all(isinstance(key, str) for key in state_keys):
        raise ValueError(f"Expected ambench.state_keys to be a list of strings. Got {state_keys!r}.")
    unknown = [key for key in state_keys if key not in STATE_KEY_DIMS]
    if unknown:
        raise ValueError(f"Unsupported observation.state keys for OpenPI export: {unknown}")
    expected_dim = sum(STATE_KEY_DIMS[key] for key in state_keys)
    if expected_dim != state_dim:
        raise ValueError(f"state_keys imply dimension {expected_dim}, but observation.state has dimension {state_dim}.")
    return state_keys


def openpi_frame(
    item: dict[str, Any],
    *,
    dataset: LeRobotDataset,
    info: dict[str, Any],
    action_semantics: str,
    ee_image_key: str,
    base_image_key: str,
    include_base_image: bool,
    task_prompt: str,
    task_prompt_map: dict[str, str],
    require_task_prompt_map: bool,
) -> dict[str, Any]:
    state = to_numpy(item["observation.state"]).astype(np.float32, copy=False).reshape(-1)
    action = to_numpy(item["action"]).astype(np.float32, copy=False).reshape(-1)

    # Split the canonical state vector using its declared feature layout.
    state_parts: dict[str, np.ndarray] = {}
    offset = 0
    for key in resolve_state_keys(info, state.size, action_semantics):
        width = STATE_KEY_DIMS[key]
        state_parts[key] = state[offset : offset + width]
        offset += width

    frame: dict[str, Any] = {
        "ee_image": image_from_item(item, ee_image_key),
        "actions": action,
        "task": resolve_task_prompt(
            item,
            dataset,
            task_prompt,
            task_prompt_map,
            require_task_prompt_map,
        ),
    }
    if action_semantics == EE_ABSOLUTE:
        required = {"ee_pos", "ee_quat", "gripper_width"}
        if action.size != 8 or not required.issubset(state_parts):
            raise ValueError(
                "OpenPI EE export requires an 8D absolute action and ee_pos, ee_quat, gripper_width state. "
                f"Got action shape {action.shape} and state keys {list(state_parts)}."
            )
        frame.update({key: state_parts[key].astype(np.float32, copy=False) for key in required})
    elif action_semantics == BASE_JOINT_ABSOLUTE:
        required = {"base_pos", "base_quat", "arm_joint_pos", "gripper_width"}
        if action.size != 12 or not required.issubset(state_parts):
            raise ValueError(
                "OpenPI BaseJoint export requires a 12D absolute action and base_pos, base_quat, arm_joint_pos,"
                f" gripper_width state. Got action shape {action.shape} and state keys {list(state_parts)}."
            )
        frame.update({key: state_parts[key].astype(np.float32, copy=False) for key in required})
    else:
        raise ValueError(
            "OpenPI export supports canonical ee_absolute and base_joint_absolute datasets only. "
            f"Got action semantics {action_semantics!r}."
        )
    if include_base_image:
        frame["base_image"] = image_from_item(item, base_image_key)
    return frame


def build_features(sample_frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for key, value in sample_frame.items():
        if key == "task":
            continue
        array = np.asarray(value)
        if key.endswith("_image"):
            height, width, channels = array.shape
            features[key] = {
                "dtype": "image",
                "shape": (channels, height, width),
                "names": ["channel", "height", "width"],
            }
        else:
            features[key] = {
                "dtype": "float32",
                "shape": tuple(array.shape),
                "names": [f"{key}_{index}" for index in range(array.size)],
            }
    return features


def build_openpi_v21_features(features: dict[str, dict[str, Any]]) -> Features:
    """Build the Hugging Face schema used by the pinned OpenPI LeRobot reader."""
    hf_features: dict[str, Any] = {}
    for key, feature in features.items():
        shape = tuple(feature["shape"])
        if feature["dtype"] == "image":
            hf_features[key] = Image()
        elif shape == (1,):
            hf_features[key] = Value(feature["dtype"])
        elif len(shape) == 1:
            hf_features[key] = Sequence(Value(feature["dtype"]), length=shape[0])
        else:
            raise ValueError(f"OpenPI export supports scalar and vector non-image features. Got {key}: {feature}.")
    return Features(hf_features)


def compute_episode_stats(columns: dict[str, list[Any]], features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute v2.1-compatible per-episode statistics."""
    stats: dict[str, Any] = {}
    for key, feature in features.items():
        values = columns[key]
        if feature["dtype"] == "image":
            array = np.stack([np.moveaxis(np.asarray(value), -1, 0) for value in values]).astype(np.float64)
            array /= 255.0
            axes: int | tuple[int, ...] = (0, 2, 3)
            keepdims = True
        else:
            array = np.asarray(values)
            axes = 0
            keepdims = array.ndim == 1
        feature_stats = {
            "min": np.min(array, axis=axes, keepdims=keepdims),
            "max": np.max(array, axis=axes, keepdims=keepdims),
            "mean": np.mean(array, axis=axes, keepdims=keepdims),
            "std": np.std(array, axis=axes, keepdims=keepdims),
            "count": np.asarray([len(array)]),
        }
        if feature["dtype"] == "image":
            feature_stats = {
                name: value if name == "count" else np.squeeze(value, axis=0) for name, value in feature_stats.items()
            }
        stats[key] = {name: value.tolist() for name, value in feature_stats.items()}
    return stats


def write_openpi_v21_parquet(columns: dict[str, list[Any]], features: Features, path: Path) -> None:
    """Write parquet with feature metadata understood by OpenPI's datasets release."""
    table = Dataset.from_dict(columns, features=features, split="train").data.table
    metadata = dict(table.schema.metadata or {})
    huggingface_metadata = json.loads(metadata[b"huggingface"])
    for feature in huggingface_metadata["info"]["features"].values():
        if feature.get("_type") == "List":
            feature["_type"] = "Sequence"
    metadata[b"huggingface"] = json.dumps(huggingface_metadata).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path)


class OpenPIV21Writer:
    """Write the local LeRobot layout consumed by the pinned OpenPI checkout."""

    def __init__(
        self,
        root: Path,
        *,
        fps: int,
        features: dict[str, dict[str, Any]],
        export_metadata: dict[str, Any],
    ) -> None:
        self.root = root
        self.fps = fps
        self.features = {
            **features,
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        }
        self.hf_features = build_openpi_v21_features(self.features)
        self.export_metadata = export_metadata
        self.tasks: dict[str, int] = {}
        self.episode_count = 0
        self.frame_count = 0
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "meta").mkdir()

    def save_episode(self, frames: list[dict[str, Any]]) -> None:
        """Write one complete episode and append its metadata records."""
        if not frames:
            return
        episode_index = self.episode_count
        columns = {key: [] for key in self.features}
        episode_tasks: list[str] = []
        for frame_index, frame in enumerate(frames):
            task = str(frame["task"])
            if task not in self.tasks:
                self.tasks[task] = len(self.tasks)
                self._append_jsonl(
                    self.root / "meta" / "tasks.jsonl",
                    {"task_index": self.tasks[task], "task": task},
                )
            if task not in episode_tasks:
                episode_tasks.append(task)
            for key in self.features:
                if key in frame:
                    value = frame[key]
                    if tuple(self.features[key]["shape"]) == (1,):
                        value = np.asarray(value).reshape(()).item()
                    columns[key].append(value)
            columns["timestamp"].append(frame_index / self.fps)
            columns["frame_index"].append(frame_index)
            columns["episode_index"].append(episode_index)
            columns["index"].append(self.frame_count + frame_index)
            columns["task_index"].append(self.tasks[task])

        episode_chunk = episode_index // OPENPI_CHUNK_SIZE
        parquet_path = self.root / OPENPI_DATA_PATH.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_openpi_v21_parquet(columns, self.hf_features, parquet_path)
        self._append_jsonl(
            self.root / "meta" / "episodes.jsonl",
            {"episode_index": episode_index, "tasks": episode_tasks, "length": len(frames)},
        )
        self._append_jsonl(
            self.root / "meta" / "episodes_stats.jsonl",
            {
                "episode_index": episode_index,
                "stats": compute_episode_stats(columns, self.features),
            },
        )
        self.episode_count += 1
        self.frame_count += len(frames)
        self._write_info()

    def _write_info(self) -> None:
        total_chunks = (self.episode_count + OPENPI_CHUNK_SIZE - 1) // OPENPI_CHUNK_SIZE
        info = {
            "codebase_version": OPENPI_LEROBOT_VERSION,
            "robot_type": "am_bench",
            "total_episodes": self.episode_count,
            "total_frames": self.frame_count,
            "total_tasks": len(self.tasks),
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": OPENPI_CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{self.episode_count}"},
            "data_path": OPENPI_DATA_PATH,
            "video_path": None,
            "features": self.features,
            "ambench": {"openpi_export": self.export_metadata},
        }
        (self.root / "meta" / "info.json").write_text(
            json.dumps(info, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_roots",
        type=Path,
        nargs="+",
        required=True,
        help="Canonical LeRobot roots or recorder session roots containing a lerobot/ child.",
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Output LeRobot repository ID.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Output dataset root. Defaults to $HF_LEROBOT_HOME/<repo_id>.",
    )
    parser.add_argument("--target_hz", type=int, default=20, help="Logical policy FPS for the OpenPI export.")
    parser.add_argument("--ee_image_key", default=DEFAULT_EE_IMAGE_KEY, help="Canonical EE camera feature key.")
    parser.add_argument("--base_image_key", default=DEFAULT_BASE_IMAGE_KEY, help="Canonical base camera feature key.")
    parser.add_argument(
        "--omit_base_image",
        action="store_true",
        help="Write only the EE camera for OpenPI configurations that mask the base image.",
    )
    parser.add_argument("--task_prompt", default="", help="Prompt override for every exported frame.")
    parser.add_argument("--task_prompt_map", type=Path, default=None, help="JSON mapping from source tasks to prompts.")
    parser.add_argument(
        "--require_task_prompt_map",
        action="store_true",
        help="Fail when a source task is absent from --task_prompt_map.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.require_task_prompt_map and args.task_prompt_map is None and not args.task_prompt:
        raise ValueError("--require_task_prompt_map requires --task_prompt_map unless --task_prompt is set.")

    output_root = (args.output_root or (Path(HF_LEROBOT_HOME) / args.repo_id)).expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    resolved_roots = [resolve_dataset_root(path) for path in args.dataset_roots]
    prompt_map = load_task_prompt_map(args.task_prompt_map)
    output_dataset: OpenPIV21Writer | None = None
    output_action_semantics: str | None = None
    reports: list[dict[str, Any]] = []
    for source_index, dataset_root in enumerate(resolved_roots):
        # Load and validate the source metadata before opening the dataset.
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"Expected a JSON object in {info_path}.")
        repo_id = info.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            repo_id = f"ambench/local_export_{source_index}"
        raw_dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=False)
        action_semantics = resolve_dataset_action_semantics(raw_dataset)
        if action_semantics not in (EE_ABSOLUTE, BASE_JOINT_ABSOLUTE):
            raise ValueError(
                "OpenPI export supports canonical ee_absolute and base_joint_absolute datasets only. "
                f"Dataset {dataset_root} has action semantics {action_semantics!r}."
            )
        if output_action_semantics is None:
            output_action_semantics = action_semantics
        elif action_semantics != output_action_semantics:
            raise ValueError(
                "Cannot merge EE and BaseJoint datasets into one OpenPI export. "
                f"Expected {output_action_semantics!r}, got {action_semantics!r} from {dataset_root}."
            )
        stride = compute_stride(int(raw_dataset.fps), args.target_hz)
        episodes = build_resampled_episodes(raw_dataset, stride)
        reports.append({
            "dataset_root": str(dataset_root),
            "repo_id": repo_id,
            "action_semantics": action_semantics,
            "raw_fps": int(raw_dataset.fps),
            "target_hz": int(args.target_hz),
            "stride": stride,
            "raw_frames": int(raw_dataset.num_frames),
            "logical_frames": sum(episode.logical_length for episode in episodes),
        })

        for episode in episodes:
            frames: list[dict[str, Any]] = []
            for logical_step in range(episode.logical_length):
                raw_index = episode.raw_from_index + logical_step * stride
                frame = openpi_frame(
                    raw_dataset[raw_index],
                    dataset=raw_dataset,
                    info=info,
                    action_semantics=action_semantics,
                    ee_image_key=args.ee_image_key,
                    base_image_key=args.base_image_key,
                    include_base_image=not args.omit_base_image,
                    task_prompt=args.task_prompt,
                    task_prompt_map=prompt_map,
                    require_task_prompt_map=args.require_task_prompt_map,
                )
                if output_dataset is None:
                    output_dataset = OpenPIV21Writer(
                        output_root,
                        fps=args.target_hz,
                        features=build_features(frame),
                        export_metadata={
                            "source_dataset_roots": [str(path) for path in resolved_roots],
                            "source_action_semantics": action_semantics,
                            "target_hz": int(args.target_hz),
                            "schema": "openpi_am_bench",
                            "lerobot_version": OPENPI_LEROBOT_VERSION,
                            "include_base_image": not args.omit_base_image,
                        },
                    )
                frames.append(frame)
            if output_dataset is not None:
                output_dataset.save_episode(frames)

    if output_dataset is None:
        raise ValueError("No logical frames were available to export.")
    report_path = output_root / "ambench_openpi_export_report.json"
    report_path.write_text(
        json.dumps(
            {
                "output_root": str(output_root),
                "repo_id": args.repo_id,
                "target_hz": int(args.target_hz),
                "include_base_image": not args.omit_base_image,
                "sources": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported OpenPI AM-Bench dataset to {output_root}")
    print(f"Wrote export report to {report_path}")


if __name__ == "__main__":
    main()
