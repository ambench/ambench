#!/usr/bin/env python
# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize a canonical AM-Bench LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ambench_learn.data.action_resampling import resolve_dataset_action_semantics
from ambench_learn.data.resampled_lerobot_dataset import ResampledLeRobotDataset

DEFAULT_REPO_ID = "ambench/local_validation"
REQUIRED_FEATURE_KEYS = ("observation.state", "action")


def resolve_dataset_root(dataset_root: Path) -> Path:
    """Resolve either a direct LeRobot root or a recorder session root."""
    if (dataset_root / "meta" / "info.json").is_file():
        return dataset_root

    canonical_root = dataset_root / "lerobot"
    if (canonical_root / "meta" / "info.json").is_file():
        return canonical_root

    raise FileNotFoundError(
        f"Could not find a LeRobot dataset under '{dataset_root}'. "
        "Expected either 'meta/info.json' directly or a 'lerobot/meta/info.json' child."
    )


def load_info_json(dataset_root: Path) -> dict[str, Any]:
    """Load canonical LeRobot dataset metadata."""
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    if not isinstance(info, dict):
        raise ValueError(f"Expected a JSON object in {info_path}.")
    return info


def load_episode_lengths(dataset: LeRobotDataset) -> list[int]:
    """Extract and validate per-episode lengths from finalized metadata."""
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        raise ValueError("LeRobot episode metadata is missing.")
    try:
        lengths = [int(length) for length in episodes["length"]]
    except Exception as exc:
        raise ValueError("LeRobot episode metadata has no readable length column.") from exc
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"Episode lengths must be positive; got {lengths}.")
    return lengths


def _validate_declared_metadata(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a feature dictionary.")
    missing_features = [key for key in REQUIRED_FEATURE_KEYS if key not in features]
    if missing_features:
        raise ValueError(f"Dataset metadata is missing required features: {missing_features}.")

    for key in ("total_episodes", "total_frames", "fps"):
        value = info.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"meta/info.json field {key!r} must be positive; got {value!r}.")
    return features


def _validate_sample(sample: dict[str, Any], features: dict[str, Any], sample_index: int) -> None:
    for key in REQUIRED_FEATURE_KEYS:
        if key not in sample:
            raise ValueError(f"Frame {sample_index} is missing required feature {key!r}.")
        expected_shape = tuple(features[key].get("shape", ()))
        actual_shape = tuple(sample[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Frame {sample_index} feature {key!r} has shape {actual_shape}; expected {expected_shape}."
            )

    for key, feature in features.items():
        if not str(key).startswith("observation.images."):
            continue
        if key not in sample:
            raise ValueError(f"Frame {sample_index} is missing declared image feature {key!r}.")
        actual_shape = tuple(sample[key].shape)
        expected_shape = tuple(feature.get("shape", ()))
        channel_first_shape = (
            (expected_shape[2], expected_shape[0], expected_shape[1]) if len(expected_shape) == 3 else ()
        )
        if actual_shape not in {expected_shape, channel_first_shape}:
            raise ValueError(
                f"Frame {sample_index} image {key!r} has shape {actual_shape}; expected "
                f"{expected_shape} or {channel_first_shape}."
            )


def build_summary(
    dataset_root: Path,
    repo_id: str,
    *,
    target_hz: int | None,
) -> list[str]:
    """Validate the dataset and return a concise human-readable summary."""
    if target_hz is not None and target_hz <= 0:
        raise ValueError("target_hz must be positive when provided.")

    info = load_info_json(dataset_root)
    features = _validate_declared_metadata(info)
    episode_files = list((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise ValueError("Finalized episode metadata Parquet files are missing.")

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, download_videos=False)
    episode_lengths = load_episode_lengths(dataset)
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if len(episode_lengths) != total_episodes:
        raise ValueError(
            f"Episode metadata contains {len(episode_lengths)} episodes, but info.json reports {total_episodes}."
        )
    if sum(episode_lengths) != total_frames:
        raise ValueError(f"Episode lengths sum to {sum(episode_lengths)} frames, but info.json reports {total_frames}.")
    if len(dataset) != total_frames:
        raise ValueError(f"LeRobotDataset loads {len(dataset)} frames, but info.json reports {total_frames}.")

    for sample_index in sorted({0, total_frames - 1}):
        _validate_sample(dataset[sample_index], features, sample_index)

    action_semantics = resolve_dataset_action_semantics(dataset)
    lines = [
        "LeRobot dataset validation passed",
        f"Resolved root: {dataset_root}",
        f"Episodes: {total_episodes}",
        f"Frames: {total_frames}",
        f"FPS: {info['fps']}",
        f"Action semantics: {action_semantics}",
        f"Feature keys: {list(features)}",
        (
            "Episode lengths: "
            f"min={min(episode_lengths)}, max={max(episode_lengths)}, "
            f"mean={statistics.fmean(episode_lengths):.2f}, median={statistics.median(episode_lengths)}"
        ),
    ]
    if len(episode_lengths) <= 12:
        lines.append(f"Episode length list: {episode_lengths}")

    image_features = [key for key in features if str(key).startswith("observation.images.")]
    lines.append(f"Image features: {image_features or 'none'}")

    if target_hz is not None:
        resampled_dataset = ResampledLeRobotDataset(
            dataset,
            target_fps=target_hz,
            action_window_length=1,
        )
        report = resampled_dataset.build_report()
        logical_episode_lengths = [episode.logical_length for episode in resampled_dataset.episodes]
        lines.extend([
            "Benchmark resampling validation passed",
            f"Raw FPS: {report['raw_fps']}",
            f"Target FPS: {report['target_fps']}",
            f"Stride: {report['stride']}",
            f"Raw frames -> logical frames: {report['raw_num_frames']} -> {report['logical_num_frames']}",
            f"Benchmark episode length list: {logical_episode_lengths}",
        ])
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to a LeRobot dataset root, or its parent recorder session root.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Repository id used by the local LeRobot loader.",
    )
    parser.add_argument(
        "--target_hz",
        type=int,
        default=None,
        help="Optionally validate the benchmark's logically resampled dataset view.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset_root = resolve_dataset_root(args.dataset_root.expanduser().resolve())
        lines = build_summary(dataset_root, args.repo_id, target_hz=args.target_hz)
    except Exception as exc:
        print(f"LeRobot dataset validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
