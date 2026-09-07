# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset

from ambench_learn.data.action_resampling import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
    compute_stride,
    resolve_dataset_action_semantics,
)
from ambench_learn.data.action_semantics import (
    localize_base_joint_observation_state,
    localize_ee_observation_state,
    resolve_policy_action_representation_for_dataset,
    to_base_joint_relative_trajectory,
    to_ee_local_relative_trajectory,
)


@dataclass(frozen=True)
class ResampledEpisode:
    """Logical benchmark episode aligned to the canonical raw episode."""

    episode_index: int
    raw_from_index: int
    raw_to_index: int
    raw_length: int
    logical_from_index: int
    logical_to_index: int
    logical_length: int


class _ResampledMetaView:
    """Runtime LeRobot-like metadata for a resampled logical dataset view."""

    def __init__(
        self,
        raw_meta: Any,
        *,
        target_fps: int,
        stride: int,
        action_semantics: str,
        policy_action_representation: str,
        logical_num_frames: int,
        logical_num_episodes: int,
        logical_stats: dict[str, dict[str, torch.Tensor]],
    ) -> None:
        self.repo_id = raw_meta.repo_id
        self.revision = getattr(raw_meta, "revision", None)
        self.root = raw_meta.root
        self.info = dict(raw_meta.info)
        self.info["fps"] = target_fps
        self.info["total_frames"] = logical_num_frames
        self.info["total_episodes"] = logical_num_episodes
        self.tasks = raw_meta.tasks
        self.stats = logical_stats
        self.resampling = {
            "raw_fps": int(raw_meta.fps),
            "target_fps": int(target_fps),
            "stride": int(stride),
            "action_semantics": action_semantics,
            "policy_action_representation": policy_action_representation,
            "logical_total_frames": int(logical_num_frames),
        }

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])


def build_resampled_episodes(raw_dataset: LeRobotDataset, stride: int) -> list[ResampledEpisode]:
    """Build logical episode boundaries from canonical raw metadata."""

    episodes: list[ResampledEpisode] = []
    logical_cursor = 0
    for episode_index in range(len(raw_dataset.meta.episodes)):
        raw_episode = raw_dataset.meta.episodes[episode_index]
        raw_from_index = int(raw_episode["dataset_from_index"])
        raw_to_index = int(raw_episode["dataset_to_index"])
        raw_length = int(raw_episode["length"])
        logical_length = raw_length // stride
        episodes.append(
            ResampledEpisode(
                episode_index=episode_index,
                raw_from_index=raw_from_index,
                raw_to_index=raw_to_index,
                raw_length=raw_length,
                logical_from_index=logical_cursor,
                logical_to_index=logical_cursor + logical_length,
                logical_length=logical_length,
            )
        )
        logical_cursor += logical_length
    return episodes


def _compute_feature_stats(values: torch.Tensor) -> dict[str, torch.Tensor]:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    flattened = values.reshape(values.shape[0], -1)
    quantile_levels = {
        "q01": 0.01,
        "q10": 0.10,
        "q50": 0.50,
        "q90": 0.90,
        "q99": 0.99,
    }
    stats = {
        "min": flattened.min(dim=0).values,
        "max": flattened.max(dim=0).values,
        "mean": flattened.mean(dim=0),
        "std": flattened.std(dim=0, correction=0),
        "count": torch.tensor([flattened.shape[0]], dtype=torch.int64),
    }
    for key, quantile in quantile_levels.items():
        stats[key] = torch.quantile(flattened, quantile, dim=0)
    return stats


class ResampledLeRobotDataset(Dataset[dict[str, Any]]):
    """LeRobot-compatible logical dataset view for benchmark-aligned training."""

    def __init__(
        self,
        raw_dataset: LeRobotDataset,
        *,
        target_fps: int,
        action_window_length: int,
        policy_action_representation: str | None = None,
    ) -> None:
        if action_window_length < 1:
            raise ValueError(f"`action_window_length` must be >= 1. Got {action_window_length}.")

        self.raw_dataset = raw_dataset
        self.raw_fps = int(raw_dataset.fps)
        self.target_fps = int(target_fps)
        self.stride = compute_stride(self.raw_fps, self.target_fps)
        self.action_window_length = int(action_window_length)
        self.action_semantics = resolve_dataset_action_semantics(raw_dataset)
        self.policy_action_representation = resolve_policy_action_representation_for_dataset(
            self.action_semantics,
            policy_action_representation,
            policy_name="Resampled LeRobot dataset",
        )
        self._raw_actions = self._load_raw_actions()

        self.episodes = build_resampled_episodes(raw_dataset, self.stride)
        self._episode_logical_stops = [episode.logical_to_index for episode in self.episodes]
        self.num_frames = sum(episode.logical_length for episode in self.episodes)
        self.num_episodes = len(self.episodes)
        self._episode_actions = self._build_episode_actions()
        self._episode_states = self._build_episode_states()
        self._logical_states = (
            torch.cat(self._episode_states, dim=0) if self._episode_states else torch.empty((0, 0), dtype=torch.float32)
        )
        self._logical_stats = self._build_logical_stats()
        self.meta = _ResampledMetaView(
            raw_dataset.meta,
            target_fps=self.target_fps,
            stride=self.stride,
            action_semantics=self.action_semantics,
            policy_action_representation=self.policy_action_representation,
            logical_num_frames=self.num_frames,
            logical_num_episodes=self.num_episodes,
            logical_stats=self._logical_stats,
        )

    def __len__(self) -> int:
        return self.num_frames

    def _resolve_logical_index(self, idx: int) -> tuple[ResampledEpisode, int, int]:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Logical index {idx} is out of range for dataset length {self.num_frames}.")

        episode_list_index = bisect_right(self._episode_logical_stops, idx)
        episode = self.episodes[episode_list_index]
        logical_step_in_episode = idx - episode.logical_from_index
        raw_index = episode.raw_from_index + logical_step_in_episode * self.stride
        return episode, logical_step_in_episode, raw_index

    def _load_raw_actions(self) -> torch.Tensor:
        self.raw_dataset._ensure_hf_dataset_loaded()
        return torch.stack(list(self.raw_dataset.hf_dataset["action"])).to(dtype=torch.float32)

    def _load_raw_states(self) -> torch.Tensor:
        self.raw_dataset._ensure_hf_dataset_loaded()
        return torch.stack(list(self.raw_dataset.hf_dataset["observation.state"])).to(dtype=torch.float32)

    def _build_episode_actions(self) -> list[torch.Tensor]:
        episode_actions: list[torch.Tensor] = []
        for episode in self.episodes:
            if episode.logical_length == 0:
                action_dim = int(self.raw_dataset.meta.features["action"]["shape"][0])
                episode_actions.append(torch.empty((0, action_dim), dtype=torch.float32))
                continue

            if self.action_semantics in (EE_ABSOLUTE, BASE_JOINT_ABSOLUTE):
                raw_actions = self._raw_actions[episode.raw_from_index : episode.raw_to_index]
                episode_actions.append(raw_actions[:: self.stride][: episode.logical_length])
                continue

            raise ValueError(f"Unsupported action semantics: {self.action_semantics}")

        return episode_actions

    def _build_episode_states(self) -> list[torch.Tensor]:
        raw_states = self._load_raw_states()
        logical_states: list[torch.Tensor] = []
        for episode in self.episodes:
            if episode.logical_length == 0:
                continue
            episode_states = raw_states[episode.raw_from_index : episode.raw_to_index]
            logical_states.append(episode_states[:: self.stride][: episode.logical_length])
        return logical_states

    def _build_logical_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        logical_stats = deepcopy(self.raw_dataset.meta.stats)
        if self.num_frames == 0:
            return logical_stats

        logical_stats["action"] = self._build_action_stats()
        if "observation.state" in self.raw_dataset.meta.features:
            logical_states = self._logical_states
            if self.policy_action_representation == "ee_local_relative":
                logical_states = localize_ee_observation_state(logical_states)
            elif self.policy_action_representation == "base_joint_relative":
                logical_states = localize_base_joint_observation_state(logical_states)
            logical_stats["observation.state"] = _compute_feature_stats(logical_states)
        return logical_stats

    def _build_action_stats(self) -> dict[str, torch.Tensor]:
        if self.policy_action_representation not in ("ee_local_relative", "base_joint_relative"):
            return _compute_feature_stats(torch.cat(self._episode_actions, dim=0))

        relative_chunks: list[torch.Tensor] = []
        for episode, episode_actions, episode_states in zip(
            self.episodes, self._episode_actions, self._episode_states, strict=True
        ):
            if episode.logical_length == 0:
                continue
            for logical_step_in_episode in range(episode.logical_length):
                action_window, _ = self._build_action_window(logical_step_in_episode, episode)
                anchor_state = episode_states[logical_step_in_episode]
                relative_chunks.append(self._transform_action_window(action_window, anchor_state))

        if not relative_chunks:
            return _compute_feature_stats(torch.cat(self._episode_actions, dim=0))
        return _compute_feature_stats(torch.cat(relative_chunks, dim=0))

    def _build_action_window(
        self,
        logical_step_in_episode: int,
        episode: ResampledEpisode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        episode_actions = self._episode_actions[episode.episode_index]
        action_is_pad = torch.zeros(self.action_window_length, dtype=torch.bool)
        last_valid_step = max(episode.logical_length - 1, 0)
        action_indices: list[int] = []

        for offset in range(self.action_window_length):
            logical_step = logical_step_in_episode + offset
            if logical_step >= episode.logical_length:
                action_is_pad[offset] = True
                logical_step = last_valid_step
            action_indices.append(logical_step)

        return episode_actions[action_indices], action_is_pad

    def _transform_action_window(self, action_window: torch.Tensor, anchor_state: torch.Tensor) -> torch.Tensor:
        if self.policy_action_representation == "ee_local_relative":
            return to_ee_local_relative_trajectory(action_window.unsqueeze(0), anchor_state.unsqueeze(0))[0]
        if self.policy_action_representation == "base_joint_relative":
            return to_base_joint_relative_trajectory(action_window.unsqueeze(0), anchor_state.unsqueeze(0))[0]
        return action_window

    def _transform_observation_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.policy_action_representation == "ee_local_relative":
            return localize_ee_observation_state(state)
        if self.policy_action_representation == "base_joint_relative":
            return localize_base_joint_observation_state(state)
        return state

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, logical_step_in_episode, raw_index = self._resolve_logical_index(idx)
        raw_item = self.raw_dataset[raw_index]
        item = dict(raw_item)
        item["index"] = torch.tensor(idx, dtype=torch.int64)
        item["frame_index"] = torch.tensor(logical_step_in_episode, dtype=torch.int64)
        item["raw_frame_index"] = raw_item["frame_index"].clone()
        item["raw_index"] = raw_item["index"].clone()
        action_window, action_is_pad = self._build_action_window(logical_step_in_episode, episode)
        anchor_state = self._episode_states[episode.episode_index][logical_step_in_episode]
        item["action"] = self._transform_action_window(action_window, anchor_state)
        item["action_is_pad"] = action_is_pad
        if "observation.state" in item:
            item["observation.state"] = self._transform_observation_state(item["observation.state"])
        item["benchmark.target_fps"] = torch.tensor(self.target_fps, dtype=torch.int64)
        item["benchmark.raw_fps"] = torch.tensor(self.raw_fps, dtype=torch.int64)
        item["benchmark.stride"] = torch.tensor(self.stride, dtype=torch.int64)
        return item

    def build_report(self) -> dict[str, Any]:
        """Build a compact summary of the raw-vs-logical dataset view."""

        return {
            "raw_fps": self.raw_fps,
            "target_fps": self.target_fps,
            "stride": self.stride,
            "action_semantics": self.action_semantics,
            "policy_action_representation": self.policy_action_representation,
            "raw_num_frames": int(self.raw_dataset.num_frames),
            "logical_num_frames": int(self.num_frames),
            "num_episodes": int(self.num_episodes),
            "raw_episode_lengths": tuple(episode.raw_length for episode in self.episodes),
            "logical_episode_lengths": tuple(episode.logical_length for episode in self.episodes),
        }
