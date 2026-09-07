# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info

from .dataset_common import (
    EpisodeWriterProtocol,
    build_action_vector,
    build_lerobot_features,
    build_state_vector,
    resolve_action_semantics_from_env_cfg,
    to_numpy_array,
)
from .paths import write_env_cfg

logger = logging.getLogger(__name__)


class LeRobotSessionWriter:
    """Own the write-mode LeRobot dataset for one recording session."""

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        robot_type: str,
        fps: int,
        state_keys: tuple[str, ...],
        task_prompt: str,
        use_videos: bool,
        env_cfg: Any,
    ) -> None:
        self.root = Path(root).resolve()
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps
        self.state_keys = state_keys
        self.task_prompt = task_prompt
        self.use_videos = use_videos
        self.env_cfg = env_cfg
        self._dataset: LeRobotDataset | None = None

    def build_frame(
        self,
        obs: dict[str, Any],
        action: Any,
        images: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "observation.state": build_state_vector(obs, self.state_keys),
            "action": build_action_vector(action),
            "task": self.task_prompt,
        }
        for camera_name, image in (images or {}).items():
            frame[f"observation.images.{camera_name}"] = to_numpy_array(image).astype(np.uint8, copy=False)
        return frame

    def save_episode(self, frames: list[dict[str, Any]]) -> Path:
        if not frames:
            raise RuntimeError("Cannot save an empty LeRobot episode.")

        if self._dataset is None:
            self._create_dataset(frames[0])

        if self._dataset is None:
            raise RuntimeError("LeRobot dataset was not initialized.")

        for frame in frames:
            self._dataset.add_frame(dict(frame))
        self._dataset.save_episode()
        return self.root

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.finalize()

    def _create_dataset(self, sample_frame: dict[str, Any]) -> None:
        features = build_lerobot_features(sample_frame, use_videos=self.use_videos)
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.root,
            robot_type=self.robot_type,
            fps=self.fps,
            features=features,
            use_videos=self.use_videos,
            image_writer_threads=4 if not self.use_videos else 0,
            image_writer_processes=0,
        )
        self._dataset.meta.info.setdefault("ambench", {})
        self._dataset.meta.info["ambench"]["action_semantics"] = resolve_action_semantics_from_env_cfg(self.env_cfg)
        self._dataset.meta.info["ambench"]["state_keys"] = list(self.state_keys)
        write_info(self._dataset.meta.info, self.root)
        try:
            write_env_cfg(self.root, self.env_cfg)
        except Exception as exc:
            logger.warning("Failed to write env_cfg.yaml into canonical LeRobot dataset '%s': %s", self.root, exc)


class EpisodeLeRobotWriter(EpisodeWriterProtocol):
    """Stage one environment's episode in memory until it is committed on success."""

    def __init__(self, session_writer: LeRobotSessionWriter) -> None:
        self._session_writer = session_writer
        self._frames: list[dict[str, Any]] = []

    @property
    def has_data(self) -> bool:
        return bool(self._frames)

    @property
    def step_count(self) -> int:
        return len(self._frames)

    def add_step(self, obs: dict[str, Any], action: Any, images: dict[str, Any] | None = None) -> None:
        self._frames.append(self._session_writer.build_frame(obs, action, images))

    def finalize_episode(self, episode_idx: int, env_cfg: Any, success: bool) -> Path:
        _ = episode_idx, env_cfg, success
        saved_path = self._session_writer.save_episode(self._frames)
        self.reset()
        return saved_path

    def reset(self) -> None:
        self._frames.clear()
