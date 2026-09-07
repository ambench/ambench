# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Public dataset recorder used by demo collection scripts."""

from __future__ import annotations

import logging
from itertools import count
from pathlib import Path
from typing import Any

from .dataset_common import (
    EpisodeWriterProtocol,
    default_repo_id,
    default_task_prompt,
    infer_dataset_fps,
)
from .lerobot_writer import EpisodeLeRobotWriter, LeRobotSessionWriter
from .paths import LEROBOT_DIRNAME, VIDEO_DIRNAME
from .utils import capture_camera_frames, resolve_camera_names
from .video import _EpisodeVideoWriter

logger = logging.getLogger(__name__)

DEFAULT_STATE_KEYS = ("ee_pos", "ee_quat", "gripper_width")


class DatasetRecorder:
    """Record LeRobot demos for one collection session."""

    def __init__(
        self,
        *,
        env: Any,
        output_dir: str | Path,
        env_cfg: Any,
        camera_names: list[str] | None = None,
        record_video: bool = False,
        video_name_prefix: str = "video",
        video_fps: int = 30,
        video_frame_skip: int = 1,
        repo_id: str | None = None,
        state_keys: list[str] | tuple[str, ...] | None = None,
        task_prompt: str | None = None,
        lerobot_use_videos: bool = False,
        save_failed_episodes: bool = False,
    ) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.output_dir = Path(output_dir).resolve()
        self.env_cfg = env_cfg
        self.camera_names = resolve_camera_names(self.env, camera_names, require_any=record_video)
        self.video_dir = self.output_dir / VIDEO_DIRNAME if record_video else None
        self.lerobot_root = self.output_dir / LEROBOT_DIRNAME
        self.state_keys = tuple(state_keys or DEFAULT_STATE_KEYS)
        self.task_prompt = task_prompt or default_task_prompt(env_cfg, self.output_dir)
        self.save_failed_episodes = save_failed_episodes

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.video_dir is not None:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        self._episode_index_counter = count()
        self._successful_episode_count = 0
        self._episode_steps: list[int] = []
        self._episode_backends: list[list[EpisodeWriterProtocol]] = [[] for _ in range(self.env.num_envs)]
        self._lerobot_session: LeRobotSessionWriter | None = None

        robot_type = env_cfg.robot_profile.robot.robot_id
        self._lerobot_session = LeRobotSessionWriter(
            root=self.lerobot_root,
            repo_id=repo_id or default_repo_id(self.output_dir),
            robot_type=robot_type,
            fps=infer_dataset_fps(self.env, env_cfg),
            state_keys=self.state_keys,
            task_prompt=self.task_prompt,
            use_videos=lerobot_use_videos,
            env_cfg=env_cfg,
        )
        for env_id in range(self.env.num_envs):
            self._episode_backends[env_id].append(EpisodeLeRobotWriter(self._lerobot_session))

        self._video_writers = (
            [
                _EpisodeVideoWriter(
                    self.video_dir,
                    env_id=env_id,
                    name_prefix=video_name_prefix,
                    fps=video_fps,
                    frame_skip=video_frame_skip,
                    output_format="mp4",
                )
                for env_id in range(self.env.num_envs)
            ]
            if self.video_dir is not None
            else None
        )

    @property
    def successful_episode_count(self) -> int:
        return self._successful_episode_count

    @property
    def episode_steps(self) -> list[int]:
        return list(self._episode_steps)

    @property
    def canonical_output_dir(self) -> Path:
        if self._lerobot_session is not None:
            return self._lerobot_session.root
        return self.output_dir

    def add_step(self, obs_batch: list[dict[str, Any]], action_batch: Any) -> None:
        for env_id, episode_backends in enumerate(self._episode_backends):
            try:
                action = action_batch[env_id]
            except Exception as exc:
                raise TypeError(
                    f"Expected batch-first values indexable by environment, got {type(action_batch)!r}"
                ) from exc

            images = (
                capture_camera_frames(self.env, env_id, self.camera_names, data_types=("rgb",))
                if self.camera_names
                else None
            )
            if not images:
                images = None

            for backend in episode_backends:
                backend.add_step(dict(obs_batch[env_id]), action, images=images)

            if self._video_writers is not None:
                self._video_writers[env_id].add_step(images)

    def finish_episode(self, env_id: int, success: bool = False) -> None:
        episode_backends = self._episode_backends[env_id]
        video_writer = self._video_writers[env_id] if self._video_writers is not None else None

        if not any(backend.has_data for backend in episode_backends):
            self._reset_episode_state(env_id, video_writer)
            return

        if not success and not self.save_failed_episodes:
            self._reset_episode_state(env_id, video_writer)
            return

        episode_idx = next(self._episode_index_counter)
        step_count = max((backend.step_count for backend in episode_backends), default=0)
        for backend in episode_backends:
            if backend.has_data:
                backend.finalize_episode(episode_idx, self.env_cfg, success=success)

        if video_writer is not None:
            video_writer.finalize_episode(episode_idx)

        if step_count > 0:
            if success:
                self._successful_episode_count += 1
            self._episode_steps.append(step_count)
            logger.info(
                "Env %s saved %s episode %s with %s steps",
                env_id,
                "successful" if success else "failed",
                episode_idx,
                step_count,
            )

    def reset(self, env_ids: list[int] | None = None) -> None:
        target_env_ids = env_ids if env_ids is not None else list(range(len(self._episode_backends)))
        for env_id in target_env_ids:
            self._reset_episode_state(
                env_id,
                self._video_writers[env_id] if self._video_writers is not None else None,
            )

    def close(self) -> None:
        self.reset()
        if self._lerobot_session is not None:
            self._lerobot_session.close()

    def _reset_episode_state(self, env_id: int, video_writer: _EpisodeVideoWriter | None) -> None:
        for backend in self._episode_backends[env_id]:
            backend.reset()
        if video_writer is not None:
            video_writer.reset()
