# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Public video recording wrapper for rollout and evaluation scripts."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from ambench.utils.image_processing import depth_to_preview_rgb

from .utils import (
    DEPTH_SUFFIX,
    capture_camera_frames,
    resolve_camera_names,
)

logger = logging.getLogger(__name__)


class RecordVideo(gym.Wrapper):
    """Thin wrapper that records rollout videos from Isaac Lab environments."""

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        camera_names: str | list[str] | None = None,
        name_prefix: str = "video",
        fps: int = 30,
        frame_skip: int = 1,
        output_format: str = "gif",
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ) -> None:
        super().__init__(env)
        self.video_folder = Path(video_folder).resolve()
        self.video_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(camera_names, str):
            requested_camera_names = [camera_names]
        else:
            requested_camera_names = list(camera_names or [])

        self.camera_names = resolve_camera_names(self.env, requested_camera_names, require_any=True)
        self.name_prefix = name_prefix
        self.num_envs = getattr(self.env.unwrapped, "num_envs", 1)
        self.episode_ids = [-1 for _ in range(self.num_envs)]
        self.recording = False
        self._writers = [
            _EpisodeVideoWriter(
                self.video_folder,
                env_id=env_id,
                name_prefix=self.name_prefix,
                fps=fps,
                frame_skip=frame_skip,
                output_format=output_format,
                gc_trigger=gc_trigger,
            )
            for env_id in range(self.num_envs)
        ]

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        was_recording = self.recording
        self.recording = True
        for env_id, writer in enumerate(self._writers):
            if was_recording and writer.has_data:
                writer.finalize_episode(self.episode_ids[env_id])
                self.episode_ids[env_id] += 1
            elif not was_recording:
                self.episode_ids[env_id] += 1
            writer.reset()

        self._capture_frame()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.recording:
            terminated_np = terminated.detach().cpu().numpy() if torch.is_tensor(terminated) else np.asarray(terminated)
            truncated_np = truncated.detach().cpu().numpy() if torch.is_tensor(truncated) else np.asarray(truncated)
            done_mask = np.logical_or(terminated_np, truncated_np).reshape(-1)

            # Capture the post-step frame before finalizing done envs so the
            # terminal frame stays with the current episode instead of creating
            # a one-frame "next episode" clip.
            self._capture_frame()

            for env_id, writer in enumerate(self._writers):
                if env_id < len(done_mask) and done_mask[env_id]:
                    writer.finalize_episode(self.episode_ids[env_id])
                    self.episode_ids[env_id] += 1
                    writer.reset()

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.recording:
            for env_id, writer in enumerate(self._writers):
                if writer.has_data:
                    writer.finalize_episode(self.episode_ids[env_id])
        super().close()

    def _capture_frame(self) -> None:
        captured_any = False
        for env_id, writer in enumerate(self._writers):
            images = capture_camera_frames(self.env, env_id, self.camera_names, data_types=("rgb", "depth"))
            if images:
                captured_any = True
            writer.add_step(images)

        if not captured_any:
            raise RuntimeError(
                "No video frames were captured from environment sensors."
                f" Requested camera names: {self.camera_names}."
                " Video recording requires at least one valid camera sensor."
            )


class _EpisodeVideoWriter:
    """Collect frames for one episode and persist them as video artifacts."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        env_id: int = 0,
        name_prefix: str = "video",
        fps: int = 30,
        frame_skip: int = 1,
        output_format: str = "gif",
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.env_id = env_id
        self.name_prefix = name_prefix
        self.fps = fps
        self.frame_skip = max(1, frame_skip)
        self.output_format = output_format
        self.gc_trigger = gc_trigger

        self._frames: dict[str, list[np.ndarray]] = {}
        self._step_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_data(self) -> bool:
        return any(frames for frames in self._frames.values())

    def add_step(self, images: dict[str, Any] | None) -> None:
        if not images:
            self._step_count += 1
            return

        should_capture = self._step_count == 0 or (self._step_count % self.frame_skip == 0)
        if should_capture:
            for stream_name, image in images.items():
                preview = depth_to_preview_rgb(image) if stream_name.endswith(DEPTH_SUFFIX) else np.asarray(image)
                self._frames.setdefault(stream_name, []).append(preview)

        self._step_count += 1

    def finalize_episode(self, episode_idx: int) -> None:
        if not self.has_data:
            self.reset()
            return

        for camera_name, frames in self._frames.items():
            if not frames:
                continue

            file_path = (
                self.output_dir
                / f"{self.name_prefix}-{camera_name}-env{self.env_id}-eps{episode_idx}.{self.output_format}"
            )
            try:
                if self.output_format == "gif":
                    import imageio

                    imageio.mimsave(file_path, frames, fps=self.fps, loop=0)
                elif self.output_format == "mp4":
                    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                    clip = ImageSequenceClip(frames, fps=self.fps)
                    clip.write_videofile(str(file_path), logger=None)
                else:
                    logger.error("Unsupported video format: %s", self.output_format)
                    continue

                logger.info("Saved video to %s", file_path)
            except Exception:
                logger.exception("Failed to save video to %s", file_path)

        if self.gc_trigger and self.gc_trigger(episode_idx):
            gc.collect()

        self.reset()

    def reset(self) -> None:
        self._frames = {}
        self._step_count = 0
