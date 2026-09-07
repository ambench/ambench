# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


class EpisodeWriterProtocol(Protocol):
    @property
    def has_data(self) -> bool:
        pass

    @property
    def step_count(self) -> int:
        pass

    def add_step(
        self,
        obs: dict[str, Any],
        action: Any,
        images: dict[str, Any] | None = None,
    ) -> None:
        pass

    def finalize_episode(
        self,
        episode_idx: int,
        env_cfg: Any,
        success: bool,
    ) -> Path:
        pass

    def reset(self) -> None:
        pass


def to_numpy_array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def infer_dataset_fps(env: Any, env_cfg: Any) -> int:
    env_unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    step_dt = getattr(env_unwrapped, "step_dt", None)
    if step_dt is not None and float(step_dt) > 0.0:
        return int(round(1.0 / float(step_dt)))

    sim_cfg = getattr(env_cfg, "sim", None)
    dt = getattr(sim_cfg, "dt", None) if sim_cfg is not None else None
    decimation = getattr(env_cfg, "decimation", None)
    if dt is not None and decimation is not None and float(dt) > 0.0 and float(decimation) > 0.0:
        return int(round(1.0 / (float(dt) * float(decimation))))

    logger.warning("Falling back to 30 FPS for LeRobot metadata because env.step_dt was unavailable.")
    return 30


def default_task_prompt(env_cfg: Any, output_dir: Path) -> str:
    env_name = getattr(env_cfg, "env_name", None)
    if env_name:
        return str(env_name)
    return output_dir.parent.name or output_dir.name


def default_repo_id(output_dir: Path) -> str:
    parent_name = output_dir.parent.name or "dataset"
    slug = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        f"{parent_name}_{output_dir.name}".strip().lower(),
    )
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    dataset_name = slug or "dataset"
    return f"ambench/{dataset_name}"


def build_state_vector(obs: dict[str, Any], state_keys: tuple[str, ...]) -> np.ndarray:
    missing_state_keys = [key for key in state_keys if key not in obs]
    if missing_state_keys:
        raise KeyError(
            f"Missing observation keys {missing_state_keys} while building observation.state. "
            f"Available keys: {sorted(obs.keys())}"
        )

    state_parts = []
    for key in state_keys:
        value_array = to_numpy_array(obs[key]).astype(np.float32, copy=False)
        if value_array.ndim == 0:
            state_parts.append(value_array.reshape(1))
            continue
        if value_array.ndim > 1:
            logger.debug(
                "Flattening multi-dimensional observation key '%s' from shape %s",
                key,
                value_array.shape,
            )
        state_parts.append(value_array.reshape(-1))
    return np.concatenate(state_parts, axis=0).astype(np.float32, copy=False)


def build_action_vector(action: Any) -> np.ndarray:
    return to_numpy_array(action).astype(np.float32, copy=False).reshape(-1)


def resolve_action_semantics_from_env_cfg(env_cfg: Any) -> str:
    from ambench.controllers.control_pipeline import ActionMode

    action_mode = env_cfg.robot_profile.control.action_mode
    if action_mode == ActionMode.ABSOLUTE_EE_POSE:
        return "ee_absolute"
    if action_mode == ActionMode.ABSOLUTE_BASE_JOINTS:
        return "base_joint_absolute"
    raise ValueError(f"Unsupported action mode: {action_mode}.")


def build_lerobot_features(
    sample_frame: dict[str, Any],
    *,
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    state = np.asarray(sample_frame["observation.state"])
    action = np.asarray(sample_frame["action"])
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(state.shape),
            "names": [f"state_{idx}" for idx in range(state.shape[-1])],
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(action.shape),
            "names": [f"action_{idx}" for idx in range(action.shape[-1])],
        },
    }

    image_dtype = "video" if use_videos else "image"
    for key, value in sample_frame.items():
        if not key.startswith("observation.images."):
            continue
        image_array = np.asarray(value)
        features[key] = {
            "dtype": image_dtype,
            "shape": tuple(image_array.shape),
            "names": ["height", "width", "channels"],
        }

    return features
