# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ambench.recording.paths import load_env_cfg
from ambench_learn.data.action_semantics import (
    ACTION_SEMANTICS_CHOICES,
    BASE_JOINT_ABSOLUTE,
    BASE_JOINT_ABSOLUTE_DIM,
    EE_ABSOLUTE,
    EE_ABSOLUTE_DIM,
    dataset_metadata,
)
from ambench_learn.utils.rotation_math import (
    normalize_quat,
    slerp_quat,
)


def resolve_dataset_action_semantics(raw_dataset: LeRobotDataset) -> str:
    """Resolve action semantics from dataset metadata, falling back to env_cfg for older datasets."""

    info = getattr(raw_dataset.meta, "info", {})
    ambench_info = dataset_metadata(info)
    action_semantics = ambench_info.get("action_semantics")
    if action_semantics in ACTION_SEMANTICS_CHOICES:
        return str(action_semantics)

    action_dim = int(raw_dataset.meta.features["action"]["shape"][0])
    if action_dim == EE_ABSOLUTE_DIM:
        return EE_ABSOLUTE
    if action_dim == BASE_JOINT_ABSOLUTE_DIM:
        return BASE_JOINT_ABSOLUTE

    try:
        env_cfg = load_env_cfg(raw_dataset.root)
    except Exception:
        env_cfg = None
    if isinstance(env_cfg, dict):
        action_semantics = env_cfg.get("action_semantics")
        if action_semantics in ACTION_SEMANTICS_CHOICES:
            return str(action_semantics)
    raise ValueError(
        f"Could not infer action semantics for action_dim={action_dim}. "
        "This dataset is missing both `meta/info.json -> ambench.action_semantics` and a recoverable `env_cfg.yaml`."
    )


def compute_stride(raw_fps: int, target_fps: int) -> int:
    """Validate the requested logical benchmark rate and return the raw-step stride."""

    if target_fps < 1:
        raise ValueError(f"`target_fps` must be >= 1. Got {target_fps}.")
    if raw_fps < 1:
        raise ValueError(f"`raw_fps` must be >= 1. Got {raw_fps}.")
    if raw_fps % target_fps != 0:
        raise ValueError(
            f"Only integer downsampling is supported in the first release. Got raw_fps={raw_fps}, "
            f"target_fps={target_fps}."
        )
    return raw_fps // target_fps


def infer_action_window_length(policy_cfg: Any) -> int:
    """Resolve the logical action horizon length expected by the policy."""

    action_delta_indices = getattr(policy_cfg, "action_delta_indices", None)
    if action_delta_indices:
        expected = list(range(len(action_delta_indices)))
        if list(action_delta_indices) != expected:
            raise ValueError(
                "The on-the-fly benchmark resampler currently expects contiguous action_delta_indices "
                f"starting at zero. Got {action_delta_indices}."
            )
        return len(action_delta_indices)

    chunk_size = getattr(policy_cfg, "chunk_size", None)
    if chunk_size is not None:
        return int(chunk_size)

    return 1


def interpolate_absolute_action_for_execution(
    previous_action: torch.Tensor,
    action: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Interpolate one macro absolute action into raw-rate micro setpoints.

    The interpolation anchor is the previously commanded absolute setpoint. Position and gripper
    are linearly interpolated. Orientation uses SLERP after canonicalizing the endpoint sign
    against the previous command quaternion.
    """

    if previous_action.ndim != 1 or previous_action.shape[0] != 8:
        raise ValueError(f"Expected one 8D previous absolute action. Got {tuple(previous_action.shape)}.")
    if action.ndim != 1 or action.shape[0] != 8:
        raise ValueError(f"Expected one 8D absolute action. Got {tuple(action.shape)}.")
    if stride < 1:
        raise ValueError(f"`stride` must be >= 1. Got {stride}.")

    previous_action = previous_action.to(dtype=torch.float32)
    action = action.to(dtype=torch.float32)
    if stride == 1:
        return action.unsqueeze(0)

    interpolation_fractions = torch.arange(1, stride + 1, device=action.device, dtype=torch.float32).unsqueeze(
        1
    ) / float(stride)

    previous_pos = previous_action[0:3].unsqueeze(0)
    target_pos = action[0:3].unsqueeze(0)
    interpolated_pos = torch.lerp(previous_pos, target_pos, interpolation_fractions)

    previous_quat = normalize_quat(previous_action[3:7]).unsqueeze(0)
    target_quat = normalize_quat(action[3:7]).unsqueeze(0)
    dot = torch.sum(previous_quat * target_quat, dim=-1, keepdim=True)
    target_quat = torch.where(dot < 0.0, -target_quat, target_quat)
    interpolated_quat = slerp_quat(
        previous_quat.expand(stride, -1),
        target_quat.expand(stride, -1),
        interpolation_fractions,
    )

    previous_gripper = previous_action[7:8].unsqueeze(0)
    target_gripper = action[7:8].unsqueeze(0)
    interpolated_gripper = torch.lerp(previous_gripper, target_gripper, interpolation_fractions)

    return torch.cat((interpolated_pos, interpolated_quat, interpolated_gripper), dim=1)


def interpolate_base_joint_action_for_execution(
    previous_action: torch.Tensor,
    action: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Interpolate one 12D Base+joints absolute action into raw-rate setpoints."""

    if previous_action.ndim != 1 or previous_action.shape[0] != 12:
        raise ValueError(f"Expected one 12D previous Base+joints action. Got {tuple(previous_action.shape)}.")
    if action.ndim != 1 or action.shape[0] != 12:
        raise ValueError(f"Expected one 12D Base+joints action. Got {tuple(action.shape)}.")
    if stride < 1:
        raise ValueError(f"`stride` must be >= 1. Got {stride}.")

    previous_action = previous_action.to(dtype=torch.float32)
    action = action.to(dtype=torch.float32)
    if stride == 1:
        return action.unsqueeze(0)

    interpolation_fractions = torch.arange(1, stride + 1, device=action.device, dtype=torch.float32).unsqueeze(
        1
    ) / float(stride)

    previous_pos = previous_action[0:3].unsqueeze(0)
    target_pos = action[0:3].unsqueeze(0)
    interpolated_pos = torch.lerp(previous_pos, target_pos, interpolation_fractions)

    previous_quat = normalize_quat(previous_action[3:7]).unsqueeze(0)
    target_quat = normalize_quat(action[3:7]).unsqueeze(0)
    dot = torch.sum(previous_quat * target_quat, dim=-1, keepdim=True)
    target_quat = torch.where(dot < 0.0, -target_quat, target_quat)
    interpolated_quat = slerp_quat(
        previous_quat.expand(stride, -1),
        target_quat.expand(stride, -1),
        interpolation_fractions,
    )

    previous_joints = previous_action[7:11].unsqueeze(0)
    target_joints = action[7:11].unsqueeze(0)
    interpolated_joints = torch.lerp(previous_joints, target_joints, interpolation_fractions)

    previous_gripper = previous_action[11:12].unsqueeze(0)
    target_gripper = action[11:12].unsqueeze(0)
    interpolated_gripper = torch.lerp(previous_gripper, target_gripper, interpolation_fractions)

    return torch.cat((interpolated_pos, interpolated_quat, interpolated_joints, interpolated_gripper), dim=1)
