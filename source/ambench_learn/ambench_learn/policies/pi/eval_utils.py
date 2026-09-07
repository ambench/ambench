# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
import contextlib
from typing import Any

import numpy as np
import torch

from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_image
from ambench_learn.data.action_resampling import (
    interpolate_absolute_action_for_execution,
    interpolate_base_joint_action_for_execution,
)
from ambench_learn.data.action_semantics import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
    canonicalize_absolute_action_quaternion_sign,
    canonicalize_action_quaternion_sign,
    resolve_current_absolute_action,
    resolve_current_base_joint_action,
)


def _tensor_to_numpy(tensor: torch.Tensor, *, shape: tuple[int, ...]) -> np.ndarray:
    array = tensor.detach().to(torch.float32).cpu().numpy().reshape(shape)
    return array.astype(np.float32, copy=False)


def expand_policy_actions(
    actions: np.ndarray,
    env: Any,
    *,
    action_semantics: str,
    n_action_steps: int,
    action_execution_stride: int,
    env_index: int = 0,
) -> collections.deque[torch.Tensor]:
    """Expand one OpenPI action chunk into environment-rate commands."""
    if actions.ndim != 2:
        raise ValueError(f"Expected action chunk with shape (H, A), got {actions.shape}.")
    if actions.shape[1] != env.action_space.shape[-1]:
        raise ValueError(
            f"Expected action dim {env.action_space.shape[-1]} to match env action space, got {actions.shape[1]}."
        )
    if len(actions) < n_action_steps:
        raise ValueError(
            f"We want to execute {n_action_steps} policy steps, but policy only predicts {len(actions)} steps."
        )

    action_plan: collections.deque[torch.Tensor] = collections.deque()
    previous_abs_action = None
    if action_semantics == EE_ABSOLUTE:
        previous_abs_action = resolve_current_absolute_action(env.unwrapped, env_index)
    elif action_semantics == BASE_JOINT_ABSOLUTE:
        previous_abs_action = resolve_current_base_joint_action(env.unwrapped, env_index)

    for action_np in actions[:n_action_steps]:
        action_i = torch.from_numpy(action_np).to(device=env.unwrapped.device, dtype=torch.float32)
        if action_execution_stride == 1:
            if action_semantics == EE_ABSOLUTE:
                action_i = canonicalize_absolute_action_quaternion_sign(
                    action_i,
                    env.unwrapped,
                    env_index=env_index,
                    reference_action=previous_abs_action,
                )
                previous_abs_action = action_i
            elif action_semantics == BASE_JOINT_ABSOLUTE:
                if action_i.shape[-1] != 12:
                    raise ValueError(
                        "12D Base+joints action execution is required for base_joint_absolute envs. "
                        f"Got action shape {tuple(action_i.shape)}."
                    )
                if previous_abs_action is None:
                    previous_abs_action = resolve_current_base_joint_action(env.unwrapped, env_index)
                action_i = canonicalize_action_quaternion_sign(action_i, previous_abs_action)
                previous_abs_action = action_i
            action_plan.append(action_i)
            continue

        if action_semantics == EE_ABSOLUTE:
            if action_i.shape[-1] != 8:
                raise ValueError(
                    "8D EE absolute action interpolation is required for --policy-target-hz with ee_absolute envs. "
                    f"Got action shape {tuple(action_i.shape)}."
                )
            if previous_abs_action is None:
                previous_abs_action = resolve_current_absolute_action(env.unwrapped, env_index)
            action_i = canonicalize_absolute_action_quaternion_sign(
                action_i,
                env.unwrapped,
                env_index=env_index,
                reference_action=previous_abs_action,
            )
            action_plan.extend(
                interpolate_absolute_action_for_execution(
                    previous_abs_action.to(device=action_i.device, dtype=torch.float32),
                    action_i,
                    action_execution_stride,
                )
            )
            previous_abs_action = action_i
        elif action_semantics == BASE_JOINT_ABSOLUTE:
            if action_i.shape[-1] != 12:
                raise ValueError(
                    "12D Base+joints action interpolation is required for --policy-target-hz with "
                    f"base_joint_absolute envs. Got action shape {tuple(action_i.shape)}."
                )
            if previous_abs_action is None:
                previous_abs_action = resolve_current_base_joint_action(env.unwrapped, env_index)
            action_i = canonicalize_action_quaternion_sign(action_i, previous_abs_action)
            action_plan.extend(
                interpolate_base_joint_action_for_execution(
                    previous_abs_action.to(device=action_i.device, dtype=torch.float32),
                    action_i,
                    action_execution_stride,
                )
            )
            previous_abs_action = action_i
        else:
            raise ValueError(f"Unsupported action semantics for OpenPI eval: {action_semantics}")

    return action_plan


def default_action_for_env(env: Any, *, action_semantics: str, env_index: int) -> torch.Tensor:
    """Return a safe environment command for an already-finished rollout."""
    if action_semantics == EE_ABSOLUTE:
        return resolve_current_absolute_action(env.unwrapped, env_index).to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
    if action_semantics == BASE_JOINT_ABSOLUTE:
        return resolve_current_base_joint_action(env.unwrapped, env_index).to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
    raise ValueError(f"Unsupported action semantics for OpenPI eval: {action_semantics}")


def build_openpi_example(
    raw_obs: dict,
    env: Any,
    *,
    env_index: int,
    action_semantics: str,
    prompt: str,
) -> dict:
    """Build an EE or BaseJoint OpenPI example from measured observations."""
    obs_i = raw_obs["policy"][env_index]

    # Prefer the measured policy observation, with live joint state as fallback.
    if "gripper_width" in obs_i:
        gripper_width = _tensor_to_numpy(obs_i["gripper_width"], shape=(1,))
    elif hasattr(env.unwrapped, "gripper_joint_ids"):
        joint_pos = env.unwrapped.robot.data.joint_pos[env_index, env.unwrapped.gripper_joint_ids]
        joint_pos = joint_pos.detach().to(torch.float32)
        gripper_width = joint_pos.mean().reshape(1).cpu().numpy().astype(np.float32, copy=False)
    else:
        raise KeyError("Could not find gripper width in policy observation or environment state.")

    example = {
        "am_bench/gripper_width": gripper_width,
        "am_bench/ee_image": prepare_rgb_image(get_camera_rgb(env.unwrapped, "ee_camera", env_index)),
        "prompt": prompt,
    }
    if action_semantics == BASE_JOINT_ABSOLUTE:
        example.update({
            "am_bench/base_pos": _tensor_to_numpy(obs_i["base_pos"], shape=(3,)),
            "am_bench/base_quat": _tensor_to_numpy(obs_i["base_quat"], shape=(4,)),
            "am_bench/arm_joint_pos": _tensor_to_numpy(obs_i["arm_joint_pos"], shape=(4,)),
        })
    else:
        example.update({
            "am_bench/ee_pos": _tensor_to_numpy(obs_i["ee_pos"], shape=(3,)),
            "am_bench/ee_quat": _tensor_to_numpy(obs_i["ee_quat"], shape=(4,)),
        })

    with contextlib.suppress(AttributeError, KeyError):
        example["am_bench/base_image"] = prepare_rgb_image(get_camera_rgb(env.unwrapped, "base_camera", env_index))

    return example
