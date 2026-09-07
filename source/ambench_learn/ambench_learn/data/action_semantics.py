# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ambench_learn.utils.rotation_math import normalize_quat, quat_conjugate, quat_mul

# Key of the AM-Bench block inside a LeRobot dataset's meta/info.json. The
# legacy key is what datasets recorded before the package rename carry, and is
# still accepted on read so those datasets stay usable.
METADATA_NAMESPACE = "ambench"
LEGACY_METADATA_NAMESPACE = "am_isaac"

EE_ABSOLUTE = "ee_absolute"
BASE_JOINT_ABSOLUTE = "base_joint_absolute"

EE_LOCAL_RELATIVE = "ee_local_relative"
BASE_JOINT_RELATIVE = "base_joint_relative"

ACTION_SEMANTICS_CHOICES = (EE_ABSOLUTE, BASE_JOINT_ABSOLUTE)
POLICY_ACTION_REPRESENTATION_CHOICES = (EE_LOCAL_RELATIVE, BASE_JOINT_RELATIVE)

EE_ABSOLUTE_DIM = 8
EE_LOCAL_RELATIVE_DIM = 8
BASE_JOINT_ABSOLUTE_DIM = 12
BASE_JOINT_RELATIVE_DIM = 12

DP_EE_RAW_ACTION_DIM = 7
DP_EE_MODEL_ACTION_DIM = 10
DP_BASE_JOINT_RAW_ACTION_DIM = 11
DP_BASE_JOINT_MODEL_ACTION_DIM = 14

EE_POS_SLICE = slice(0, 3)
EE_QUAT_SLICE = slice(3, 7)
EE_GRIPPER_SLICE = slice(7, 8)

BASE_JOINT_POS_SLICE = slice(0, 3)
BASE_JOINT_QUAT_SLICE = slice(3, 7)
BASE_JOINT_ARM_SLICE = slice(7, 11)
BASE_JOINT_GRIPPER_SLICE = slice(11, 12)


def dataset_metadata(info: Any) -> dict:
    """Return the AM-Bench block from a dataset's ``meta/info.json``.

    Falls back to the pre-rename key so datasets recorded before the package
    was renamed remain readable.
    """

    if not isinstance(info, dict):
        return {}
    block = info.get(METADATA_NAMESPACE)
    if block is None:
        block = info.get(LEGACY_METADATA_NAMESPACE)
    return block if isinstance(block, dict) else {}


def resolve_policy_action_representation_for_dataset(
    action_semantics: str,
    requested: str | None = None,
    *,
    policy_name: str = "Policy",
) -> str:
    """Resolve the canonical policy representation for a dataset action contract."""

    if requested is not None and requested not in POLICY_ACTION_REPRESENTATION_CHOICES:
        raise ValueError(
            f"policy_action_representation must be one of {POLICY_ACTION_REPRESENTATION_CHOICES}. Got {requested!r}."
        )

    if action_semantics == EE_ABSOLUTE:
        if requested == BASE_JOINT_RELATIVE:
            raise ValueError(
                f"{policy_name} absolute EE datasets require policy_action_representation='ee_local_relative'."
            )
        return EE_LOCAL_RELATIVE

    if action_semantics == BASE_JOINT_ABSOLUTE:
        if requested == EE_LOCAL_RELATIVE:
            raise ValueError(
                f"{policy_name} Base+joints absolute datasets require "
                "policy_action_representation='base_joint_relative'."
            )
        return BASE_JOINT_RELATIVE

    raise ValueError(f"Unsupported action semantics: {action_semantics!r}.")


def resolve_eval_action_semantics(env_or_cfg: Any) -> str:
    """Resolve eval-time action semantics from an environment or its config."""

    from ambench.controllers.control_pipeline import ActionMode

    env_or_cfg = getattr(env_or_cfg, "unwrapped", env_or_cfg)
    env_cfg = getattr(env_or_cfg, "cfg", env_or_cfg)
    action_mode = env_cfg.robot_profile.control.action_mode
    if action_mode == ActionMode.ABSOLUTE_EE_POSE:
        return EE_ABSOLUTE
    if action_mode == ActionMode.ABSOLUTE_BASE_JOINTS:
        return BASE_JOINT_ABSOLUTE
    raise ValueError(f"Unsupported action mode: {action_mode}.")


def resolve_current_absolute_action(env: Any, env_index: int = 0) -> torch.Tensor:
    """Return the current env-origin-relative [pos, quat, gripper] EE setpoint."""

    env_unwrapped = getattr(env, "unwrapped", env)
    env_origin = env_unwrapped.scene.env_origins[env_index].to(dtype=torch.float32)
    current_pos = env_unwrapped.ee_cmd_pos_w[env_index].to(dtype=torch.float32) - env_origin
    current_quat = normalize_quat(env_unwrapped.ee_cmd_quat_w[env_index].to(dtype=torch.float32))
    current_gripper = resolve_current_gripper_command(env_unwrapped, env_index)
    return torch.cat((current_pos, current_quat, current_gripper), dim=0)


def resolve_current_base_joint_action(env: Any, env_index: int = 0) -> torch.Tensor:
    """Return the current absolute 12D BaseJoint command/state for one env."""

    env_unwrapped = getattr(env, "unwrapped", env)
    env_origin = env_unwrapped.scene.env_origins[env_index].to(dtype=torch.float32)

    is_first_step = getattr(env_unwrapped, "is_first_step", None)
    is_reset_anchor = bool(is_first_step[env_index]) if is_first_step is not None else False
    if not is_reset_anchor and hasattr(env_unwrapped, "actions") and env_unwrapped.actions.shape[-1] == 12:
        current_action = env_unwrapped.actions[env_index].to(dtype=torch.float32).clone()
        current_action[BASE_JOINT_POS_SLICE] -= env_origin
        current_action[BASE_JOINT_QUAT_SLICE] = normalize_quat(current_action[BASE_JOINT_QUAT_SLICE])
        return current_action

    base_state = env_unwrapped.robot.data.body_link_state_w[env_index, env_unwrapped.base_link_idx, :]
    current_base_pos = base_state[0:3].to(dtype=torch.float32) - env_origin
    current_base_quat = normalize_quat(base_state[3:7].to(dtype=torch.float32))

    if hasattr(env_unwrapped, "arm_targets"):
        current_arm = env_unwrapped.arm_targets[env_index].to(dtype=torch.float32)
    else:
        current_arm = env_unwrapped.robot.data.joint_pos[env_index, env_unwrapped.arm_joint_ids].to(dtype=torch.float32)

    current_gripper = resolve_current_gripper_command(env_unwrapped, env_index)
    return torch.cat((current_base_pos, current_base_quat, current_arm, current_gripper), dim=0)


def resolve_current_base_joint_actions(env: Any, env_indices: list[int] | None = None) -> torch.Tensor:
    env_unwrapped = getattr(env, "unwrapped", env)
    if env_indices is None:
        env_indices = list(range(env_unwrapped.num_envs))
    return torch.stack(
        [resolve_current_base_joint_action(env_unwrapped, env_index) for env_index in env_indices], dim=0
    )


def resolve_current_gripper_command(env: Any, env_index: int = 0) -> torch.Tensor:
    """Return the current normalized gripper command for one env."""

    env_unwrapped = getattr(env, "unwrapped", env)
    if hasattr(env_unwrapped, "actions") and env_unwrapped.actions.shape[-1] >= 1:
        return env_unwrapped.actions[env_index, -1:].to(dtype=torch.float32)

    joint_limits = env_unwrapped.robot.data.soft_joint_pos_limits[0, env_unwrapped.gripper_joint_ids, :]
    current_targets = env_unwrapped.gripper_targets[env_index]
    normalized = (current_targets - joint_limits[:, 0]) / (joint_limits[:, 1] - joint_limits[:, 0]).clamp_min(1.0e-9)
    return (2.0 * normalized.mean().unsqueeze(0) - 1.0).to(dtype=torch.float32)


def canonicalize_action_quaternion_sign(action: torch.Tensor, reference_action: torch.Tensor) -> torch.Tensor:
    """Flip one action quaternion to match a reference action's quaternion sign."""

    if action.shape[-1] < 7:
        return action
    result = action.clone()
    quat = normalize_quat(result[EE_QUAT_SLICE].to(dtype=torch.float32))
    reference_quat = normalize_quat(reference_action[EE_QUAT_SLICE].to(device=quat.device, dtype=quat.dtype))
    if torch.sum(quat * reference_quat) < 0.0:
        quat = -quat
    result[EE_QUAT_SLICE] = quat
    return result


def canonicalize_absolute_action_quaternion_sign(
    action: torch.Tensor,
    env: Any,
    env_index: int = 0,
    reference_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flip one EE absolute action quaternion against a command-state reference."""

    if action.shape[-1] < 7:
        return action
    if reference_action is None:
        env_unwrapped = getattr(env, "unwrapped", env)
        reference_action = torch.cat(
            (
                env_unwrapped.ee_cmd_pos_w[env_index].to(dtype=torch.float32),
                normalize_quat(env_unwrapped.ee_cmd_quat_w[env_index].to(dtype=torch.float32)),
                resolve_current_gripper_command(env_unwrapped, env_index),
            ),
            dim=0,
        )
    return canonicalize_action_quaternion_sign(action, reference_action)


def canonicalize_abs_quaternion_signs(
    actions: torch.Tensor,
    env: Any,
    *,
    action_semantics: str = EE_ABSOLUTE,
    env_indices: list[int] | None = None,
    reference_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flip batched absolute-action quaternions to match previous command signs."""

    if actions.shape[-1] < 7:
        return actions
    env_unwrapped = getattr(env, "unwrapped", env)

    actions = actions.clone()
    quat = torch.nn.functional.normalize(actions[:, EE_QUAT_SLICE], dim=-1)
    if reference_actions is not None:
        prev_quat = reference_actions[:, EE_QUAT_SLICE].to(device=quat.device, dtype=quat.dtype)
    elif action_semantics == BASE_JOINT_ABSOLUTE:
        prev_quat = resolve_current_base_joint_actions(env_unwrapped, env_indices)[:, BASE_JOINT_QUAT_SLICE].to(
            device=quat.device,
            dtype=quat.dtype,
        )
    else:
        if env_indices is None:
            env_indices = list(range(actions.shape[0]))
        prev_quat = env_unwrapped.ee_cmd_quat_w[env_indices].to(device=quat.device, dtype=quat.dtype)
    sign = torch.where(torch.sum(quat * prev_quat, dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
    actions[:, EE_QUAT_SLICE] = quat * sign
    return actions


def to_ee_local_relative_trajectory(actions: Tensor, state: Tensor) -> Tensor:
    """Express absolute EE trajectories in the current EE frame."""

    _validate_ee_trajectory_inputs(actions, state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_view = _broadcast_state(state[..., :EE_LOCAL_RELATIVE_DIM], actions.ndim)
    local = actions.clone()
    state_quat = normalize_quat(state_view[..., EE_QUAT_SLICE])
    action_quat = _canonicalize_quat_sign(normalize_quat(local[..., EE_QUAT_SLICE]), state_quat)
    local[..., EE_POS_SLICE] = _rotate_vector(
        quat_conjugate(state_quat),
        actions[..., EE_POS_SLICE] - state_view[..., EE_POS_SLICE],
    )
    local[..., EE_QUAT_SLICE] = _canonicalize_quat_sign(
        normalize_quat(quat_mul(quat_conjugate(state_quat), action_quat))
    )
    local[..., EE_GRIPPER_SLICE] = actions[..., EE_GRIPPER_SLICE]
    return local


def to_ee_local_absolute_trajectory(actions: Tensor, state: Tensor) -> Tensor:
    """Decode EE-local trajectories into absolute EE commands."""

    _validate_ee_trajectory_inputs(actions, state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_view = _broadcast_state(state[..., :EE_LOCAL_RELATIVE_DIM], actions.ndim)
    absolute = actions.clone()
    state_quat = normalize_quat(state_view[..., EE_QUAT_SLICE])
    relative_quat = _canonicalize_quat_sign(normalize_quat(absolute[..., EE_QUAT_SLICE]))
    absolute[..., EE_POS_SLICE] = state_view[..., EE_POS_SLICE] + _rotate_vector(
        state_quat,
        actions[..., EE_POS_SLICE],
    )
    absolute[..., EE_QUAT_SLICE] = _canonicalize_quat_sign(
        normalize_quat(quat_mul(state_quat, relative_quat)),
        state_quat,
    )
    absolute[..., EE_GRIPPER_SLICE] = actions[..., EE_GRIPPER_SLICE]
    return absolute


def localize_ee_observation_state(state: Tensor) -> Tensor:
    """Replace the absolute EE pose prefix with the local-frame identity pose."""

    if state.shape[-1] < EE_LOCAL_RELATIVE_DIM:
        raise ValueError(
            "ee_local_relative requires observation.state to begin with "
            "[ee_pos(3), ee_quat(4), gripper_width(1)]. "
            f"Got state_dim={state.shape[-1]}."
        )
    localized = state.clone()
    localized[..., EE_POS_SLICE] = 0.0
    localized[..., EE_QUAT_SLICE] = localized[..., EE_QUAT_SLICE] * 0.0
    localized[..., 3] = 1.0
    return localized


def localize_base_joint_observation_state(state: Tensor) -> Tensor:
    """Replace the absolute base pose prefix with the local-frame identity pose."""

    if state.shape[-1] < BASE_JOINT_RELATIVE_DIM:
        raise ValueError(
            "base_joint_relative requires observation.state to begin with "
            "[base_pos(3), base_quat(4), arm_joint_pos(4), gripper_width(1)]. "
            f"Got state_dim={state.shape[-1]}."
        )
    localized = state.clone()
    localized[..., BASE_JOINT_POS_SLICE] = 0.0
    localized[..., BASE_JOINT_QUAT_SLICE] = localized[..., BASE_JOINT_QUAT_SLICE] * 0.0
    localized[..., 3] = 1.0
    return localized


def to_base_joint_relative_trajectory(actions: Tensor, state: Tensor) -> Tensor:
    """Express absolute BaseJoint trajectories relative to the current state."""

    _validate_base_joint_trajectory_inputs(actions, state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_view = _broadcast_state(state[..., :BASE_JOINT_RELATIVE_DIM], actions.ndim)
    relative = actions.clone()
    state_quat = normalize_quat(state_view[..., BASE_JOINT_QUAT_SLICE])
    relative[..., BASE_JOINT_POS_SLICE] = _rotate_vector(
        quat_conjugate(state_quat),
        actions[..., BASE_JOINT_POS_SLICE] - state_view[..., BASE_JOINT_POS_SLICE],
    )
    action_quat = _canonicalize_quat_sign(normalize_quat(relative[..., BASE_JOINT_QUAT_SLICE]), state_quat)
    relative[..., BASE_JOINT_QUAT_SLICE] = _canonicalize_quat_sign(
        normalize_quat(quat_mul(quat_conjugate(state_quat), action_quat))
    )
    relative[..., BASE_JOINT_ARM_SLICE] -= state_view[..., BASE_JOINT_ARM_SLICE]
    # Gripper stays in command space because observation.state stores physical width.
    relative[..., BASE_JOINT_GRIPPER_SLICE] = actions[..., BASE_JOINT_GRIPPER_SLICE]
    return relative


def to_base_joint_absolute_trajectory(actions: Tensor, state: Tensor) -> Tensor:
    """Decode relative BaseJoint trajectories into absolute commands."""

    _validate_base_joint_trajectory_inputs(actions, state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    state_view = _broadcast_state(state[..., :BASE_JOINT_RELATIVE_DIM], actions.ndim)
    absolute = actions.clone()
    state_quat = normalize_quat(state_view[..., BASE_JOINT_QUAT_SLICE])
    relative_quat = _canonicalize_quat_sign(normalize_quat(absolute[..., BASE_JOINT_QUAT_SLICE]))
    absolute[..., BASE_JOINT_POS_SLICE] = state_view[..., BASE_JOINT_POS_SLICE] + _rotate_vector(
        state_quat,
        actions[..., BASE_JOINT_POS_SLICE],
    )
    absolute[..., BASE_JOINT_QUAT_SLICE] = _canonicalize_quat_sign(
        normalize_quat(quat_mul(state_quat, relative_quat)),
        state_quat,
    )
    absolute[..., BASE_JOINT_ARM_SLICE] += state_view[..., BASE_JOINT_ARM_SLICE]
    absolute[..., BASE_JOINT_GRIPPER_SLICE] = actions[..., BASE_JOINT_GRIPPER_SLICE]
    return absolute


def _canonicalize_quat_sign(quat: Tensor, reference: Tensor | None = None) -> Tensor:
    result = quat.clone()
    if reference is None:
        flip_mask = result[..., :1] < 0.0
    else:
        flip_mask = (result * reference).sum(dim=-1, keepdim=True) < 0.0
    return torch.where(flip_mask, -result, result)


def _broadcast_state(state: Tensor, action_ndim: int) -> Tensor:
    if action_ndim == 3:
        return state.unsqueeze(-2)
    return state


def _rotate_vector(quat: Tensor, vector: Tensor) -> Tensor:
    quat = normalize_quat(quat)
    zeros = torch.zeros_like(vector[..., :1])
    vector_quat = torch.cat((zeros, vector), dim=-1)
    return quat_mul(quat_mul(quat, vector_quat), quat_conjugate(quat))[..., 1:]


def _validate_ee_trajectory_inputs(actions: Tensor, state: Tensor) -> None:
    if actions.ndim not in (2, 3):
        raise ValueError(f"Expected actions with shape (B, 8) or (B, T, 8). Got {tuple(actions.shape)}.")
    if actions.shape[-1] != EE_LOCAL_RELATIVE_DIM:
        raise ValueError(f"Expected 8D EE actions. Got {tuple(actions.shape)}.")
    if state.ndim != 2:
        raise ValueError(f"Expected state with shape (B, state_dim). Got {tuple(state.shape)}.")
    if state.shape[-1] < EE_LOCAL_RELATIVE_DIM:
        raise ValueError(
            "ee_local_relative requires observation.state to begin with "
            "[ee_pos(3), ee_quat(4), gripper_width(1)]. "
            f"Got state_dim={state.shape[-1]}."
        )


def _validate_base_joint_trajectory_inputs(actions: Tensor, state: Tensor) -> None:
    if actions.ndim not in (2, 3):
        raise ValueError(f"Expected actions with shape (B, 12) or (B, T, 12). Got {tuple(actions.shape)}.")
    if actions.shape[-1] != BASE_JOINT_RELATIVE_DIM:
        raise ValueError(f"Expected 12D Base+joints actions. Got {tuple(actions.shape)}.")
    if state.ndim != 2:
        raise ValueError(f"Expected state with shape (B, state_dim). Got {tuple(state.shape)}.")
    if state.shape[-1] < BASE_JOINT_RELATIVE_DIM:
        raise ValueError(
            "base_joint_relative requires observation.state to begin with "
            "[base_pos(3), base_quat(4), arm_joint_pos(4), gripper_width(1)]. "
            f"Got state_dim={state.shape[-1]}."
        )
