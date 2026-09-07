#!/usr/bin/env python
# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a LeRobot ACT checkpoint on an AM Isaac environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import lerobot.policies.act.processor_act as lerobot_act_processor
from isaaclab.app import AppLauncher

from ambench_learn.data.action_resampling import (
    compute_stride,
    interpolate_absolute_action_for_execution,
    interpolate_base_joint_action_for_execution,
)
from ambench_learn.data.action_semantics import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
    canonicalize_abs_quaternion_signs,
    resolve_current_absolute_action,
    resolve_current_base_joint_action,
    resolve_eval_action_semantics,
)
from ambench_learn.eval.common import (
    EvalBatch,
    add_common_eval_args,
    build_common_eval_metadata,
    configure_env_cfg,
    make_scene_camera_cfg,
    resolve_eval_output_dir,
    run_evaluation_batches,
    validate_common_eval_args,
    validate_video_camera_names,
    wrap_video,
)
from ambench_learn.eval.run import EvalRun
from ambench_learn.policies.act.eval_utils import resolve_act_action_representation
from ambench_learn.policies.act.se_relative_processor import (
    BaseJointRelativeTrajectoryProcessorStep,
    EELocalRelativeTrajectoryProcessorStep,
)
from ambench_learn.policies.act.se_relative_processor import (
    make_act_pre_post_processors as make_am_act_pre_post_processors,
)
from ambench_learn.policies.act.se_relative_processor import (
    reconnect_se_relative_absolute_steps,
)

# Argument parsing
parser = argparse.ArgumentParser(
    description="Evaluate a LeRobot ACT checkpoint on Isaac Lab environments.",
)
add_common_eval_args(parser)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to a LeRobot ACT checkpoint directory or its pretrained_model subdirectory.",
)
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="Optional inference-time override for LeRobot ACT n_action_steps.",
)
parser.add_argument(
    "--temporal-ensemble-coeff",
    type=float,
    default=None,
    help="Optional inference-time override for LeRobot ACT temporal ensembling coefficient.",
)
parser.add_argument(
    "--policy-target-hz",
    type=int,
    default=None,
    help="Optional logical policy control rate to execute at below the raw env rate.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
try:
    validate_common_eval_args(args_cli)
except ValueError as error:
    parser.error(str(error))
if args_cli.n_action_steps is not None and args_cli.n_action_steps < 1:
    parser.error(f"--n-action-steps must be >= 1, got {args_cli.n_action_steps}.")
if args_cli.policy_target_hz is not None and args_cli.policy_target_hz < 1:
    parser.error(f"--policy-target-hz must be >= 1, got {args_cli.policy_target_hz}.")
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
checkpoint = args_cli.checkpoint.expanduser().resolve()
checkpoint = checkpoint if checkpoint.name == "pretrained_model" else checkpoint / "pretrained_model"
if not checkpoint.is_dir():
    parser.error(f"Could not find a LeRobot pretrained_model directory at '{checkpoint}'.")
args_cli.checkpoint = checkpoint

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.utils.seed import configure_seed
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.device_processor import DeviceProcessorStep

import ambench.tasks  # noqa: F401
from ambench.evaluation import SuccessCriteriaTracker
from ambench.recording import RecordVideo
from ambench.utils.camera_utils import get_camera_rgb
from ambench.utils.image_processing import prepare_rgb_tensor


def policy_feature_to_camera_name(feature_key: str) -> str:
    prefix = "observation.images."
    if not feature_key.startswith(prefix):
        raise ValueError(f"Expected visual feature key to start with '{prefix}', got '{feature_key}'.")
    return feature_key[len(prefix) :]


def build_policy_observation_batch(
    raw_policy_obs: list[dict[str, torch.Tensor]],
    env,
    state_keys: list[str],
    policy_cfg: PreTrainedConfig,
    env_indices: list[int],
) -> dict[str, torch.Tensor]:
    """Build a batched policy observation for the selected env indices."""

    state_rows = []
    for env_index in env_indices:
        env_obs = raw_policy_obs[env_index]
        state_parts = []
        for key in state_keys:
            if key not in env_obs:
                raise KeyError(f"Key '{key}' not found in observation dict. Available keys: {list(env_obs.keys())}")
            value = env_obs[key]
            if value.ndim != 1:
                value = value.reshape(-1)
            state_parts.append(value)
        state_rows.append(torch.cat(state_parts, dim=-1))

    observation = {"observation.state": torch.stack(state_rows, dim=0)}
    for feature_key, feature in policy_cfg.image_features.items():
        camera_name = policy_feature_to_camera_name(feature_key)
        expected_shape = tuple(feature.shape) if feature.shape is not None else None
        image = prepare_rgb_tensor(get_camera_rgb(env, camera_name, env_indices))
        produced_shape = tuple(image.shape[1:])
        if expected_shape is not None and produced_shape != expected_shape:
            raise ValueError(
                f"Camera '{camera_name}' produced shape {produced_shape}, but checkpoint expects {expected_shape}."
            )
        observation[feature_key] = image
    return observation


def load_lerobot_act(
    policy_path: str | Path, device: str, n_action_steps: int | None, temporal_ensemble_coeff: float | None
):
    policy_path = Path(policy_path).expanduser().resolve()
    pretrained_dir = policy_path if policy_path.name == "pretrained_model" else policy_path / "pretrained_model"
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find a LeRobot pretrained_model directory under '{policy_path}'. "
            "Pass either the checkpoint dir or the pretrained_model dir directly."
        )
    lerobot_act_processor.make_act_pre_post_processors = make_am_act_pre_post_processors

    cli_overrides = [f"--device={device}"]
    if n_action_steps is not None:
        cli_overrides.append(f"--n_action_steps={n_action_steps}")
    if temporal_ensemble_coeff is not None:
        cli_overrides.append(f"--temporal_ensemble_coeff={temporal_ensemble_coeff}")

    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_dir, cli_overrides=cli_overrides)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(pretrained_dir, config=policy_cfg)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=pretrained_dir)

    # Move every LeRobot preprocessing step onto the evaluation device.
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = device
            step.__post_init__()
        elif hasattr(step, "to"):
            step.to(device=device)

    reconnect_se_relative_absolute_steps(preprocessor, postprocessor)
    policy.to(device)
    policy.eval()
    return pretrained_dir, policy_cfg, policy, preprocessor, postprocessor


def build_execution_sequence(
    env,
    macro_action: torch.Tensor,
    *,
    stride: int,
    action_semantics: str,
    env_index: int = 0,
) -> list[torch.Tensor]:
    """Expand one logical benchmark action into the raw env-rate actions to execute."""

    if stride == 1:
        return [macro_action.to(dtype=torch.float32)]
    if action_semantics == EE_ABSOLUTE:
        previous_action = resolve_current_absolute_action(env, env_index).to(
            device=macro_action.device, dtype=torch.float32
        )
        macro_action = canonicalize_abs_quaternion_signs(
            macro_action.unsqueeze(0),
            env,
            action_semantics=action_semantics,
            env_indices=[env_index],
        )[0]
        return list(interpolate_absolute_action_for_execution(previous_action, macro_action, stride))
    if action_semantics == BASE_JOINT_ABSOLUTE:
        previous_action = resolve_current_base_joint_action(env, env_index).to(
            device=macro_action.device, dtype=torch.float32
        )
        macro_action = canonicalize_abs_quaternion_signs(
            macro_action.unsqueeze(0),
            env,
            action_semantics=action_semantics,
            env_indices=[env_index],
        )[0]
        return list(interpolate_base_joint_action_for_execution(previous_action, macro_action, stride))
    raise ValueError(f"Unsupported action semantics: {action_semantics}")


def refresh_se_relative_decode_anchor(
    policy,
    preprocessor,
    policy_action_representation: str,
) -> None:
    """Cache the measured chunk-start policy state for queued relative ACT actions."""

    if not policy_action_representation.endswith("_relative"):
        return
    if not hasattr(policy, "_action_queue") or len(policy._action_queue) != 0:
        return

    if policy_action_representation == "base_joint_relative":
        relative_step_type = BaseJointRelativeTrajectoryProcessorStep
    elif policy_action_representation == "ee_local_relative":
        relative_step_type = EELocalRelativeTrajectoryProcessorStep
    else:
        raise ValueError(f"Unsupported relative ACT action representation: {policy_action_representation}")
    relative_step = next((step for step in preprocessor.steps if isinstance(step, relative_step_type)), None)
    if relative_step is None or relative_step._last_state is None:
        raise RuntimeError(
            f"{policy_action_representation} eval requires a cached chunk-start observation.state "
            "before decoding ACT chunks."
        )
    relative_step.cache_decode_anchor()


def eval_policy(
    env,
    policy,
    policy_cfg,
    preprocessor,
    postprocessor,
    state_keys: list[str],
    policy_target_hz: int | None,
    action_semantics: str,
    policy_action_representation: str,
    eval_run: EvalRun,
    batch: EvalBatch,
    progress_every: int,
    rollout_seed: int | None = None,
) -> bool:
    device = env.unwrapped.device
    if rollout_seed is None:
        raw_obs, _ = env.reset()
    else:
        raw_obs, _ = env.reset(seed=rollout_seed)
    max_timesteps = env.unwrapped.max_episode_length
    num_envs = env.unwrapped.num_envs
    tracked_env_indices = list(range(batch.size))
    success = [False for _ in range(num_envs)]
    done = [env_index not in tracked_env_indices for env_index in range(num_envs)]
    executed_steps = [0 for _ in range(num_envs)]
    termination_reasons = ["not_evaluated" if done[env_index] else "timeout" for env_index in range(num_envs)]
    success_criteria = [SuccessCriteriaTracker(env_index=env_index) for env_index in range(num_envs)]
    latest_tracking = [None for _ in range(num_envs)]
    for env_index in tracked_env_indices:
        eval_run.start_rollout(
            batch.start + env_index,
            batch_env_index=env_index,
        )

    policy.reset()
    env_fps = round(1.0 / float(env.unwrapped.dt))
    execution_stride = 1 if policy_target_hz is None else compute_stride(env_fps, policy_target_hz)
    pending_actions: list[torch.Tensor] = []
    env_indices = list(range(num_envs))
    ee_local_relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, EELocalRelativeTrajectoryProcessorStep)),
        None,
    )
    if ee_local_relative_step is not None:
        ee_local_relative_step.clear_decode_anchor()
    base_joint_relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, BaseJointRelativeTrajectoryProcessorStep)),
        None,
    )
    if base_joint_relative_step is not None:
        base_joint_relative_step.clear_decode_anchor()

    stop_evaluation = False
    with torch.no_grad():
        for step in range(1, max_timesteps + 1):
            if not simulation_app.is_running() or simulation_app.is_exiting():
                stop_evaluation = True
                for env_index in tracked_env_indices:
                    if not done[env_index]:
                        termination_reasons[env_index] = "app_stopped"
                break
            if all(done):
                break

            if not pending_actions:
                policy_obs = build_policy_observation_batch(
                    raw_obs["policy"],
                    env.unwrapped,
                    state_keys,
                    policy_cfg,
                    env_indices,
                )
                processed_obs = preprocessor(policy_obs)
                refresh_se_relative_decode_anchor(
                    policy,
                    preprocessor,
                    policy_action_representation,
                )
                raw_action = policy.select_action(processed_obs)
                action = postprocessor(raw_action)
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                if action.shape[0] != num_envs:
                    raise ValueError(f"Expected policy action batch {num_envs}, got {tuple(action.shape)}.")
                env_action_sequences = [
                    build_execution_sequence(
                        env,
                        action[env_index],
                        stride=execution_stride,
                        action_semantics=action_semantics,
                        env_index=env_index,
                    )
                    for env_index in env_indices
                ]
                pending_actions = [
                    torch.stack([env_action_sequences[env_index][step_index] for env_index in env_indices], dim=0)
                    for step_index in range(execution_stride)
                ]

            actions = pending_actions.pop(0).to(device=device, dtype=torch.float32)
            if execution_stride == 1:
                actions = canonicalize_abs_quaternion_signs(
                    actions,
                    env,
                    action_semantics=action_semantics,
                    env_indices=env_indices,
                )
            if actions.shape[0] != env.unwrapped.num_envs:
                actions = actions.repeat(env.unwrapped.num_envs, 1)

            raw_obs, reward, terminated, truncated, info = env.step(actions)
            actions_are_nan = torch.isnan(actions).flatten(start_dim=1).any(dim=1)
            stepped_env_indices = [env_index for env_index in tracked_env_indices if not done[env_index]]
            for env_index in env_indices:
                if done[env_index]:
                    continue
                executed_steps[env_index] += 1
                success_criteria[env_index].update(info)
                episode_done = bool(terminated[env_index] or truncated[env_index] or actions_are_nan[env_index])
                if env_index in tracked_env_indices and eval_run.tracking_enabled and not episode_done:
                    latest_tracking[env_index] = eval_run.record_timestep(
                        env=env,
                        raw_obs=raw_obs,
                        rollout_idx=batch.start + env_index,
                        timestep=executed_steps[env_index],
                        env_index=env_index,
                        batch_env_index=env_index,
                    )
                if terminated[env_index]:
                    success[env_index] = True
                    done[env_index] = True
                    termination_reasons[env_index] = "success"
                elif actions_are_nan[env_index]:
                    done[env_index] = True
                    termination_reasons[env_index] = "nan_action"
                elif truncated[env_index]:
                    done[env_index] = True
                    termination_reasons[env_index] = "timeout"

            representative_env = next(
                (env_index for env_index in tracked_env_indices if latest_tracking[env_index] is not None),
                tracked_env_indices[0],
            )
            mean_reward = sum(float(reward[env_index].item()) for env_index in stepped_env_indices) / len(
                stepped_env_indices
            )
            eval_run.print_progress(
                rollout_batch=batch.label(),
                step=step,
                active=(sum(not done[index] for index in tracked_env_indices), batch.size),
                reward=mean_reward,
                tracking_record=latest_tracking[representative_env],
                tracking_env_index=representative_env,
                progress_every=progress_every,
            )

    for env_index in tracked_env_indices:
        rollout_record = eval_run.finish_rollout(
            batch.start + env_index,
            batch_env_index=env_index,
            success=success[env_index],
            executed_steps=executed_steps[env_index],
            subtask_completion=success_criteria[env_index].completion_fraction,
            termination_reason=termination_reasons[env_index],
            extra=success_criteria[env_index].to_record(),
        )
        eval_run.print_rollout_finished(rollout_record, total=eval_run.requested_rollouts)
    return stop_evaluation


def main() -> None:
    if args_cli.seed is not None:
        configure_seed(args_cli.seed)

    pretrained_dir, policy_cfg, policy, preprocessor, postprocessor = load_lerobot_act(
        args_cli.checkpoint,
        args_cli.device,
        args_cli.n_action_steps,
        args_cli.temporal_ensemble_coeff,
    )

    env_name = args_cli.task
    # Enable cameras required by the checkpoint and optional rollout videos.
    camera_names = {policy_feature_to_camera_name(feature_key) for feature_key in policy_cfg.image_features}
    if args_cli.save_video:
        camera_names.update(args_cli.video_camera_names)
    scene_camera_cfg = make_scene_camera_cfg() if "scene_camera" in camera_names else None
    env_cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=env_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        episode_length_s=args_cli.episode_length_s,
        disturbance=args_cli.disturbance,
        scene_camera_cfg=scene_camera_cfg,
    )

    resolved_action_semantics = resolve_eval_action_semantics(env_cfg)
    resolved_policy_action_representation = resolve_act_action_representation(
        resolved_action_semantics,
        policy_cfg,
        postprocessor,
    )
    effective_temporal_ensemble_coeff = getattr(policy_cfg, "temporal_ensemble_coeff", None)
    if effective_temporal_ensemble_coeff is not None and resolved_policy_action_representation.endswith("_relative"):
        raise ValueError(
            "ACT temporal ensembling is not supported for relative action representations because predictions "
            "from different measured-state anchors cannot be combined safely. Disable temporal ensembling or "
            "use an absolute action representation."
        )

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    video_camera_names = validate_video_camera_names(env, args_cli.video_camera_names) if args_cli.save_video else []
    state_keys = (
        ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]
        if resolved_policy_action_representation == "base_joint_relative"
        else ["ee_pos", "ee_quat", "gripper_width"]
    )
    eval_save_dir = resolve_eval_output_dir(policy="ACT", task=env_name, output_dir=args_cli.output_dir)

    env, video_folder = wrap_video(
        env,
        enabled=args_cli.save_video,
        output_dir=eval_save_dir,
        camera_names=video_camera_names,
        name_prefix="act-eval",
        record_video_cls=RecordVideo,
    )

    eval_overrides = {
        key: value
        for key, value in {
            "n_action_steps": args_cli.n_action_steps,
            "temporal_ensemble_coeff": args_cli.temporal_ensemble_coeff,
            "policy_target_hz": args_cli.policy_target_hz,
        }.items()
        if value is not None
    }
    eval_run = EvalRun(
        output_dir=eval_save_dir,
        policy="ACT",
        task=env_name,
        policy_source=pretrained_dir,
        requested_rollouts=args_cli.num_rollouts,
        num_envs=env.unwrapped.num_envs,
        action_semantics=resolved_action_semantics,
        action_representation=resolved_policy_action_representation,
        metadata={
            **build_common_eval_metadata(args_cli, output_dir=eval_save_dir),
            "Eval overrides": eval_overrides or None,
        },
    )
    eval_run.print_startup()

    def run_batch(batch: EvalBatch) -> bool:
        return eval_policy(
            env,
            policy,
            policy_cfg,
            preprocessor,
            postprocessor,
            state_keys,
            policy_target_hz=args_cli.policy_target_hz,
            action_semantics=resolved_action_semantics,
            policy_action_representation=resolved_policy_action_representation,
            eval_run=eval_run,
            batch=batch,
            progress_every=args_cli.progress_every,
            rollout_seed=batch.seed,
        )

    run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=simulation_app,
        requested_rollouts=args_cli.num_rollouts,
        num_envs=env.unwrapped.num_envs,
        run_batch=run_batch,
        base_seed=args_cli.seed,
        video_folder=video_folder,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
