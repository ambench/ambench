# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Evaluate served OpenPI AM bench policies on Isaac Lab tasks."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import collections
import socket
from urllib.parse import urlparse

from isaaclab.app import AppLauncher

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

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate OpenPI policies zero-shot on Isaac Lab environments.")
add_common_eval_args(parser)
parser.add_argument("--prompt", type=str, default=None, help="Language instruction prompt (required).")
parser.add_argument("--host", type=str, default="localhost", help="Host running the remote OpenPI policy server.")
parser.add_argument("--port", type=int, default=8000, help="Port used by the remote OpenPI policy server.")
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=8,
    help="Number of low-rate policy actions to execute before replanning.",
)
parser.add_argument(
    "--policy-target-hz",
    type=int,
    default=None,
    help="Policy action rate. If set below env rate, each low-rate action is expanded to env-rate actions.",
)
parser.add_argument(
    "--policy-id",
    type=str,
    default=None,
    help="Optional human-readable identifier for the policy served at --host/--port.",
)
parser.add_argument(
    "--connect-timeout-s",
    type=float,
    default=10.0,
    help="Maximum time to wait for the policy server TCP connection before starting Isaac Sim.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
try:
    validate_common_eval_args(args_cli)
except ValueError as error:
    parser.error(str(error))
if not args_cli.prompt:
    parser.error("--prompt is required.")
if args_cli.n_action_steps < 1:
    parser.error(f"--n-action-steps must be >= 1, got {args_cli.n_action_steps}")
if args_cli.policy_target_hz is not None and args_cli.policy_target_hz < 1:
    parser.error(f"--policy-target-hz must be >= 1, got {args_cli.policy_target_hz}")
if args_cli.connect_timeout_s <= 0:
    parser.error(f"--connect-timeout-s must be > 0, got {args_cli.connect_timeout_s}")

# Fail before launching Isaac Sim when the configured policy server cannot be reached.
parsed_host = urlparse(args_cli.host if "://" in args_cli.host else f"//{args_cli.host}")
socket_host = parsed_host.hostname or args_cli.host
socket_port = parsed_host.port or args_cli.port
try:
    with socket.create_connection((socket_host, socket_port), timeout=args_cli.connect_timeout_s):
        pass
except OSError as error:
    parser.error(f"Could not connect to OpenPI server at {socket_host}:{socket_port}: {error}")

# Launch the simulator.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from typing import Any

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from openpi_client import websocket_client_policy

import ambench.tasks  # noqa: F401
from ambench.evaluation import SuccessCriteriaTracker
from ambench.recording import RecordVideo
from ambench_learn.data.action_resampling import compute_stride
from ambench_learn.data.action_semantics import (
    resolve_eval_action_semantics as _resolve_eval_action_semantics,
)
from ambench_learn.policies.pi.eval_utils import (
    build_openpi_example,
    default_action_for_env,
    expand_policy_actions,
)

# set numpy print options
np.set_printoptions(precision=4, suppress=True, floatmode="fixed")


def _run_rollout_batch(
    args: argparse.Namespace,
    client: Any,
    env: Any,
    eval_run: EvalRun,
    batch: EvalBatch,
    action_semantics: str,
    max_timesteps: int,
) -> bool:
    num_envs = env.unwrapped.num_envs
    env_indices = list(range(num_envs))
    tracked_env_indices = list(range(batch.size))
    if batch.seed is None:
        raw_obs, _ = env.reset()
    else:
        raw_obs, _ = env.reset(seed=batch.seed)

    action_plans: list[collections.deque[torch.Tensor]] = [collections.deque() for _ in env_indices]
    success = [False for _ in env_indices]
    done = [env_index not in tracked_env_indices for env_index in env_indices]
    executed_steps = [0 for _ in env_indices]
    termination_reasons = ["not_evaluated" if done[env_index] else "timeout" for env_index in env_indices]
    success_criteria = [SuccessCriteriaTracker(env_index=env_index) for env_index in env_indices]
    latest_tracking: list[dict[str, Any] | None] = [None for _ in env_indices]
    for env_index in tracked_env_indices:
        eval_run.start_rollout(rollout_idx=batch.start + env_index, batch_env_index=env_index)

    stop_evaluation = False
    for step in range(max_timesteps):
        if not simulation_app.is_running() or simulation_app.is_exiting():
            stop_evaluation = True
            for env_index in tracked_env_indices:
                if not done[env_index]:
                    termination_reasons[env_index] = "app_stopped"
            break
        if all(done):
            break

        # Refill per-environment plans as needed and assemble the next action batch.
        action_rows = []
        for env_index in env_indices:
            if done[env_index]:
                action_row = default_action_for_env(
                    env,
                    action_semantics=action_semantics,
                    env_index=env_index,
                )
            else:
                if not action_plans[env_index]:
                    example = build_openpi_example(
                        raw_obs,
                        env,
                        env_index=env_index,
                        action_semantics=action_semantics,
                        prompt=args.prompt,
                    )
                    output = client.infer(example)
                    actions = np.asarray(output["actions"], dtype=np.float32).copy()
                    action_plans[env_index] = expand_policy_actions(
                        actions,
                        env,
                        action_semantics=action_semantics,
                        n_action_steps=int(args.n_action_steps),
                        action_execution_stride=int(args.action_execution_stride),
                        env_index=env_index,
                    )
                action_row = action_plans[env_index].popleft()
            action_rows.append(action_row.reshape(env.action_space.shape[-1]))
        actions_env = torch.stack(action_rows, dim=0).to(device=env.unwrapped.device, dtype=torch.float32)

        raw_obs, reward, terminated, truncated, info = env.step(actions_env)
        actions_are_nan = torch.isnan(actions_env).flatten(start_dim=1).any(dim=1)
        stepped_env_indices = [env_index for env_index in tracked_env_indices if not done[env_index]]

        # Update per-environment tracking and completion state.
        for env_index in tracked_env_indices:
            if done[env_index]:
                continue

            executed_steps[env_index] += 1
            success_criteria[env_index].update(info)
            episode_done = (
                bool(terminated[env_index].item())
                or bool(truncated[env_index].item())
                or bool(actions_are_nan[env_index].item())
            )
            if eval_run.tracking_enabled and not episode_done:
                latest_tracking[env_index] = eval_run.record_timestep(
                    env=env,
                    raw_obs=raw_obs,
                    rollout_idx=batch.start + env_index,
                    timestep=executed_steps[env_index],
                    env_index=env_index,
                    batch_env_index=env_index,
                )

            if bool(terminated[env_index].item()):
                success[env_index] = True
                done[env_index] = True
                termination_reasons[env_index] = "success"
            elif bool(actions_are_nan[env_index].item()):
                done[env_index] = True
                termination_reasons[env_index] = "nan_action"
            elif bool(truncated[env_index].item()):
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
            step=step + 1,
            active=(sum(not done[index] for index in tracked_env_indices), batch.size),
            reward=mean_reward,
            tracking_record=latest_tracking[representative_env],
            tracking_env_index=representative_env,
            progress_every=args.progress_every,
        )

    # Finalize every rollout represented by this vectorized batch.
    for env_index in tracked_env_indices:
        rollout_idx = batch.start + env_index
        rollout_record = eval_run.finish_rollout(
            rollout_idx,
            batch_env_index=env_index,
            success=success[env_index],
            executed_steps=executed_steps[env_index],
            subtask_completion=success_criteria[env_index].completion_fraction,
            termination_reason=termination_reasons[env_index],
            extra=success_criteria[env_index].to_record(),
        )
        eval_run.print_rollout_finished(rollout_record, total=args.num_rollouts)
    return stop_evaluation


def main() -> None:
    args = args_cli
    client = websocket_client_policy.WebsocketClientPolicy(host=socket_host, port=socket_port)
    server_metadata = client.get_server_metadata()
    run_dir = resolve_eval_output_dir(policy="PI", task=args.task, output_dir=args.output_dir)

    scene_camera_cfg = (
        make_scene_camera_cfg() if args.save_video and "scene_camera" in args.video_camera_names else None
    )
    env_cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=args.task,
        device=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        episode_length_s=args.episode_length_s,
        disturbance=args.disturbance,
        scene_camera_cfg=scene_camera_cfg,
    )
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env_fps = round(1.0 / float(env.unwrapped.dt))
    args.action_execution_stride = 1
    if args.policy_target_hz is not None:
        args.action_execution_stride = compute_stride(env_fps, int(args.policy_target_hz))
    action_semantics = _resolve_eval_action_semantics(env)
    video_camera_names = validate_video_camera_names(env, args.video_camera_names) if args.save_video else []

    # Record the resolved policy and execution settings with the evaluation.
    eval_overrides = {
        key: value
        for key, value in {
            "n_action_steps": args.n_action_steps,
            "policy_target_hz": args.policy_target_hz,
            "action_execution_stride": args.action_execution_stride,
        }.items()
        if value is not None
    }
    eval_run = EvalRun(
        output_dir=run_dir,
        policy="PI",
        task=args.task,
        policy_source=f"ws://{socket_host}:{socket_port}",
        requested_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        action_semantics=action_semantics,
        metadata={
            **build_common_eval_metadata(args, output_dir=run_dir),
            "Policy ID": args.policy_id,
            "Prompt": args.prompt,
            "Server metadata": server_metadata,
            "Eval overrides": eval_overrides or None,
        },
    )
    eval_run.print_startup()
    env, video_folder = wrap_video(
        env,
        enabled=args.save_video,
        output_dir=run_dir,
        camera_names=video_camera_names,
        name_prefix="pi-eval",
        record_video_cls=RecordVideo,
    )
    max_timesteps = env.unwrapped.max_episode_length

    def run_batch(batch: EvalBatch) -> bool:
        return _run_rollout_batch(args, client, env, eval_run, batch, action_semantics, max_timesteps)

    run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=simulation_app,
        requested_rollouts=args.num_rollouts,
        num_envs=env.unwrapped.num_envs,
        run_batch=run_batch,
        base_seed=args.seed,
        video_folder=video_folder,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
