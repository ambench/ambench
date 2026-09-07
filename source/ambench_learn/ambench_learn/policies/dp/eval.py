# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation script for Diffusion Policy (UMI) on Isaac Lab environments.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

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

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on Isaac Lab environments.")
add_common_eval_args(parser)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Path to the trained Diffusion Policy checkpoint file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
try:
    validate_common_eval_args(args_cli)
except ValueError as error:
    parser.error(str(error))
if args_cli.num_envs != 1:
    parser.error("DP evaluation requires --num-envs 1 because its UMI observation history is not vectorized.")
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required.")
args_cli.checkpoint = args_cli.checkpoint.expanduser().resolve()
if not args_cli.checkpoint.is_file():
    parser.error(f"Checkpoint file not found: {args_cli.checkpoint}")

# Launch simulator.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dill
import gymnasium as gym
import hydra
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from isaaclab_tasks.utils import parse_env_cfg
from omegaconf import OmegaConf
from umi.real_world.real_inference_util import (
    get_real_umi_action,
    get_real_umi_obs_dict,
)

import ambench.tasks  # noqa: F401
from ambench.evaluation import SuccessCriteriaTracker
from ambench.recording import RecordVideo
from ambench.utils.camera_utils import get_camera_rgb
from ambench_learn.policies.dp.eval_utils import (
    ObservationBufferManager,
    extract_robot_state,
    get_real_base_joint_action,
    get_real_base_joint_obs_dict,
    interpolate_action_sequence,
    interpolate_base_joint_action_sequence,
    prepare_umi_image_observation,
    quat_wxyz_to_axis_angle,
    resolve_dp_eval_action_semantics,
    resolve_dp_execution_schedule,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def eval_policy(
    env,
    policy,
    config,
    eval_run: EvalRun,
    batch: EvalBatch,
    progress_every: int,
    rollout_seed: int | None = None,
) -> tuple[dict, bool]:
    shape_meta = config["shape_meta"]
    img_obs_horizon = shape_meta.obs.camera0_rgb.horizon
    low_dim_obs_horizon = next(
        attr.horizon for attr in shape_meta.obs.values() if attr.get("type", "low_dim") == "low_dim"
    )
    action_horizon = shape_meta.action.horizon
    action_mode = shape_meta.action.get("mode", "ee_pose")
    obs_pose_repr = config["obs_pose_repr"]
    action_pose_repr = config["action_pose_repr"]
    obs_down_sample_steps = config["obs_down_sample_steps"]

    # Receding horizon: plan for the full horizon and execute half before re-querying.
    execution_horizon, query_frequency, prediction_horizon_steps = resolve_dp_execution_schedule(
        action_horizon,
        obs_down_sample_steps,
    )
    camera_names = config["camera_names"]
    camera_key_mapping = config.get("camera_key_mapping", {"ee_camera": "camera0_rgb"})

    raw_obs, _ = env.reset(seed=rollout_seed) if rollout_seed is not None else env.reset()
    policy.reset()

    missing_cameras = []
    for camera_name in camera_names:
        try:
            get_camera_rgb(env.unwrapped, camera_name, 0)
        except KeyError as error:
            missing_cameras.append(f"{camera_name}: {error.args[0]}")
    if missing_cameras:
        raise ValueError(
            "DP checkpoint requires camera sensors that are unavailable in this environment: "
            + "; ".join(missing_cameras)
        )

    image_keys = list(camera_key_mapping.values())
    obs_buffer_manager = ObservationBufferManager(img_obs_horizon, low_dim_obs_horizon, image_keys=image_keys)
    action_buffer = []

    ee_pos, ee_quat_wxyz, _, _, base_pos, base_quat_wxyz = extract_robot_state(raw_obs)
    ee_rot_axis_angle = quat_wxyz_to_axis_angle(ee_quat_wxyz)
    base_rot_axis_angle = quat_wxyz_to_axis_angle(base_quat_wxyz)
    episode_start_pose = [np.concatenate([ee_pos, ee_rot_axis_angle])]
    episode_base_start_pose = [np.concatenate([base_pos, base_rot_axis_angle])]

    max_timesteps = env.unwrapped.max_episode_length

    success = False
    executed_steps = 0
    success_criteria = SuccessCriteriaTracker(env_index=0)
    latest_tracking = None
    termination_reason = "timeout"
    stop_evaluation = False
    eval_run.start_rollout(rollout_idx=batch.start, batch_env_index=0)

    for t in range(max_timesteps):
        if not simulation_app.is_running() or simulation_app.is_exiting():
            stop_evaluation = True
            termination_reason = "app_stopped"
            break

        should_update_obs = t % obs_down_sample_steps == 0
        curr_valid_images = {}
        if should_update_obs:
            for camera_name in camera_names:
                if camera_name not in camera_key_mapping:
                    continue

                image = get_camera_rgb(env.unwrapped, camera_name, 0)
                obs_key = camera_key_mapping[camera_name]
                if obs_key in shape_meta.obs:
                    camera_shape = shape_meta.obs[obs_key].shape
                    target_size = (camera_shape[1], camera_shape[2])
                else:
                    target_size = (224, 224)

                curr_valid_images[obs_key] = prepare_umi_image_observation(image, target_size=target_size)

        ee_pos, ee_quat_wxyz, gripper_width, arm_joint_pos, base_pos, base_quat_wxyz = extract_robot_state(raw_obs)
        ee_rot_axis_angle = quat_wxyz_to_axis_angle(ee_quat_wxyz)
        base_rot_axis_angle = quat_wxyz_to_axis_angle(base_quat_wxyz)

        obs_dict = None
        env_obs_stacked = None

        if should_update_obs:
            if curr_valid_images:
                obs_buffer_manager.add_observation(
                    curr_valid_images,
                    ee_pos,
                    ee_rot_axis_angle,
                    gripper_width,
                    robot_joint_pos=arm_joint_pos if "robot0_joint_pos" in shape_meta.obs else None,
                    robot_base_pos=base_pos if "robot0_base_pos" in shape_meta.obs else None,
                    robot_base_rot=base_rot_axis_angle if "robot0_base_rot_axis_angle" in shape_meta.obs else None,
                )

            env_obs_stacked = obs_buffer_manager.get_stacked_observation()

            if env_obs_stacked is not None:
                filtered_env_obs_stacked = {
                    key: value for key, value in env_obs_stacked.items() if key in shape_meta.obs
                }

                robot_prefix = "robot0"
                base_pos_key = robot_prefix + "_base_pos"
                base_rot_key = robot_prefix + "_base_rot_axis_angle"
                effective_episode_base_start_pose = episode_base_start_pose
                if episode_base_start_pose is not None:
                    if base_pos_key not in filtered_env_obs_stacked or base_rot_key not in filtered_env_obs_stacked:
                        # Base keys not available, don't use episode_base_start_pose
                        effective_episode_base_start_pose = None

                if action_mode == "base_joint":
                    obs_dict_np = get_real_base_joint_obs_dict(
                        env_obs=filtered_env_obs_stacked,
                        shape_meta=shape_meta,
                        obs_pose_repr=obs_pose_repr,
                    )
                else:
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=filtered_env_obs_stacked,
                        shape_meta=shape_meta,
                        obs_pose_repr=obs_pose_repr,
                        episode_start_pose=episode_start_pose,
                        episode_base_start_pose=effective_episode_base_start_pose,
                    )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device=env.unwrapped.device),
                )

                for key, attr in shape_meta.obs.items():
                    if attr.get("ignore_by_policy", False):
                        if key in obs_dict:
                            del obs_dict[key]

        should_infer = t % query_frequency == 0
        if should_infer and obs_dict is not None:
            with torch.no_grad():
                res = policy.predict_action(obs_dict)
                raw_action = res["action_pred"][0].detach().to("cpu").numpy()
                if action_mode == "base_joint":
                    umi_action = get_real_base_joint_action(raw_action, env_obs_stacked, action_pose_repr)
                else:
                    umi_action = get_real_umi_action(raw_action, env_obs_stacked, action_pose_repr)

            if action_mode == "base_joint":
                action_buffer = interpolate_base_joint_action_sequence(umi_action, prediction_horizon_steps)
            else:
                action_buffer = interpolate_action_sequence(umi_action, prediction_horizon_steps, output_type="quat")

        if action_mode != "base_joint" and hasattr(env.unwrapped.controller, "reference_from_dp"):
            # Update MPC controller reference trajectory
            # Note: execution_horizon must match the horizon used in MPC
            env.unwrapped.controller.reference_from_dp = True
            mpc_steps = env.unwrapped.controller.N
            action_traj = np.array(action_buffer[:mpc_steps])  # (N, 7)
            env.unwrapped.controller.reference_traj = action_traj

        raw_action = action_buffer.pop(0)
        action = torch.from_numpy(raw_action).float().to(env.unwrapped.device).unsqueeze(0)

        # Expand action to all environments
        if action.shape[0] != env.unwrapped.num_envs:
            actions = action.repeat(env.unwrapped.num_envs, 1)
        else:
            actions = action

        raw_obs, reward, terminated, truncated, info = env.step(actions)
        executed_steps = t + 1
        action_is_nan = bool(torch.isnan(actions).any().item())
        success_criteria.update(info)
        episode_done = bool(terminated[0] or truncated[0]) or action_is_nan
        if not episode_done:
            latest_tracking = eval_run.record_timestep(
                env=env,
                raw_obs=raw_obs,
                rollout_idx=batch.start,
                timestep=executed_steps,
                env_index=0,
                batch_env_index=0,
            )
        eval_run.print_progress(
            rollout_batch=batch.label(),
            step=executed_steps,
            active=(0 if episode_done else 1, 1),
            reward=reward[0].item(),
            tracking_record=latest_tracking,
            tracking_env_index=0,
            progress_every=progress_every,
        )

        if terminated[0]:
            success = True
            termination_reason = "success"
            break

        if action_is_nan:
            termination_reason = "nan_action"
            break

        if truncated[0]:
            termination_reason = "timeout"
            break

    record = eval_run.finish_rollout(
        batch.start,
        batch_env_index=0,
        success=success,
        executed_steps=executed_steps,
        subtask_completion=success_criteria.completion_fraction,
        termination_reason=termination_reason,
        extra=success_criteria.to_record(),
    )
    return record, stop_evaluation


def main():
    """Main evaluation loop."""

    device = args_cli.device
    num_rollouts = args_cli.num_rollouts
    ckpt_path = args_cli.checkpoint

    # Load payload
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    # Restore the trained policy from its Hydra workspace payload.
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.action_pose_repr = cfg.task.pose_repr.action_pose_repr
    policy.to(device)
    policy.eval()

    # Create environment
    env_name = args_cli.task
    num_envs = args_cli.num_envs

    scene_camera_cfg = (
        make_scene_camera_cfg() if args_cli.save_video and "scene_camera" in args_cli.video_camera_names else None
    )
    env_cfg = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task=env_name,
        device=device,
        num_envs=num_envs,
        seed=args_cli.seed,
        episode_length_s=args_cli.episode_length_s,
        disturbance=args_cli.disturbance,
        scene_camera_cfg=scene_camera_cfg,
    )
    action_mode = cfg.task.shape_meta.action.get("mode", "ee_pose")
    action_semantics = resolve_dp_eval_action_semantics(action_mode, env_cfg)

    # Extract config
    shape_meta = cfg.task.shape_meta

    # Determine camera names and mapping keys
    camera_names = []
    camera_key_mapping = {}

    if "camera0_rgb" in shape_meta.obs:
        camera_names.append("ee_camera")
        camera_key_mapping["ee_camera"] = "camera0_rgb"

    if "camera1_rgb" in shape_meta.obs:
        # Assuming camera1 is base_camera if not specified otherwise
        camera_names.append("base_camera")
        camera_key_mapping["base_camera"] = "camera1_rgb"

    if not camera_names:
        raise ValueError(
            "DP checkpoint shape_meta.obs must declare at least one supported RGB observation key: "
            "camera0_rgb or camera1_rgb."
        )

    eval_config = {
        "shape_meta": shape_meta,
        "obs_pose_repr": cfg.task.pose_repr.obs_pose_repr,
        "action_pose_repr": cfg.task.pose_repr.action_pose_repr,
        "obs_down_sample_steps": cfg.task.obs_down_sample_steps,
        "camera_names": camera_names,
        "camera_key_mapping": camera_key_mapping,
    }

    # Create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    video_camera_names = validate_video_camera_names(env, args_cli.video_camera_names) if args_cli.save_video else []

    eval_save_dir = resolve_eval_output_dir(policy="DP", task=env_name, output_dir=args_cli.output_dir)
    eval_run = EvalRun(
        output_dir=eval_save_dir,
        policy="DP",
        task=env_name,
        policy_source=ckpt_path,
        requested_rollouts=num_rollouts,
        num_envs=num_envs,
        action_semantics=action_semantics,
        action_representation=cfg.task.pose_repr.action_pose_repr,
        metadata={
            **build_common_eval_metadata(args_cli, output_dir=eval_save_dir),
            "DP action mode": action_mode,
            "Hydra workspace": cfg._target_,
        },
    )
    eval_run.print_startup()

    video_folder = None
    env, video_folder = wrap_video(
        env,
        enabled=args_cli.save_video,
        output_dir=eval_save_dir,
        camera_names=video_camera_names,
        name_prefix="dp-eval",
        record_video_cls=RecordVideo,
    )

    def run_batch(batch: EvalBatch) -> bool:
        rollout_record, stop_evaluation = eval_policy(
            env,
            policy,
            eval_config,
            eval_run,
            batch=batch,
            progress_every=args_cli.progress_every,
            rollout_seed=batch.seed,
        )
        eval_run.print_rollout_finished(rollout_record, total=num_rollouts)
        return stop_evaluation

    run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=simulation_app,
        requested_rollouts=num_rollouts,
        num_envs=1,
        run_batch=run_batch,
        base_seed=args_cli.seed,
        video_folder=video_folder,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
