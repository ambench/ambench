# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to record demonstrations with Isaac Lab environments using scripted policies.

This script allows users to record demonstrations operated by scripted policies for a specified task.
The recorded demonstrations are stored in the LeRobot dataset format. Users can specify the task, dataset
directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --dataset_root           Root directory path to export recorded demos. (default: "./datasets")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)

    --inject_noise            Inject noise into the scripted policy actions. (default: False)
    --env_length_s            Length of each episode in seconds. (default: 15)
"""

"""Launch Isaac Sim before importing the rest of the runtime stack."""

import argparse
import contextlib
import importlib
from pathlib import Path

from _record_cli import add_dataset_export_args, resolve_dataset_output_dir
from isaaclab.app import AppLauncher

# Add argparse arguments before extending them with AppLauncher options.
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--dataset_root",
    type=str,
    default="",
    help="Root directory path to export recorded demos. If empty, defaults to repo/datasets.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument(
    "--inject_noise",
    action="store_true",
    default=False,
    help="Inject noise into the scripted policy actions.",
)
parser.add_argument(
    "--env_length_s",
    type=int,
    default=15,
    help="Length of each episode in seconds.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video of the agent.")
parser.add_argument(
    "--save_failed_episodes",
    action="store_true",
    default=False,
    help="Save timeout/failed episodes instead of discarding them.",
)
parser.add_argument(
    "--camera_names",
    type=str,
    nargs="+",
    default=["ee_camera"],
    help=(
        "List of camera names to record images from. Default is ['ee_camera']. "
        "Add other cameras (e.g. 'base_camera') explicitly if they exist for the selected task."
    ),
)
add_dataset_export_args(parser)

# Append AppLauncher CLI args before parsing.
AppLauncher.add_app_launcher_args(parser)

# Parse CLI arguments and enable cameras for dataset recording.
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# Launch Isaac Sim through AppLauncher.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator has been launched."""


import logging
import math
import time

# Third-party imports
import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from ambench.controllers.control_pipeline import ActionMode
from ambench.controllers.pyroki_ik_ctrl import PyrokiIKController, ik_compute
from ambench.policies.scripted.base import BasePolicy
from ambench.recording import DatasetRecorder
from ambench.recording.paths import write_env_cfg
from ambench.utils.camera_utils import compute_camera_quat_from_lookat

logger = logging.getLogger(__name__)

BASE_JOINT_ABSOLUTE = "base_joint_absolute"
SCENE_CAMERA_POS = (-2.0, -2.5, 1.6)
SCENE_CAMERA_LOOKAT = (2.0, 0.0, 1.0)


def make_scene_camera_cfg() -> CameraCfg:
    """Create the optional external scene camera used for side-view recording."""
    return CameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=SCENE_CAMERA_POS,
            rot=compute_camera_quat_from_lookat(SCENE_CAMERA_POS, SCENE_CAMERA_LOOKAT),
            convention="ros",
        ),
    )


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in Hertz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # Catch up if the loop falls behind the requested rate.
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class BaseJointActionAdapter:
    """Convert scripted EE targets into absolute Base+joints actions."""

    def __init__(self, env) -> None:
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        control_cfg = self.env.cfg.robot_profile.control
        self.enabled = control_cfg.action_mode == ActionMode.ABSOLUTE_BASE_JOINTS
        self.ik_controller = None
        if self.enabled:
            if control_cfg.ik is None:
                raise AttributeError("Base+joints scripted recording requires robot_profile.control.ik.")
            self.ik_controller = PyrokiIKController(
                num_envs=self.env.num_envs,
                device=self.env.device,
                config=control_cfg.ik,
            )

    def convert(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return actions
        if self.ik_controller is None:
            raise RuntimeError("BaseJointActionAdapter is enabled without an IK controller.")
        if actions.shape[-1] != 8:
            raise ValueError(f"Expected scripted absolute EE actions with shape (N, 8). Got {tuple(actions.shape)}.")

        target_pos_w = actions[:, 0:3] + self.env.scene.env_origins
        target_quat = actions[:, 3:7]
        base_joint_actions = ik_compute(
            ik_controller=self.ik_controller,
            env=self.env,
            target_pos=target_pos_w,
            target_quat=target_quat,
        )
        # IK returns world-frame base positions; no-IK absolute env actions expect env-origin-relative positions.
        base_joint_actions[:, 0:3] -= self.env.scene.env_origins
        base_joint_actions[:, -1] = actions[:, -1]
        return base_joint_actions


def setup_output_directories() -> Path:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist. The directory is used as
    the session root for the recorded LeRobot dataset.

    Returns:
        Full path to the output directory.
    """
    output_dir = resolve_dataset_output_dir(args_cli.task, args_cli.dataset_root)

    print(f"Dataset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def process_success_condition(
    recorder: DatasetRecorder,
    obv: tuple,
    env_id: int,
) -> bool:
    """Process the success condition for the current step for a specific environment.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        recorders: List of episode recorder instances (one per environment)
        obv: Observation tuple from env.step() (obs, reward, terminated, truncated, info)
        env_id: The environment ID to process

    Returns:
        `True` if the recorder and policy state should be reset, otherwise `False`.
    """
    _, _, terminated, truncated, _ = obv
    if terminated[env_id]:
        print(f"✔️ [Env {env_id}] Episode terminated (success)")
        recorder.finish_episode(env_id, success=True)
        return True

    if truncated[env_id]:
        print(f"❌ [Env {env_id}] Episode truncated (timeout without success)")
        recorder.finish_episode(env_id, success=False)
        return True

    return False


def handle_reset(
    env,
    recorder: DatasetRecorder,
    policies: list[BasePolicy],
    env_ids: list[int],
):
    """Reset recorder and policy state for specific environments.

    DirectRLEnv automatically resets environments internally when they
    terminate or truncate. This function only resets local recorder and
    policy state tracking.

    Args:
        env: The environment instance
        recorders: List of recorders (one per environment)
        policies: List of policy instances (one per environment)
        env_ids: List of environment IDs to reset
    """
    for env_id in env_ids:
        recorder.reset([env_id])
        policies[env_id].reset()


def run_simulation_loop(
    env,
    recorder: DatasetRecorder,
    PolicyClass: type[BasePolicy],
    rate_limiter: RateLimiter | None,
    num_envs: int,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Runs the main loop that executes scripted policy actions and environment
    steps. Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        recorder: Dataset recorder instance
        PolicyClass: Scripted policy class to instantiate per environment
        rate_limiter: Optional rate limiter to control simulation speed
        num_envs: Number of parallel environments

    Returns:
        int: Number of successful demonstrations recorded
    """
    total_recorded_demo_count = 0

    # Reset all environments at the start of recording.
    print("[INFO]: Resetting environments before scripted recording loop...", flush=True)
    obs, _ = env.reset()
    print("[INFO]: Initial environment reset complete.", flush=True)

    policies: list[BasePolicy] = []
    # Instantiate one policy per environment for independent state tracking.
    for env_id in range(num_envs):
        policy = PolicyClass(env=env, inject_noise=args_cli.inject_noise)
        policies.append(policy)

    env_unwrapped = env.unwrapped
    action_adapter = BaseJointActionAdapter(env_unwrapped)
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        loop_step_count = 0
        while simulation_app.is_running():
            loop_step_count += 1
            if loop_step_count == 1 or loop_step_count % 600 == 0:
                print(f"[INFO]: Scripted recorder loop step {loop_step_count}", flush=True)
            # Get actions from each policy independently.
            actions_list = []
            for env_id in range(num_envs):
                action = policies[env_id].advance(obs, env_id)
                actions_list.append(action)

            # Stack actions from all policies.
            actions = torch.cat(actions_list, dim=0)
            actions = action_adapter.convert(actions)

            recorder.add_step(obs["policy"], actions)
            obv = env.step(actions)
            obs, reward, terminated, truncated, info = obv
            # Handle termination and truncation for each environment.
            envs_to_reset = []
            for env_id in range(num_envs):
                reset_needed = process_success_condition(recorder, obv, env_id)
                if reset_needed:
                    envs_to_reset.append(env_id)

            # Reset recorder and policy state for environments that finished.
            if len(envs_to_reset) > 0:
                handle_reset(env_unwrapped, recorder, policies, envs_to_reset)
                print(f"Episode reset complete for envs {envs_to_reset}, continuing...")

            # Update the total number of exported successful demonstrations.
            new_total_count = recorder.successful_episode_count
            if new_total_count > total_recorded_demo_count:
                demos_added = new_total_count - total_recorded_demo_count
                total_recorded_demo_count = new_total_count
                print(f"Recorded {demos_added} new demonstrations. Total: {total_recorded_demo_count}")

            # Stop once the requested number of demos has been collected.
            if args_cli.num_demos > 0 and total_recorded_demo_count >= args_cli.num_demos:
                print(f"All {total_recorded_demo_count} demonstrations recorded.\nExiting the app.")
                break

            # Stop if the simulator has been stopped externally.
            if env.unwrapped.sim.is_stopped():
                break

            # Maintain the requested stepping rate when configured.
            if rate_limiter:
                rate_limiter.sleep(env.unwrapped)

    return total_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using scripted policies.

    Main function that orchestrates the full recording flow:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Creates episode recorder
    5. Creates scripted policy
    6. Runs the simulation loop to collect demonstrations
    7. Cleans up resources when done
    """
    # Only DirectRLEnv-style tasks are supported in this script.
    if "Direct" not in args_cli.task:
        print("Manager-based environments are not yet supported in this script.")
        return

    # Set up the rate limiter.
    rate_limiter = RateLimiter(args_cli.step_hz)

    # Create the output directory for the recording session.
    output_dir_path = setup_output_directories()

    # Create and configure the environment.
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
        # env_cfg.sim.logging_level = "DEBUG"
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        raise SystemExit(1) from e
    env_cfg.episode_length_s = args_cli.env_length_s

    # Ensure rendering happens during internal resets to prevent stale observations.
    if hasattr(env_cfg, "num_rerenders_on_reset"):
        env_cfg.num_rerenders_on_reset = 1
    if "scene_camera" in args_cli.camera_names:
        env_cfg.scene_camera_cfg = make_scene_camera_cfg()
        print(f"[INFO]: Added scene_camera at {SCENE_CAMERA_POS} looking at {SCENE_CAMERA_LOOKAT}")

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise SystemExit(1) from e

    # Save the resolved environment configuration alongside the dataset.
    # This captures overrides such as device, num_envs, and episode_length_s.
    try:
        write_env_cfg(output_dir_path, env_cfg)
    except Exception as e:
        logger.warning(f"Failed to write env_cfg.yaml to '{output_dir_path}': {e}")

    recorder = DatasetRecorder(
        env=env,
        output_dir=output_dir_path,
        env_cfg=env_cfg,
        camera_names=args_cli.camera_names,
        record_video=args_cli.video,
        video_name_prefix="scripted",
        video_fps=30,
        video_frame_skip=4,
        repo_id=args_cli.repo_id or None,
        state_keys=args_cli.state_keys,
        task_prompt=args_cli.task_prompt or None,
        save_failed_episodes=args_cli.save_failed_episodes,
    )
    print(f"Created dataset recorder for {env.num_envs} parallel environments")

    # Load the scripted policy entry point declared by the environment spec.
    PolicyClass: type[BasePolicy] | None = None
    try:
        env_spec = env.spec
        if hasattr(env_spec, "kwargs") and "scripted_policy_entry_point" in env_spec.kwargs:
            policy_entry_point = env_spec.kwargs["scripted_policy_entry_point"]
            module_path, class_name = policy_entry_point.split(":")
            module = importlib.import_module(module_path)
            PolicyClass = getattr(module, class_name)
    except Exception as e:
        print(f"Error loading scripted policy entry point: {e}")
        env.close()
        return

    if PolicyClass is None:
        print("Error: No scripted_policy_entry_point found for this environment.")
        env.close()
        return

    if args_cli.video:
        fps = 30
        frame_skip = 4
        playback_speed = math.ceil(env.step_dt * fps * frame_skip)
        print(f"[INFO]: Recording video at {fps} FPS with playback speed {playback_speed}x")

    # Run the recording loop.
    total_recorded_demo_count = run_simulation_loop(env, recorder, PolicyClass, rate_limiter, env.num_envs)

    # Emit explicit stage markers so pipeline logs show where shutdown fails.
    print("Finalizing dataset recorder...")
    recorder.close()
    print("Dataset recorder finalized.")

    print("Closing environment...")
    env.close()
    print("Environment closed.")

    print(f"\n{'='*80}")
    print(f"Recording session completed with {total_recorded_demo_count} successful demonstrations")
    print(f"Episodes saved to: {output_dir_path}")
    print(f"SESSION_ROOT={output_dir_path}")
    if recorder.canonical_output_dir != output_dir_path:
        print(f"Canonical LeRobot dataset: {recorder.canonical_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        # Run the main function.
        main()
    finally:
        print("Closing simulation app...")
        simulation_app.close()
        print("Simulation app closed.")
