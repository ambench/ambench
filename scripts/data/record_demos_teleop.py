# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to record demonstrations with Isaac Lab environments using teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored in the LeRobot dataset format. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_root            Root directory path to export recorded demos. (default: "./datasets")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 5)
"""

"""Launch Isaac Sim before importing the rest of the runtime stack."""

import argparse
import contextlib
from pathlib import Path

from _record_cli import add_dataset_export_args, resolve_dataset_output_dir
from isaaclab.app import AppLauncher

# Add argparse arguments before extending them with AppLauncher options.
parser = argparse.ArgumentParser(description="Record teleop demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=0.2, help="Sensitivity factor.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
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
    default=5,
    help="Number of demonstrations to record. Set to 0 for infinite.",
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
parser.add_argument(
    "--env_length_s",
    type=int,
    default=25,
    help="Length of each episode in seconds.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video of the agent.")
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
from collections.abc import Callable

# Third-party imports
import carb
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.devices import (
    DeviceBase,
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from ambench.controllers.control_pipeline import ActionMode
from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator
from ambench.recording import DatasetRecorder
from ambench.recording.paths import write_env_cfg

logger = logging.getLogger(__name__)


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

    def sleep(self, env: gym.Env):
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


def setup_output_directories() -> Path:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist. The directory is used as
    the session root for the recorded LeRobot dataset.

    Returns:
        Full path to the output directory.
    """
    output_dir = resolve_dataset_output_dir(args_cli.task, args_cli.dataset_root)
    if not args_cli.dataset_root:
        print(f"Auto-generated dataset directory: {output_dir}")
    args_cli.dataset_root = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def setup_teleop_device(callbacks: dict[str, Callable]) -> DeviceBase | Se3Keyboard | Se3SpaceMouse | Se3Gamepad:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions attached to
            the teleop device.

    Returns:
        The configured teleoperation device interface.

    Raises:
        RuntimeError: If teleop device creation fails.
    """
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                raise SystemExit(1)

            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        raise RuntimeError("Failed to create teleop device") from e

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        raise RuntimeError("Failed to create teleop interface")

    return teleop_interface


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
        `True` if the recorder should be reset, otherwise `False`.
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


def run_simulation_loop(
    env: gym.Env,
    recorder: DatasetRecorder,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up the teleop device and runs the main loop that processes user
    inputs and environment steps. Records demonstrations when success
    conditions are met.

    Args:
        env: The environment instance
        recorder: The episode recorder instance
        rate_limiter: Optional RateLimiter instance for controlling loop rate
    Returns:
        int: Number of successful demonstrations recorded
    """
    total_recorded_demo_count = 0
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    teleop_interface = setup_teleop_device(callbacks={})
    if isinstance(teleop_interface, Se3Gamepad):
        teleop_interface.add_callback(carb.input.GamepadInput.Y, reset_recording_instance)
    else:
        teleop_interface.add_callback("R", reset_recording_instance)

    # Reset the environment and teleop device before starting.
    obs, _ = env.reset()
    teleop_interface.reset()

    reset_key = "Y button on gamepad" if isinstance(teleop_interface, Se3Gamepad) else "'R'"
    print(f"Teleoperation started. Press {reset_key} to reset the environment.")

    env_unwrapped = env.unwrapped

    # The device reports incremental motion; the environment takes absolute
    # end-effector targets, so the running target is integrated here. The
    # recorded action is the absolute target, matching the scripted recorder.
    ee_target = AbsoluteEETargetIntegrator(env_unwrapped)

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            action = teleop_interface.advance()

            actions = ee_target.advance(action)

            recorder.add_step(obs["policy"], actions)

            obv = env.step(actions)
            obs, reward, terminated, truncated, info = obv

            # Check for success condition.
            reset_needed = process_success_condition(recorder, obv, 0)
            if reset_needed:
                should_reset_recording_instance = True

            # Update the total number of exported successful demonstrations.
            new_total_count = recorder.successful_episode_count
            if new_total_count > total_recorded_demo_count:
                demos_added = new_total_count - total_recorded_demo_count
                total_recorded_demo_count = new_total_count
                print(f"Recorded {demos_added} new demonstrations. Total: {total_recorded_demo_count}")

            # Stop once the requested number of demos has been collected.
            if args_cli.num_demos > 0 and total_recorded_demo_count >= args_cli.num_demos:
                print(f"All {total_recorded_demo_count} demonstrations recorded.\nExiting the app.")
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env_unwrapped)
                    else:
                        env_unwrapped.render()
                break

            # Reset local recorder state if the current episode ended or the user requested a reset.
            if should_reset_recording_instance:
                obs, _ = env_unwrapped.reset()
                ee_target.sync_from_env()
                recorder.reset([0])
                should_reset_recording_instance = False
                print(f"Environment reset.")

            # Stop if the simulator has been stopped externally.
            if env_unwrapped.sim.is_stopped():
                break

            # Maintain the requested stepping rate when configured.
            if rate_limiter:
                rate_limiter.sleep(env_unwrapped)

    return total_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleoperation.

    Main function that orchestrates the full recording flow:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the teleoperation loop to collect demonstrations
    5. Cleans up resources when done
    """
    # Only DirectRLEnv-style tasks are supported in this script.
    if "Direct" not in args_cli.task:
        print("Manager-based environments are not yet supported in this script.")
        return

    rate_limiter = RateLimiter(args_cli.step_hz)

    # Create the output directory for the recording session.
    output_dir_path = setup_output_directories()

    # Create and configure the environment.
    global env_cfg
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        raise SystemExit(1) from e
    env_cfg.episode_length_s = args_cli.env_length_s

    # Create the environment instance.
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise SystemExit(1) from e

    if env.cfg.robot_profile.control.action_mode != ActionMode.ABSOLUTE_EE_POSE:
        logger.error("Teleoperation only supports environments with absolute end-effector actions.")
        env.close()
        simulation_app.close()
        return

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
        video_name_prefix="teleop",
        video_fps=30,
        video_frame_skip=4,
        repo_id=args_cli.repo_id or None,
        state_keys=args_cli.state_keys,
        task_prompt=args_cli.task_prompt or None,
    )

    if args_cli.video:
        fps = 30
        frame_skip = 4
        playback_speed = math.ceil(env.step_dt * fps * frame_skip)
        print(f"[INFO]: Recording video at {fps} FPS with playback speed {playback_speed}x")

    # Run the recording loop.
    total_recorded_demo_count = run_simulation_loop(env, recorder, rate_limiter)

    # Close the environment before post-processing the exported files.
    recorder.close()
    env.close()

    # Print summary statistics when at least one episode was recorded.
    if recorder.episode_steps:

        episode_steps_array = np.array(recorder.episode_steps)
        avg_steps = np.mean(episode_steps_array)
        avg_time_s = avg_steps * env.unwrapped.step_dt

        print(f"\n{'='*80}")
        print(f"Recording session completed with {total_recorded_demo_count} successful demonstrations")
        print(f"Episodes saved to: {output_dir_path}")
        print(f"SESSION_ROOT={output_dir_path}")
        if recorder.canonical_output_dir != output_dir_path:
            print(f"Canonical LeRobot dataset: {recorder.canonical_output_dir}")
        print(f"\nEpisode Statistics:")
        print(f"  Steps per episode: {episode_steps_array.tolist()}")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average time: {avg_time_s:.2f} seconds")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Recording session completed with {total_recorded_demo_count} successful demonstrations")
        print(f"Episodes saved to: {output_dir_path}")
        print(f"SESSION_ROOT={output_dir_path}")
        if recorder.canonical_output_dir != output_dir_path:
            print(f"Canonical LeRobot dataset: {recorder.canonical_output_dir}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run the main function.
    main()
    # Close the simulator app.
    simulation_app.close()
