# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, The AM-Bench Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim before importing the rest of the runtime stack."""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments before extending them with AppLauncher options.
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the agent.")
parser.add_argument(
    "--camera_names",
    type=str,
    nargs="+",
    default=["ee_camera"],
    help=(
        "List of camera names to record images from. Default is ['ee_camera']. "
        "Add other cameras explicitly if they exist for the selected task."
    ),
)
# Append AppLauncher CLI args before parsing.
AppLauncher.add_app_launcher_args(parser)
# Parse CLI arguments and enable cameras for runtime compatibility.
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Launch Isaac Sim through AppLauncher.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after the simulator has been launched."""

import math
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg

import ambench.tasks  # noqa: F401
from ambench.recording import RecordVideo
from ambench.recording.paths import resolve_video_dir, task_id_to_output_name
from ambench.tasks.base.base_env import BaseEnv
from ambench.utils.pprint import pprint


def main() -> None:
    """Zero actions agent with Isaac Lab environment."""
    # Parse the environment configuration.
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.sim.logging_level = "INFO"
    # Create the environment.
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped: BaseEnv = env.unwrapped

    if args_cli.video:
        video_folder = resolve_video_dir(
            task_id_to_output_name(args_cli.task),
            root_dir=(Path(__file__).resolve().parent / "../../videos").resolve(),
        )
        fps = 30
        frame_skip = 8
        # The simulator runs at 120 Hz, so frame_skip=8 saves every eighth frame.
        playback_speed = math.ceil(env_unwrapped.step_dt * fps * frame_skip)
        print(f"[INFO]: Recording video at {fps} FPS with playback speed {playback_speed}x")
        env = RecordVideo(
            env,
            video_folder=str(video_folder),
            camera_names=args_cli.camera_names,
            name_prefix="zero_agent",
            fps=fps,
            frame_skip=frame_skip,
            output_format="mp4",
        )
    else:
        video_folder = None

    # Print environment information.
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # Reset the environment before stepping.
    env.reset()
    action_shape = env.action_space.shape
    if action_shape is None:
        raise RuntimeError("Action space shape is required for zero_agent.")

    # Run the simulation loop.
    while simulation_app.is_running():
        # Run in inference mode.
        with torch.inference_mode():
            # Compute zero actions.
            init_state = env_unwrapped.robot.cfg.init_state
            init_pos = torch.tensor(init_state.pos, device=env_unwrapped.device)
            init_quat = torch.tensor(init_state.rot, device=env_unwrapped.device)
            action = torch.zeros(action_shape, device=env_unwrapped.device)
            action[..., 0:3] = init_pos
            action[..., 3:7] = init_quat
            action[..., -1] = 1.0
            obs, reward, terminated, truncated, info = env.step(action)
            pprint(obs, reward, terminated, truncated, decimals=4)

    # Close the environment.
    env.close()
    if video_folder is not None:
        print(f"VIDEO_PATH={video_folder.resolve()}")


if __name__ == "__main__":
    # Run the main function.
    main()
    # Close the simulator app.
    simulation_app.close()
