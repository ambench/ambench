# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, The AM-Bench Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim before importing the rest of the runtime stack."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# Add argparse arguments before extending them with AppLauncher options.
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=0.2, help="Sensitivity factor.")
# Append AppLauncher CLI args before parsing.
AppLauncher.add_app_launcher_args(parser)
# Parse CLI arguments and enable cameras for runtime compatibility.
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher_args = vars(args_cli)

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# Launch Isaac Sim through AppLauncher.
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Everything below runs after the simulator has been launched."""


import logging

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.devices import (
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab_tasks.utils import parse_env_cfg

import ambench.tasks  # noqa: F401
from ambench.controllers.control_pipeline import ActionMode
from ambench.controllers.teleop_target import AbsoluteEETargetIntegrator
from ambench.tasks.base.base_env import BaseEnv
from ambench.utils.pprint import pprint  # noqa: F401

logger = logging.getLogger(__name__)


def main() -> None:
    """Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.
    """
    # Parse the environment configuration.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.sim.logging_level = "INFO"
    # Extend the episode length for teleoperation sessions.
    env_cfg.episode_length_s = 100000

    if args_cli.xr:
        # External cameras are not supported with XR teleop.
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # Create the environment.
        env: BaseEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    if env.cfg.robot_profile.control.action_mode != ActionMode.ABSOLUTE_EE_POSE:
        logger.error("Teleoperation only supports environments with absolute end-effector actions.")
        env.close()
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow.
    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance() -> None:
        """Request an environment reset on the next simulation step."""
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """Enable application of teleoperation commands."""
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """Disable application of teleoperation commands."""
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    if args_cli.xr:
        # Start inactive for hand tracking devices.
        teleoperation_active = False
    else:
        # Start active for non-hand-tracking devices.
        teleoperation_active = True

    # Create a teleop device from config if present, otherwise create a fallback.
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
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
                env.close()
                simulation_app.close()
                return

            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # Reset the environment before starting teleoperation.
    obs, _ = env.reset()
    teleop_interface.reset()

    # The device reports incremental motion; the environment takes absolute
    # end-effector targets, so the running target is integrated here.
    ee_target = AbsoluteEETargetIntegrator(env)

    print("Teleoperation started. Press 'R' to reset the environment.")

    # Run the simulation loop.
    while simulation_app.is_running():
        try:
            # Run in inference mode.
            with torch.inference_mode():
                action = teleop_interface.advance()

                # Only apply teleop commands when active.
                if teleoperation_active:
                    actions = ee_target.advance(action)
                    obs, reward, terminated, truncated, info = env.step(actions)
                    if terminated.any() or truncated.any():
                        should_reset_recording_instance = True
                    # pprint(obs, reward, terminated, truncated, info=info)

                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    obs, _ = env.reset()
                    ee_target.sync_from_env()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # Close the environment.
    env.close()


if __name__ == "__main__":
    # Run the main function.
    main()
    # Close the simulator app.
    simulation_app.close()
