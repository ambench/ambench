# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, The AM-Bench Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

"""Launch Isaac Sim before importing the rest of the runtime stack."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim through AppLauncher.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Everything below runs after the simulator has been launched."""

import gymnasium as gym
from prettytable import PrettyTable

import ambench.tasks  # noqa: F401


def main() -> None:
    """Print all environments registered in `ambench` extension."""
    # Print all available environments.
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # Set alignment of table columns.
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # Count of environments.
    index = 0

    # Acquire all registered `ambench` environment names.
    for task_spec in gym.registry.values():
        if "Am-" in task_spec.id:
            # Add details to the table.
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    # Run the main function.
    main()
    # Close the simulator app.
    simulation_app.close()
