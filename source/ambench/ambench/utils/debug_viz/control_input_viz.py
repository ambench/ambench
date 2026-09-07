# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class ControlInputVisualizer:
    """Visualizer for control inputs: force, torque, and motor thrusts.

    This class tracks and visualizes control inputs including forces, torques,
    and motor thrusts for multirotor systems like FAHexa.
    """

    def __init__(self):
        """Initialize the control input visualizer."""
        self.keys = ["force", "torque", "motor_thrusts"]

        # Data storage: force and torque are [x, y, z], motor_thrusts is [num_motors]
        self.force_data = []
        self.torque_data = []
        self.motor_thrusts_data = []

    def add_control_inputs(self, force=None, torque=None, motor_thrusts=None):
        """Add control input data for a single time step.

        Args:
            force: Control force in world frame [3] or [num_envs, 3]. If None, not recorded.
            torque: Control torque in body frame [3] or [num_envs, 3]. If None, not recorded.
            motor_thrusts: Motor thrust commands [num_motors] or [num_envs, num_motors]. If None, not recorded.
        """
        # Extract first environment data if multi-env tensor
        if force is not None:
            if isinstance(force, torch.Tensor):
                force_val = force[0].cpu().numpy() if force.dim() > 1 else force.cpu().numpy()
            else:
                force_val = (
                    force[0] if isinstance(force, (list, np.ndarray)) and len(np.array(force).shape) > 1 else force
                )
            self.force_data.append(force_val)

        if torque is not None:
            if isinstance(torque, torch.Tensor):
                torque_val = torque[0].cpu().numpy() if torque.dim() > 1 else torque.cpu().numpy()
            else:
                torque_val = (
                    torque[0] if isinstance(torque, (list, np.ndarray)) and len(np.array(torque).shape) > 1 else torque
                )
            self.torque_data.append(torque_val)

        if motor_thrusts is not None:
            if isinstance(motor_thrusts, torch.Tensor):
                thrusts_val = motor_thrusts[0].cpu().numpy() if motor_thrusts.dim() > 1 else motor_thrusts.cpu().numpy()
            else:
                thrusts_val = (
                    motor_thrusts[0]
                    if isinstance(motor_thrusts, (list, np.ndarray)) and len(np.array(motor_thrusts).shape) > 1
                    else motor_thrusts
                )
            self.motor_thrusts_data.append(thrusts_val)

    def plot_control_inputs(self, save_dir: str = "."):
        """Plot control input metrics and save to files.

        Creates two separate figures:
        1. Force and Torque (3 subplots each for x, y, z)
        2. Motor Thrusts (all 6 motors in one figure with different colors)

        Args:
            save_dir: Directory to save plots (default: current directory)
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Figure 1: Force and Torque
        if len(self.force_data) > 0 or len(self.torque_data) > 0:
            fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))
            fig1.suptitle("Control Inputs: Force and Torque", fontsize=14, fontweight="bold")

            # Plot Force (x, y, z)
            if len(self.force_data) > 0:
                force_array = np.array(self.force_data)
                if len(force_array) > 1:
                    force_array = force_array[1:, :]  # Skip first point like tracking_viz
                labels = ["x", "y", "z"]
                for dim in range(3):
                    axs1[0, dim].plot(force_array[:, dim], label="Force", color="blue", linewidth=2)
                    axs1[0, dim].set_title(f"Force - {labels[dim]} component")
                    axs1[0, dim].set_xlabel("Time Step")
                    axs1[0, dim].set_ylabel(f"Force {labels[dim]} (N)")
                    axs1[0, dim].grid(True, alpha=0.3)
                    axs1[0, dim].legend()

            # Plot Torque (x, y, z)
            if len(self.torque_data) > 0:
                torque_array = np.array(self.torque_data)
                if len(torque_array) > 1:
                    torque_array = torque_array[1:, :]  # Skip first point
                labels = ["x", "y", "z"]
                for dim in range(3):
                    axs1[1, dim].plot(torque_array[:, dim], label="Torque", color="red", linewidth=2)
                    axs1[1, dim].set_title(f"Torque - {labels[dim]} component")
                    axs1[1, dim].set_xlabel("Time Step")
                    axs1[1, dim].set_ylabel(f"Torque {labels[dim]} (N⋅m)")
                    axs1[1, dim].grid(True, alpha=0.3)
                    axs1[1, dim].legend()

            plt.tight_layout()
            save_name1 = save_path / "control_inputs_force_torque.png"
            plt.savefig(save_name1, dpi=150)
            print(f"Saved force/torque plot to {save_name1}")
            plt.close(fig1)

        # Figure 2: Motor Thrusts (all motors in one figure)
        if len(self.motor_thrusts_data) > 0:
            fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
            fig2.suptitle("Motor Thrusts", fontsize=14, fontweight="bold")

            motor_thrusts_array = np.array(self.motor_thrusts_data)
            if len(motor_thrusts_array) > 1:
                motor_thrusts_array = motor_thrusts_array[1:, :]  # Skip first point

            # Get actual number of motors from data
            num_motors = motor_thrusts_array.shape[1]

            # Define distinct colors for each motor (support up to 8 motors)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

            # Plot each motor with different color
            for motor_id in range(num_motors):
                ax2.plot(
                    motor_thrusts_array[:, motor_id],
                    label=f"Motor {motor_id}",
                    color=colors[motor_id % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                )

            ax2.set_xlabel("Time Step", fontsize=12)
            ax2.set_ylabel("Thrust (N)", fontsize=12)
            ax2.set_title("Motor Thrusts Over Time", fontsize=13)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", ncol=3, fontsize=10)

            plt.tight_layout()
            save_name2 = save_path / "control_inputs_motor_thrusts.png"
            plt.savefig(save_name2, dpi=150)
            print(f"Saved motor thrusts plot to {save_name2}")
            plt.close(fig2)

    def reset(self):
        """Reset all stored data."""
        self.force_data = []
        self.torque_data = []
        self.motor_thrusts_data = []
