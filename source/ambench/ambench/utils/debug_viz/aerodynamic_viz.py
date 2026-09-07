# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Visualizer for aerodynamic effects (ground effect and near wall effect).

This module provides visualization tools for tracking aerodynamic effect distances
over time, similar to tracking_viz but specifically for aerodynamic effects.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class AerodynamicEffectsVisualizer:
    """Visualizer for tracking aerodynamic effect distances.

    Tracks ground effect distance (d_ground) and near wall effect distance (d_wall, d_wall_norm)
    over time for multiple motors.
    """

    def __init__(self, num_motors: int = 1, motor_names: list[str] | None = None):
        """Initialize the aerodynamic effects visualizer.

        Args:
            num_motors: Number of motors to track (default: 1)
            motor_names: Optional list of motor names for labeling (default: ["motor1", "motor2", ...])
        """
        self.num_motors = num_motors
        self.motor_names = motor_names or [f"motor{i+1}" for i in range(num_motors)]

        self.keys = ["d_ground", "d_wall"]
        # Disturbance wrench data: [fx, fy, fz, tx, ty, tz] for each effect
        self.disturbance_keys = [
            "drag_force",
            "ground_effect_force",
            "wall_effect_force",
            "drag_torque",
            "ground_effect_torque",
            "wall_effect_torque",
        ]

        self.data = {key: [] for key in self.keys}
        self.disturbance_data = {key: [] for key in self.disturbance_keys}
        self.timesteps = []

    def add_observations(
        self,
        d_ground: torch.Tensor,
        d_wall: torch.Tensor | None = None,
    ):
        """Add observations for tracking.

        Args:
            d_ground: Ground effect distances, shape (num_envs, num_motors)
            d_wall: Optional wall distances, shape (num_envs, num_motors)
        """
        # Use first environment's data for visualization
        d_ground_np = d_ground[0].cpu().numpy()  # (num_motors,)
        self.data["d_ground"].append(d_ground_np)

        if d_wall is not None:
            d_wall_np = d_wall[0].cpu().numpy()  # (num_motors,)
            self.data["d_wall"].append(d_wall_np)

    def add_disturbance_observations(
        self,
        drag_force: torch.Tensor,
        ground_effect_force: torch.Tensor,
        wall_effect_force: torch.Tensor,
        drag_torque: torch.Tensor,
        ground_effect_torque: torch.Tensor,
        wall_effect_torque: torch.Tensor,
        timestep: int,
    ):
        """Add disturbance wrench observations for tracking.

        Args:
            drag_force: Drag force in body frame, shape (num_envs, 3)
            ground_effect_force: Ground effect force in body frame, shape (num_envs, 3)
            wall_effect_force: Wall effect force in body frame, shape (num_envs, 3)
            drag_torque: Drag torque in body frame, shape (num_envs, 3)
            ground_effect_torque: Ground effect torque in body frame, shape (num_envs, 3)
            wall_effect_torque: Wall effect torque in body frame, shape (num_envs, 3)
            timestep: Current timestep
        """
        # Use first environment's data and average across envs
        self.disturbance_data["drag_force"].append({
            "x": drag_force[:, 0].mean().item(),
            "y": drag_force[:, 1].mean().item(),
            "z": drag_force[:, 2].mean().item(),
        })
        self.disturbance_data["ground_effect_force"].append({
            "x": ground_effect_force[:, 0].mean().item(),
            "y": ground_effect_force[:, 1].mean().item(),
            "z": ground_effect_force[:, 2].mean().item(),
        })
        self.disturbance_data["wall_effect_force"].append({
            "x": wall_effect_force[:, 0].mean().item(),
            "y": wall_effect_force[:, 1].mean().item(),
            "z": wall_effect_force[:, 2].mean().item(),
        })
        self.disturbance_data["drag_torque"].append({
            "x": drag_torque[:, 0].mean().item(),
            "y": drag_torque[:, 1].mean().item(),
            "z": drag_torque[:, 2].mean().item(),
        })
        self.disturbance_data["ground_effect_torque"].append({
            "x": ground_effect_torque[:, 0].mean().item(),
            "y": ground_effect_torque[:, 1].mean().item(),
            "z": ground_effect_torque[:, 2].mean().item(),
        })
        self.disturbance_data["wall_effect_torque"].append({
            "x": wall_effect_torque[:, 0].mean().item(),
            "y": wall_effect_torque[:, 1].mean().item(),
            "z": wall_effect_torque[:, 2].mean().item(),
        })
        self.timesteps.append(timestep)

    def plot_metrics(self, save_dir: str = ".", dt: float | None = None):
        """Plot tracking metrics with fixed 2-row layout, expanding horizontally based on motor count.

        Args:
            save_dir: Directory to save plots (default: current directory)
            dt: Time step in seconds. If provided, x-axis will be in seconds instead of time steps.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if len(self.data["d_ground"]) == 0:
            print("[WARNING] No data to plot")
            return

        # 1. Calculate layout (fixed 2 rows)
        num_motors = self.num_motors
        rows = 2
        cols = math.ceil(num_motors / rows)  # Determine columns based on motor count (e.g., 5 motors -> 3 cols)

        # 2. Create figure (expand width as column count increases)
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axs_flat = axs.flatten()

        num_samples = len(self.data["d_ground"])
        time_axis = np.arange(num_samples) * (dt if dt is not None else 1)
        xlabel = "Time (s)" if dt is not None else "Time Step"

        d_ground_array = np.array(self.data["d_ground"])
        d_wall_array = np.array(self.data["d_wall"]) if self.data["d_wall"] else None

        # 3. Plot data for each motor
        for i in range(rows * cols):
            ax = axs_flat[i]

            if i < num_motors:
                # Ground Effect (solid line)
                ax.plot(time_axis, d_ground_array[:, i], color="royalblue", lw=2, label="Ground Effect")

                # Wall Effect (dashed line, only if data exists)
                if d_wall_array is not None:
                    ax.plot(time_axis, d_wall_array[:, i], color="forestgreen", lw=2, ls="--", label="Wall Effect")

                ax.set_title(f"{self.motor_names[i]}", fontsize=12, fontweight="bold")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Distance (m)")
                ax.legend(loc="upper right", fontsize="small")
                ax.grid(True, alpha=0.3, linestyle="--")
            else:
                # Hide empty subplots when motor count is odd
                ax.axis("off")

        plt.tight_layout()

        # 4. Save
        save_name = save_path / "aerodynamic_effects_2_rows.png"
        plt.savefig(save_name, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved aerodynamic effects plots to {save_path.resolve()}")

    def plot_disturbances(self, save_dir: str = ".", dt: float | None = None):
        """Plot 6DOF disturbance wrenches and save to file.

        Args:
            save_dir: Directory to save plots (default: current directory)
            dt: Time step in seconds. If provided, x-axis will be in seconds instead of time steps.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Check if we have data
        if len(self.timesteps) == 0:
            print("[WARNING] No disturbance data to plot")
            return

        # Generate time axis
        timesteps_np = np.array(self.timesteps)
        if dt is not None:
            time_axis = timesteps_np * dt
            xlabel = "Time (s)"
        else:
            time_axis = timesteps_np
            xlabel = "Step"

        # Extract 6DOF components from dictionaries
        drag_fx = np.array([d["x"] for d in self.disturbance_data["drag_force"]])
        drag_fy = np.array([d["y"] for d in self.disturbance_data["drag_force"]])
        drag_fz = np.array([d["z"] for d in self.disturbance_data["drag_force"]])
        ground_fx = np.array([d["x"] for d in self.disturbance_data["ground_effect_force"]])
        ground_fy = np.array([d["y"] for d in self.disturbance_data["ground_effect_force"]])
        ground_fz = np.array([d["z"] for d in self.disturbance_data["ground_effect_force"]])
        wall_fx = np.array([d["x"] for d in self.disturbance_data["wall_effect_force"]])
        wall_fy = np.array([d["y"] for d in self.disturbance_data["wall_effect_force"]])
        wall_fz = np.array([d["z"] for d in self.disturbance_data["wall_effect_force"]])

        drag_tx = np.array([d["x"] for d in self.disturbance_data["drag_torque"]])
        drag_ty = np.array([d["y"] for d in self.disturbance_data["drag_torque"]])
        drag_tz = np.array([d["z"] for d in self.disturbance_data["drag_torque"]])
        ground_tx = np.array([d["x"] for d in self.disturbance_data["ground_effect_torque"]])
        ground_ty = np.array([d["y"] for d in self.disturbance_data["ground_effect_torque"]])
        ground_tz = np.array([d["z"] for d in self.disturbance_data["ground_effect_torque"]])
        wall_tx = np.array([d["x"] for d in self.disturbance_data["wall_effect_torque"]])
        wall_ty = np.array([d["y"] for d in self.disturbance_data["wall_effect_torque"]])
        wall_tz = np.array([d["z"] for d in self.disturbance_data["wall_effect_torque"]])

        # Create figure with 6 subplots (one for each DOF)
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()

        # DOF names and data
        dof_names = [
            "Position X (fx)",
            "Position Y (fy)",
            "Position Z (fz)",
            "Rotation X (tx)",
            "Rotation Y (ty)",
            "Rotation Z (tz)",
        ]
        dof_data = [
            (drag_fx, ground_fx, wall_fx, "Force X (N)"),
            (drag_fy, ground_fy, wall_fy, "Force Y (N)"),
            (drag_fz, ground_fz, wall_fz, "Force Z (N)"),
            (drag_tx, ground_tx, wall_tx, "Torque X (N⋅m)"),
            (drag_ty, ground_ty, wall_ty, "Torque Y (N⋅m)"),
            (drag_tz, ground_tz, wall_tz, "Torque Z (N⋅m)"),
        ]

        # Plot each DOF
        for idx, (ax, dof_name, (drag, ground, wall, ylabel)) in enumerate(zip(axes, dof_names, dof_data)):
            # Plot individual disturbances with different line styles for better visibility
            # Drag: red solid line
            ax.plot(time_axis, drag, "r-", label="Drag", linewidth=2.5, alpha=0.8, zorder=1)
            # Ground Effect: blue dashed line
            ax.plot(time_axis, ground, "b--", label="Ground Effect", linewidth=2.5, alpha=0.8, zorder=2)
            # Wall Effect: green dotted line
            ax.plot(time_axis, wall, "g:", label="Wall Effect", linewidth=2.5, alpha=0.8, zorder=3)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(dof_name)
            ax.legend(loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        plt.tight_layout()
        disturbance_plot_path = save_path / "aerodynamic_disturbances_6dof.png"
        plt.savefig(disturbance_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved aerodynamic disturbance plots to {save_path.resolve()}")

        # Print summary statistics for each DOF
        print("\n[INFO] Aerodynamic Disturbance Summary (6DOF):")
        print("\nPosition Disturbances (Force):")
        print(
            f"  X-axis: Drag={drag_fx.mean():.4f}±{drag_fx.std():.4f} N, "
            f"Ground={ground_fx.mean():.4f}±{ground_fx.std():.4f} N, "
            f"Wall={wall_fx.mean():.4f}±{wall_fx.std():.4f} N, "
            f"Total={(drag_fx+ground_fx+wall_fx).mean():.4f}±{(drag_fx+ground_fx+wall_fx).std():.4f} N"
        )
        print(
            f"  Y-axis: Drag={drag_fy.mean():.4f}±{drag_fy.std():.4f} N, "
            f"Ground={ground_fy.mean():.4f}±{ground_fy.std():.4f} N, "
            f"Wall={wall_fy.mean():.4f}±{wall_fy.std():.4f} N, "
            f"Total={(drag_fy+ground_fy+wall_fy).mean():.4f}±{(drag_fy+ground_fy+wall_fy).std():.4f} N"
        )
        print(
            f"  Z-axis: Drag={drag_fz.mean():.4f}±{drag_fz.std():.4f} N, "
            f"Ground={ground_fz.mean():.4f}±{ground_fz.std():.4f} N, "
            f"Wall={wall_fz.mean():.4f}±{wall_fz.std():.4f} N, "
            f"Total={(drag_fz+ground_fz+wall_fz).mean():.4f}±{(drag_fz+ground_fz+wall_fz).std():.4f} N"
        )
        print("\nRotation Disturbances (Torque):")
        print(
            f"  X-axis: Drag={drag_tx.mean():.4f}±{drag_tx.std():.4f} N⋅m, "
            f"Ground={ground_tx.mean():.4f}±{ground_tx.std():.4f} N⋅m, "
            f"Wall={wall_tx.mean():.4f}±{wall_tx.std():.4f} N⋅m, "
            f"Total={(drag_tx+ground_tx+wall_tx).mean():.4f}±{(drag_tx+ground_tx+wall_tx).std():.4f} N⋅m"
        )
        print(
            f"  Y-axis: Drag={drag_ty.mean():.4f}±{drag_ty.std():.4f} N⋅m, "
            f"Ground={ground_ty.mean():.4f}±{ground_ty.std():.4f} N⋅m, "
            f"Wall={wall_ty.mean():.4f}±{wall_ty.std():.4f} N⋅m, "
            f"Total={(drag_ty+ground_ty+wall_ty).mean():.4f}±{(drag_ty+ground_ty+wall_ty).std():.4f} N⋅m"
        )
        print(
            f"  Z-axis: Drag={drag_tz.mean():.4f}±{drag_tz.std():.4f} N⋅m, "
            f"Ground={ground_tz.mean():.4f}±{ground_tz.std():.4f} N⋅m, "
            f"Wall={wall_tz.mean():.4f}±{wall_tz.std():.4f} N⋅m, "
            f"Total={(drag_tz+ground_tz+wall_tz).mean():.4f}±{(drag_tz+ground_tz+wall_tz).std():.4f} N⋅m"
        )

    def reset(self):
        """Reset all tracked data."""
        self.data = {key: [] for key in self.keys}
        self.disturbance_data = {key: [] for key in self.disturbance_keys}
        self.timesteps = []
