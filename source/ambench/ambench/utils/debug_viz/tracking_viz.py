# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class TargetPoseVisualizer:
    """Runtime marker visualizer for desired target poses and scripted waypoints."""

    def __init__(self, env, show_markers=False):
        # TODO: consider moving visualization marker setup to the env class (debug_viz).
        self.env = env.unwrapped
        self.show_markers = show_markers

        self.has_base_tracking = bool(self.env.cfg.robot_profile.robot.arm_joint_names)

        self.ee_target_visualizer = None
        self.base_target_visualizer = None
        self.waypoint_visualizer = None
        if not self.show_markers:
            return

        ee_target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ee_target_trajectory",
            markers={
                "ee_frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
            },
        )
        self.ee_target_visualizer = VisualizationMarkers(ee_target_marker_cfg)

        if self.has_base_tracking:
            base_target_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/base_target_trajectory",
                markers={
                    "base_frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.1, 0.1, 0.1),
                    ),
                },
            )
            self.base_target_visualizer = VisualizationMarkers(base_target_marker_cfg)

    def visualize_targets(self, desired_ee_pos: torch.Tensor, desired_ee_quat: torch.Tensor, x_d: torch.Tensor) -> None:
        """Visualize current desired EE and base targets when marker visualization is enabled."""
        if not self.show_markers or self.ee_target_visualizer is None:
            return

        desired_ee_pos = desired_ee_pos[0]
        desired_ee_quat = desired_ee_quat[0]
        desired_base_pos = x_d[:, 0:3][0]
        desired_base_quat = x_d[:, 3:7][0]

        self.ee_target_visualizer.visualize(
            translations=desired_ee_pos.unsqueeze(0),
            orientations=desired_ee_quat.unsqueeze(0),
        )
        if self.base_target_visualizer is not None:
            self.base_target_visualizer.visualize(
                translations=desired_base_pos.unsqueeze(0),
                orientations=desired_base_quat.unsqueeze(0),
            )

    def visualize_waypoints(self, waypoints: list[dict], env_origin: torch.Tensor | None = None):
        """Visualize scripted-policy waypoints as frame markers."""
        if not self.show_markers:
            return

        if not waypoints:
            print("[TargetPoseVisualizer] No waypoints to visualize.")
            return

        if self.waypoint_visualizer is None:
            waypoint_marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/waypoint_markers",
                markers={
                    "waypoint_frame": sim_utils.UsdFileCfg(
                        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                        scale=(0.05, 0.05, 0.05),
                    ),
                },
            )
            self.waypoint_visualizer = VisualizationMarkers(waypoint_marker_cfg)

        positions = []
        orientations = []
        for waypoint in waypoints:
            position = waypoint["xyz"].clone()
            if env_origin is not None:
                position = position + env_origin
            positions.append(position)
            orientations.append(waypoint["quat"].clone())

        self.waypoint_visualizer.visualize(
            translations=torch.stack(positions, dim=0),
            orientations=torch.stack(orientations, dim=0),
        )

        print(f"[TargetPoseVisualizer] Visualized {len(waypoints)} waypoints.")
