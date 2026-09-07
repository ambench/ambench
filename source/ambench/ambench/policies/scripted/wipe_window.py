# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, cast

import isaaclab.sim as sim_utils
from omni.usd import get_context
from pxr import UsdGeom

from .base import BasePolicy, Waypoint

if TYPE_CHECKING:
    from ambench.tasks.wipe_window.wipe_window_env import WipeWindow


class WipeWindowPolicy(BasePolicy):
    """Scripted policy for wipe-window task."""

    def _get_visible_dots(self, env_id: int = 0):
        """Return visible dot indices and positions for one environment."""
        env = cast("WipeWindow", self.env)
        visible_dots = []
        stage = get_context().get_stage()
        num_dots = env.cfg.num_dots

        for i in range(num_dots):
            if i >= len(env.dot_objects):
                continue

            dot_object = env.dot_objects[i]
            dot_prim_paths = sim_utils.find_matching_prim_paths(dot_object.cfg.prim_path)

            if env_id >= len(dot_prim_paths):
                continue

            dot_prim_path = dot_prim_paths[env_id]
            dot_prim = stage.GetPrimAtPath(dot_prim_path)

            if not dot_prim.IsValid():
                continue

            # Check visibility
            imageable = UsdGeom.Imageable(dot_prim)
            visibility_attr = imageable.GetVisibilityAttr()
            is_visible = True
            if visibility_attr and visibility_attr.HasValue():
                if visibility_attr.Get() == UsdGeom.Tokens.invisible:
                    is_visible = False

            if is_visible:
                # Get dot position in world frame
                dot_pos = dot_object.data.root_pos_w[env_id].clone()
                env_origin = env.scene.env_origins[env_id]
                dot_pos = dot_pos - env_origin
                visible_dots.append((i, dot_pos))

        return visible_dots

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for wipe-window task."""
        obs_dict = obs["policy"][env_id]
        ee_pos = obs_dict["ee_pos"]

        init_ee_pose = ee_pos.clone()

        identity_quat = self.identity_quat()

        visible_dots = self._get_visible_dots(env_id=env_id)

        waypoints: list[Waypoint] = []

        waypoints.append(Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=-1.0))

        if len(visible_dots) == 0:
            self.waypoints = waypoints
            self._log_trajectory_summary()
            return

        # Visit each visible stain with a short local wipe stroke. Sorting the dots
        # keeps the path stable instead of hopping around the window randomly.
        visible_dots = sorted(visible_dots, key=lambda dot: (dot[1][2].item(), dot[1][1].item()))
        aim_time = 100
        forward_time = 300
        making_contact_time = 100
        pause_time = 40
        sweep_time = 180
        retreat_time = 100

        dot_positions = [dot_pos for _, dot_pos in visible_dots]
        first_pos = dot_positions[0]
        init_far_pos = first_pos.clone()
        init_far_pos[0] = 0.0
        init_far_pos[1] = 0.0
        init_far_pos[2] = 0.7
        waypoints.append(Waypoint(t=100, xyz=init_far_pos, quat=identity_quat, gripper=-1.0))

        current_time = 100

        # Stand off from the stain itself. The window pose is randomised, so an
        # absolute x put the approach in the wrong place whenever it moved.
        hover_offset_x = 0.40

        for dot_pos in dot_positions:
            far = dot_pos.clone()
            far[0] = dot_pos[0] - hover_offset_x
            far[2] -= 0.1

            close = far.clone()
            close[0] = dot_pos[0] - 0.25
            close[1] += 0.01
            close[2] += 0.1

            contact = close.clone()
            contact[0] = dot_pos[0] - 0.215

            sweep = contact.clone()
            sweep[1] -= 0.04

            current_time += aim_time
            waypoints.append(Waypoint(t=current_time, xyz=far, quat=identity_quat, gripper=-1.0))
            current_time += forward_time
            waypoints.append(Waypoint(t=current_time, xyz=close, quat=identity_quat, gripper=-1.0))
            current_time += making_contact_time
            waypoints.append(Waypoint(t=current_time, xyz=contact, quat=identity_quat, gripper=-1.0))
            current_time += pause_time
            waypoints.append(Waypoint(t=current_time, xyz=contact, quat=identity_quat, gripper=-1.0))
            current_time += sweep_time
            waypoints.append(Waypoint(t=current_time, xyz=sweep, quat=identity_quat, gripper=-1.0))
            current_time += pause_time
            waypoints.append(Waypoint(t=current_time, xyz=sweep, quat=identity_quat, gripper=-1.0))
            current_time += retreat_time
            waypoints.append(Waypoint(t=current_time, xyz=far, quat=identity_quat, gripper=-1.0))

        self.waypoints = waypoints
        self._log_trajectory_summary(f"visible_dots={len(visible_dots)}")
