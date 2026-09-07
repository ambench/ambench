# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted policy for frame assembly task."""

import isaaclab.utils.math as math_utils

from .base import BasePolicy, Waypoint


class FrameAssemblyPolicy(BasePolicy):
    """Scripted policy for frame assembly task.

    The task involves:
    1. Picking up a frame from a table (grasping from upper edge with 90° roll)
    2. Moving the frame to the wall
    3. Aligning and inserting the frame onto 4 pegs on the wall
    """

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for frame assembly task.

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts.
                 Each environment dict has keys like: ee_pos, ee_quat, etc.
            env_id: The environment index.
        """
        env = self.env

        # Extract observation dictionary for specified environment
        obs_dict = obs["policy"][env_id]

        # Use the reset end-effector position in env-origin-relative coordinates.
        init_ee_pose = self._ee_pos_init_local[env_id]

        # Identity quaternion (facing forward, gripper pointing along +X)
        identity_quat = self.identity_quat()

        # Get frame and peg center positions from observations
        frame_pos = obs_dict["frame_pos"].clone()
        peg_center = obs_dict["peg_center_pos"].clone()

        # Get scaled frame dimensions from environment config
        frame_height = env.cfg.frame_height  # Scaled height (Z direction)

        # Calculate upper edge of frame (grasp position)
        # Frame center is at frame_pos, upper edge is at frame_pos + half_height in Z
        grasp_height_offset = frame_height / 2.0  # Half of scaled frame height

        # Add some randomization to approach
        # rand_y = math_utils.sample_uniform(lower=-0.0, upper=0.0, size=1, device=self.device).item()
        rand_z = math_utils.sample_uniform(lower=-0.0, upper=0.0, size=1, device=self.device).item()

        # Waypoint 1: Hover above the frame (approach from above the upper edge)
        hover_above_frame = frame_pos.clone()
        hover_above_frame[0] -= 0.60  # Approach from behind
        hover_above_frame[2] += grasp_height_offset - 0.03  # Above the upper edge

        # Waypoint 2: Descend to grasp approach position (above upper edge)
        grasp_approach = frame_pos.clone()
        grasp_approach[0] -= 0.25  # Close to frame
        grasp_approach[2] += grasp_height_offset - 0.03  # Slightly above upper edge

        # Waypoint 3: Grasp position (at upper edge of frame)
        # Gripper is rotated 90° in roll to grasp the frame edge from the side
        grasp_pos = frame_pos.clone()
        grasp_pos[0] -= 0.16  # Gripper position for grasping
        grasp_pos[2] += grasp_height_offset - 0.03  # At upper edge height

        # Waypoint 4: Lift frame
        lift_pos = grasp_pos.clone()
        lift_pos[2] += 0.20  # Lift up

        lift_far_pos = grasp_pos.clone()
        lift_far_pos[0] = 1.0
        lift_far_pos[1] = 0.0
        lift_far_pos[2] = 1.0

        aim_far_pos = peg_center.clone()
        aim_far_pos[0] = 1.0
        aim_far_pos[2] += grasp_height_offset + rand_z - 0.02  # Align gripper with peg center + offset

        approach_wall_far = aim_far_pos.clone()
        approach_wall_far[0] = peg_center[0] - 0.8

        # Waypoint 5: Move toward wall (intermediate position)
        # Since we're holding the frame from its upper edge, to align the frame center with peg center,
        # the gripper (upper edge) should be at peg_center + grasp_height_offset
        approach_wall = aim_far_pos.clone()
        approach_wall[0] = peg_center[0] - 0.45

        # Waypoint 6: Insert onto pegs
        insert_pos = aim_far_pos.clone()
        insert_pos[0] = peg_center[0] - 0.22

        # Waypoint 9: Release and retract
        retract_pos = aim_far_pos.clone()
        retract_pos[0] = peg_center[0] - 0.60

        # Define waypoints list
        self.waypoints = [
            # Start at current position with gripper open
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
            # Move to the frame
            Waypoint(t=450, xyz=hover_above_frame, quat=identity_quat, gripper=1.0),
            # Descend toward frame upper edge (gripper rotated 90° in roll)
            Waypoint(t=750, xyz=grasp_approach, quat=identity_quat, gripper=1.0),
            # Position at grasp point (upper edge of frame)
            Waypoint(t=900, xyz=grasp_pos, quat=identity_quat, gripper=1.0),
            # Close gripper to grasp frame
            Waypoint(t=950, xyz=grasp_pos, quat=identity_quat, gripper=-1.0),
            Waypoint(t=1000, xyz=grasp_pos, quat=identity_quat, gripper=-1.0),
            # Lift frame up
            Waypoint(t=1100, xyz=lift_pos, quat=identity_quat, gripper=-1.0),
            Waypoint(t=1700, xyz=lift_far_pos, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2400, xyz=approach_wall_far, quat=identity_quat, gripper=-1.0),
            # Move toward wall
            Waypoint(t=2700, xyz=approach_wall, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2750, xyz=approach_wall, quat=identity_quat, gripper=-1.0),
            # Insert frame onto pegs (faster insertion)
            Waypoint(t=3100, xyz=insert_pos, quat=identity_quat, gripper=-1.0),
            # Release frame (open gripper)
            Waypoint(t=3200, xyz=insert_pos, quat=identity_quat, gripper=1.0),
            # Retract from wall
            Waypoint(t=3400, xyz=retract_pos, quat=identity_quat, gripper=1.0),
        ]

        self._log_trajectory_summary()
