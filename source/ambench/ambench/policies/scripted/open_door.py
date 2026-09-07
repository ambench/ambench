# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Scripted policy for open door task."""

from .base import BasePolicy, Waypoint


class OpenDoorPolicy(BasePolicy):
    """Scripted policy for open door task.

    The task involves:
    1. Approaching the door
    2. Pushing the door open
    3. Moving through the door
    """

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for open door task.

        Args:
            obs: Observation dict with "policy" key containing list of per-environment dicts.
                 Each environment dict includes ee/base state plus door_joint_pos and door_pos.
            env_id: The environment index.
        """
        env = self.env

        # Extract observation dictionary for specified environment
        obs_dict = obs["policy"][env_id]

        init_ee_pose = self._ee_pos_init_local[env_id]

        # Identity quaternion (facing forward, gripper pointing along +X)
        identity_quat = self.identity_quat()

        door_position = obs_dict["door_pos"].clone()
        door_pass_dist = env.cfg.door_pass_dist

        door_contact_pos = door_position.clone()
        door_contact_pos[1] -= 0.25
        door_contact_pos[2] += 1.2

        # Waypoint 1: Approach position (in front of door, aligned with handle)
        approach_dist = 0.4  # Distance in front of door
        approach_pos = door_contact_pos.clone()
        approach_pos[0] -= approach_dist  # Stay in front of door

        # Waypoint 2: Contact position (at door surface, ready to push)
        contact_dist = 0.15  # Close to door surface
        contact_pos = door_contact_pos.clone()
        contact_pos[0] -= contact_dist

        # Waypoint 3: Push position (push the door open while moving forward)
        # Push diagonally forward and to the side to swing door open
        push_pos = door_contact_pos.clone()
        push_pos[0] += 0.4  # Move forward past door
        push_pos[1] += 0.4  # Move sideways as door swings

        # Waypoint 4: Through position (past the door threshold)
        through_pos = push_pos.clone()
        through_pos[0] += 0.2  # Past the door

        # Waypoint 5: Avoid door position (move slightly in -y to avoid collision with door)
        avoid_pos = through_pos.clone()
        avoid_pos[1] -= 0.3  # Move in -y direction to avoid door

        # Waypoint 6: Forward avoid position (move slightly forward in +x)
        forward_avoid_pos = avoid_pos.clone()
        forward_avoid_pos[0] += 0.3  # Move slightly forward in +x

        # Waypoint 7: Diagonal avoid position (move further in +x, -y to avoid door)
        diagonal_avoid_pos = forward_avoid_pos.clone()
        diagonal_avoid_pos[0] += 0.8  # Move in +x direction
        diagonal_avoid_pos[1] -= 0.8  # Move further in -y direction

        # Waypoint 8: Final position (fully through door)
        final_pos = diagonal_avoid_pos.clone()
        final_pos[0] += door_pass_dist + 1.0  # Well past the door

        # Define waypoints list
        self.waypoints = [
            # Start at current position with gripper closed (to push door)
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=-1.0),  # Closed gripper for pushing
            # Move to contact position
            Waypoint(t=300, xyz=contact_pos, quat=identity_quat, gripper=-1.0),
            # Push door open (forward and sideways motion)
            Waypoint(t=500, xyz=push_pos, quat=identity_quat, gripper=-1.0),
            # Move through the door opening
            Waypoint(t=700, xyz=through_pos, quat=identity_quat, gripper=-1.0),
            # Move in -y to avoid door collision
            Waypoint(t=850, xyz=avoid_pos, quat=identity_quat, gripper=-1.0),
            # Move slightly forward in +x
            Waypoint(t=1000, xyz=forward_avoid_pos, quat=identity_quat, gripper=-1.0),
            # Move diagonally +x, -y to further avoid door
            Waypoint(t=1200, xyz=diagonal_avoid_pos, quat=identity_quat, gripper=-1.0),
            # Final position past door
            Waypoint(t=1250, xyz=final_pos, quat=identity_quat, gripper=-1.0),
        ]

        self._log_trajectory_summary(f"door_position={door_position}")
