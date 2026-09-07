# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from .base import BasePolicy, Waypoint


class CabinetPickPlacePolicy(BasePolicy):
    """Scripted policy for cabinet pick-and-place task."""

    def generate_trajectory(self, obs, env_id: int = 0):
        """Generate waypoint trajectory for cabinet pick-and-place task."""
        obs_dict = obs["policy"][env_id]

        # Task observations.
        pulling_door_pos = obs_dict["pulling_door_pos"]
        can_pos = obs_dict["can_pos"]

        # Initial pose and orientation.
        init_ee_pose = self._ee_pos_init_local[env_id]
        identity_quat = self.identity_quat()

        # Door interaction sequence.
        door_approach_offset = 0.58
        door_approach_y_offset = 0.5
        door_approach_z_offset = 0.2
        door_approach_pose = torch.tensor(
            [
                pulling_door_pos[0] - door_approach_offset,
                pulling_door_pos[1] + door_approach_y_offset,
                pulling_door_pos[2] + door_approach_z_offset,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        door_pull_right_distance = 0.62
        door_pull_right_pose = torch.tensor(
            [
                door_approach_pose[0],
                door_approach_pose[1] - door_pull_right_distance,
                door_approach_pose[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Retract and align with can.
        door_retract_pose = torch.tensor(
            [
                door_pull_right_pose[0] - 0.1,
                door_pull_right_pose[1],
                door_pull_right_pose[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        can_align_y_pose = torch.tensor(
            [
                door_retract_pose[0],
                can_pos[1],
                door_retract_pose[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Can pickup sequence.
        can_approach_x_offset = 0.15
        can_approach_z_offset = 0.0
        can_approach_pose = torch.tensor(
            [
                can_pos[0] - can_approach_x_offset,
                can_pos[1],
                can_pos[2] + can_approach_z_offset,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        can_pick_pose = torch.tensor(
            [
                can_pos[0] - can_approach_x_offset,
                can_pos[1],
                can_pos[2] - 0.02,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        can_lift_to_approach_pose = torch.tensor(
            [
                can_pick_pose[0],
                can_pick_pose[1],
                can_pick_pose[2] + 0.01,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Transit to place region.
        cabinet_backward_pose = torch.tensor(
            [
                door_retract_pose[0],
                can_lift_to_approach_pose[1],
                can_lift_to_approach_pose[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        cabinet_upward_pose = torch.tensor(
            [
                cabinet_backward_pose[0],
                cabinet_backward_pose[1],
                cabinet_backward_pose[2] + 0.65,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        drawer_place_pose = torch.tensor(
            [
                can_pick_pose[0],
                can_pick_pose[1],
                cabinet_upward_pose[2],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        drawer_final_pose = torch.tensor(
            [
                drawer_place_pose[0],
                drawer_place_pose[1],
                cabinet_upward_pose[2] - 0.05,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Waypoint sequence.
        self.waypoints = [
            Waypoint(t=0, xyz=init_ee_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=400, xyz=door_approach_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=500, xyz=door_approach_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=1150, xyz=door_pull_right_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=1200, xyz=door_pull_right_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=1300, xyz=door_retract_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=1450, xyz=can_align_y_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=1600, xyz=can_approach_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=1850, xyz=can_pick_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=1900, xyz=can_pick_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2000, xyz=can_lift_to_approach_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2200, xyz=cabinet_backward_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2400, xyz=cabinet_upward_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2600, xyz=drawer_place_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2700, xyz=drawer_final_pose, quat=identity_quat, gripper=-1.0),
            Waypoint(t=2800, xyz=drawer_final_pose, quat=identity_quat, gripper=1.0),
            Waypoint(t=3000, xyz=drawer_final_pose, quat=identity_quat, gripper=1.0),
        ]

        self._log_trajectory_summary(f"pulling_door_pos={pulling_door_pos}", f"can_pos={can_pos}")
