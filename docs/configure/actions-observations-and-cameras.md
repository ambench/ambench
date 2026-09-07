# Configure Actions, Observations, and Cameras

The selected robot profile determines the public action representation and its control path. The task environment supplies common robot observations, adds task-specific state, and exposes only the cameras configured by the profile or task.

## Select the action mode

`ActionMode` and `ControlPipelineCfg` are defined in `source/ambench/ambench/controllers/control_pipeline.py`.

| `ActionMode` | Dimension | Environment action layout | Processing path |
| --- | ---: | --- | --- |
| `ABSOLUTE_EE_POSE` | 8 | environment-relative position (3), WXYZ quaternion (4), gripper (1) | Convert position with the cloned environment origin, then use direct EE control, manipulator IK, or MPC |
| `ABSOLUTE_BASE_JOINTS` | `8 + arm DoF` | environment-relative base position (3), WXYZ base quaternion (4), ordered arm joints, gripper (1) | Bypass IK and send base and arm targets to the direct manipulator pipeline |

Select an existing profile with the required mode rather than changing `action_space` manually. `BaseEnvCfg.__post_init__()` derives the action dimension from the profile and arm-joint count.

Choose an EE action mode when the policy produces end-effector targets and the profile should resolve the robot configuration through IK or MPC. Choose `ABSOLUTE_BASE_JOINTS` only when the policy explicitly predicts base pose and ordered arm-joint targets.

The action mode is part of the dataset and checkpoint contract. A matching tensor width does not prove semantic compatibility; evaluation must use the action representation expected by the policy adapter and checkpoint metadata.

Wipe Window is the only maintained task with task-level action preprocessing. It overwrites the final gripper value with `-0.6` to keep the sponge grasped, then sends the pose portion through the selected shared control pipeline.

## Common observations

Every maintained task returns a `policy` observation containing common end-effector, base, and gripper state. Manipulator profiles add ordered `arm_joint_pos` and `arm_joint_vel`; individual tasks add goal, object, articulation, contact, or progress fields.

`ee_pos` and `ee_quat` describe the configured end-effector link, not the command frame. The position is world-aligned relative to the environment origin; the quaternion is the link's world orientation in WXYZ order. Profiles with a command-to-link offset apply it during action processing.

The shared config declares positions and linear velocities as three-vectors, WXYZ quaternions as four-vectors, angular velocities as three-vectors, and `gripper_width` as a scalar. Task implementations return a list of per-environment dictionaries, so consumers must preserve field names and ordering through their own adapter rather than assuming one globally flattened vector.

Use [Observations and Actions](../reference/interfaces.md) for the exact common keys, shapes, frames, and dataset boundary. Inspect the selected task's `_get_observations()` for its additional fields.

## Profile cameras

The reusable camera helper in `robot_profiles.py` currently configures 384×384 RGB output, an update period of `0.0`, and a ROS camera-frame convention. Camera prim paths and offsets differ by robot.

| Sensor key | Configuration owner | Availability |
| --- | --- | --- |
| `ee_camera` | `RobotProfileCfg.ee_camera` | Present on current reusable profiles; prim path and offset are robot-specific |
| `base_camera` | `RobotProfileCfg.base_camera` | Present on current physical robot profiles; absent from the EE oracle profiles |
| `scene_camera` | `BaseEnvCfg.scene_camera_cfg` | Optional and task/config-specific; shared default is `None` |

The environment registers only non-`None` cameras in `scene.sensors`. Before passing a name to `--camera_names`, verify that the selected profile or task config creates that sensor. A successful run with one profile does not prove that the same camera exists on another profile.

## Change camera settings safely

Camera resolution, data types, clipping, pose, and prim path belong in a copied profile or task config. Keep the camera attached to a body that exists in the selected articulation. Increasing resolution, adding data types, or enabling multiple cameras increases GPU memory and rendering cost; validate one environment before scaling.

Use [Run an Environment](../workflows/run-environment.md) for a bounded camera/video check. For robot-profile composition, see [Robots and Profiles](robots-and-profiles.md).
