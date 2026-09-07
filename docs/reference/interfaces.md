# Observations and Actions

This page describes the environment-level contract. Learned policies may transform these values into policy-specific relative representations, but must convert predictions back to the selected environment action mode before stepping.

## Common policy observations

Each task returns `{"policy": ...}`. The policy value is a per-environment dictionary containing the common robot state plus task-specific fields.

| Key | Shape | Frame or convention |
| --- | ---: | --- |
| `ee_pos` | 3 | End-effector link position, world-aligned and relative to the environment origin |
| `ee_quat` | 4 | End-effector link orientation in the world frame, WXYZ |
| `ee_lin_vel` | 3 | World-frame linear velocity |
| `ee_ang_vel` | 3 | World-frame angular velocity |
| `base_pos` | 3 | World-aligned position relative to the environment origin |
| `base_quat` | 4 | Base quaternion, WXYZ |
| `base_lin_vel` | 3 | World-frame linear velocity |
| `base_ang_vel` | 3 | World-frame angular velocity |
| `gripper_width` | 1 | Sum of configured gripper-joint positions |
| `arm_joint_pos` | arm DoF | Present for manipulator profiles |
| `arm_joint_vel` | arm DoF | Present for manipulator profiles |

Task environments add goal, object, articulation, or success-state fields. Camera RGB frames are available from configured sensors such as `ee_camera`, `base_camera`, and an optional `scene_camera`; dataset and evaluator adapters select the camera keys they consume.

The EE observations above come directly from the configured end-effector link state. They are not the command-frame pose used at the action boundary. For profiles with a command-to-link orientation offset, `RobotIO` applies that conversion while processing EE actions.

## Action modes

| `ActionMode` | Dimension | Layout |
| --- | ---: | --- |
| `ABSOLUTE_EE_POSE` | 8 | position (3), quaternion WXYZ (4), gripper (1) |
| `ABSOLUTE_BASE_JOINTS` | `8 + arm DoF` | base position (3), base quaternion WXYZ (4), arm joints, gripper (1) |

Positions are relative to the cloned environment origin at the public action boundary. The control pipeline converts them to world coordinates where required.

Wipe Window is the only maintained task that changes an action before the shared pipeline: it holds the gripper command at `-0.6` while preserving the requested pose command.

## Step and episode outputs

Environment steps follow the Gymnasium tuple `(observation, reward, terminated, truncated, info)`. The maintained tasks currently return zero reward. `terminated` reports final task success, while `truncated` reports the episode time limit.

Tasks with named progress criteria expose boolean tensors in `info["success_criteria"]`; tasks without named criteria omit that key. Evaluation records whether each available criterion became true at least once during a rollout and reports their completed fraction as subtask completion. Reset first restores shared robot and control-pipeline state, then restores task-owned articulations, objects, attachments, counters, or visibility buffers.

## Environment and model representations

Canonical datasets record the environment representation. ACT, DP, and OpenPI can train on derived local-relative trajectories, so checkpoint metadata must agree with both the source action semantics and the evaluation environment. Do not infer compatibility from tensor width alone.

For the precise dataset fields, see [Dataset Format](dataset-format.md). Shared defaults are in [Configuration Reference](configuration.md); action-mode selection and camera ownership are covered by [Actions, Observations, and Cameras](../configure/actions-observations-and-cameras.md).
