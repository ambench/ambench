# Add a Robot

A robot integration defines morphology and simulator I/O independently of any task. Complete [Installation](../getting-started/installation.md) and verify one existing environment first. The result is a `RobotSpecCfg`, a compatible `RobotProfileCfg`, and at least one explicitly registered task variant.

Start from the maintained robot closest to the new morphology: `source/ambench/ambench/robots/eef.py`, `ua_quad.py`, `ua_hexa.py`, `fa_hexa.py`, or `omni_hexa.py`. Use `source/ambench/ambench/robots/robot_cfg.py` for the shared specification contract and `source/ambench/ambench/tasks/base/robot_profiles.py` for composition.

## 1. Prepare the asset

Place final local URDF, USD, meshes, and textures under `source/ambench/ambench/assets/robots/`. Do not hard-code workstation paths. Small wrapper assets that reference maintained upstream content are preferable to copied asset bundles; Git LFS is not enabled.

Verify link names, joint names, joint directions, inertial parameters, collision geometry, and default pose before tuning a controller.

## 2. Define `RobotSpecCfg`

Add the robot module under `source/ambench/ambench/robots/`. Its `RobotSpecCfg` should identify:

- articulation spawn configuration and initial state;
- base and end-effector body names;
- ordered arm, gripper, and optional motor-arm joints;
- gripper width conversion;
- multirotor geometry, thrust limits, and aerodynamics where applicable;
- command/link frame conversion for the tool.

`RobotIO` resolves live handles from this specification. Do not add a robot enum or robot-name branches to `BaseEnv`.

## 3. Compose a profile

Add cameras, IK configuration, controller settings, and a `ControlPipelineCfg` in `tasks/base/robot_profiles.py`. Choose a pipeline that matches the morphology:

- `EndEffectorPosePipeline` for the EE oracle;
- `ManipulatorIKPipeline` for EE target plus IK and tracking;
- `ManipulatorDirectPipeline` for absolute base-plus-joint targets;
- `ManipulatorMPCPipeline` for the supported whole-body MPC path.

## 4. Register task variants

Add task config subclasses that select the profile, then register explicit IDs in each supported task package. Begin with one representative task rather than registering the full suite before the robot steps reliably.

## 5. Validate in layers

First verify import, articulation spawn, resolved body/joint handles, and reset. Next check base hover or EE tracking without contact. Then run reachable EE target and FK consistency checks, a scripted task, saturation telemetry, and camera output. Report which task/controller combinations were actually tested.
