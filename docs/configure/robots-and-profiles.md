# Configure Robots and Profiles

A task selects one complete `RobotProfileCfg`. The profile binds a `RobotSpecCfg` to an action/control pipeline and camera configuration, so robot, action mode, IK, controller, and camera choices remain compatible.

## Maintained robot specifications

| Environment tag | Robot specification | Arm DoF | Base capability | Maximum arm reach | Profile cameras |
| --- | --- | ---: | --- | ---: | --- |
| `EE` | `EndEffectorRobotSpecCfg` | 0 | end-effector oracle; no multirotor | not set | EE |
| `UAQuad` | `UAQuadRobotSpecCfg` | 4 | underactuated quadrotor | 0.67 m | EE and base |
| `UAHexa` | `UAHexaRobotSpecCfg` | 4 | underactuated hexarotor | 1.24 m | EE and base |
| `FAHexa` | `FAHexaRobotSpecCfg` | 4 | fully actuated fixed-tilt hexarotor | 1.24 m | EE and base |
| `OmniHexa` | `OmniHexaRobotSpecCfg` | 3 | fully actuated variable-tilt multirotor | 0.5 m | EE and base |

The EE profile is a task and policy debugging oracle, not a physical aerial-manipulator result. It directly controls a floating end-effector articulation and is the appropriate first variant for isolating scene, observation, action, and success behavior.

## What a robot specification owns

`RobotSpecCfg` in `source/ambench/ambench/robots/robot_cfg.py` is the authoritative morphology surface.

| Field | Purpose |
| --- | --- |
| `robot_id`, `asset` | Stable internal identifier and Isaac Lab articulation config |
| `control_body_name`, `base_body_name`, `ee_body_name` | Semantic body handles resolved against the live articulation |
| `arm_joint_names` | Ordered arm joints used by observations, IK, and command application |
| `end_effector` | Tool-tip offset and optional command-to-link quaternion transform |
| `gripper` | Ordered joints and open/closed or object-width calibration |
| `max_arm_reach` | Optional reach value used by configuration and control logic |
| `multirotor` | Rotor layout, actuator properties, actuation class, aerodynamics, and optional propeller visualization |

Body and joint names must match the loaded USD. Pyroki profiles also require a matching URDF and target-link naming.

## Select a profile through a registered config

Reusable profiles are defined in `source/ambench/ambench/tasks/base/robot_profiles.py`. Registered task subclasses normally differ only in their selected profile:

```python
from ambench.tasks.base.robot_profiles import FA_HEXA_ABS_PID


class MyTaskHexaAbsPIDCfg(MyTaskDefaultCfg):
    robot_profile = FA_HEXA_ABS_PID
```

Use an existing registered environment whenever it provides the required composition. Available profile names include EE PID/L1, FAHexa PID/L1/BaseJoint/MPC, OmniHexa PID/L1/BaseJoint, and underactuated UAHexa or UAQuad PID variants. Registrations are task-specific, so confirm the exact ID with [Find an Environment](../getting-started/find-an-environment.md).

## Keep profile changes atomic

Changing only the robot asset while retaining incompatible link names, joints, IK URDF, controller type, or camera prim paths will fail during environment construction or produce an invalid command path. For a new morphology, define a new `RobotSpecCfg`, then compose and register a complete profile.

For implementation-level integration, follow [Add a Robot](../extend/robot.md). For action and controller choices inside a profile, continue with [Actions, Observations, and Cameras](actions-observations-and-cameras.md) and [Controllers and Control Pipelines](controllers-and-control-pipelines.md).
