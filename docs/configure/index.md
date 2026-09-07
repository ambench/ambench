# Configure AM-Bench

AM-Bench configuration composes a task, scene, robot, public action interface, controller, cameras, and optional physical effects into a registered environment. Start from an existing environment configuration and change one layer at a time.

## Choose the configuration layer

| Goal | Configuration surface | Guide |
| --- | --- | --- |
| Change objects, episode length, success thresholds, or reset events | task `*_env_cfg.py` | [Tasks and Scenes](tasks-and-scenes.md) |
| Change the articulation, frames, joints, gripper, or platform capabilities | `RobotSpecCfg` inside a `RobotProfileCfg` | [Robots and Profiles](robots-and-profiles.md) |
| Change the public action representation or camera inputs | `ControlPipelineCfg.action_mode` and profile/task camera configs | [Actions, Observations, and Cameras](actions-observations-and-cameras.md) |
| Select or tune IK, PID, L1, or MPC | `RobotProfileCfg.control` | [Controllers and Control Pipelines](controllers-and-control-pipelines.md) |
| Enable noise, saturation, aerodynamic effects, wind, or task randomization | `BaseEnvCfg`, robot actuator/aerodynamic specs, and task `EventCfg` | [Physics, Disturbances, and Randomization](physics-disturbances-and-randomization.md) |

## Start from a registered environment

Use [Find an Environment](../getting-started/find-an-environment.md) to select an exact environment ID. Its Gym registration points to a concrete task configuration class, which is the authoritative composition for that variant.

For example, `PressButton-Am-FAHexa-Abs-PID-Direct-v0` resolves to a Press Button config that selects the `FA_HEXA_ABS_PID` profile. The task config owns the scene and success condition; the profile owns the robot, absolute end-effector action mode, Pyroki IK, six-DoF PID controller, and cameras.

Not every theoretical task/profile combination is registered. Prefer an existing config class over constructing an ID or assembling a profile at runtime.

## Where configuration lives

The shared environment defaults are in `source/ambench/ambench/tasks/base/base_env_cfg.py`. Reusable profiles are in `source/ambench/ambench/tasks/base/robot_profiles.py`; robot specifications are in `source/ambench/ambench/robots/`; controller and action-pipeline configuration is in `source/ambench/ambench/controllers/`.

Each maintained task has its own package under `source/ambench/ambench/tasks/<task>/`. Use its `__init__.py` to see registered variants, its `*_env_cfg.py` for configurable scene values and events, and its `*_env.py` for observations and success behavior.

If configuration is insufficient and implementation work is required, continue with [Extend an Existing Environment](../extend/index.md).

Use [Configuration Reference](../reference/configuration.md) for exact shared fields, defaults, units, and advanced profile values.
