# Controllers

Controllers live in `source/ambench/ambench/controllers/`. A `ControlPipelineCfg` combines the public action mode, a controller configuration, and optional Pyroki IK settings; a `RobotProfileCfg` binds that pipeline to one robot.

## Maintained controller families

| Family | Implementation | Role | Typical ID token |
| --- | --- | --- | --- |
| Six-DoF PID | `pid_6dof_ctrl.py` | Tracks position and full orientation for fully actuated bases and the EE oracle | `PID` |
| Four-DoF PID | `pid_4dof_ctrl.py` | Tracks position and yaw for underactuated multirotors | `PID` |
| L1 adaptive | `l1_adaptive_ctrl.py` | Adds adaptive disturbance compensation around the six-DoF tracking structure | `L1` |
| Whole-body MPC | `wholebody_mpc_ctrl.py` | Jointly computes base wrench and arm targets under state and input constraints | `MPC` |
| Joint PID | `pid_joint_ctrl.py` | Tracks arm and gripper joint targets inside robot command application | internal |

Pyroki (`pyroki_ik_ctrl.py`) is an IK front end, not a wrench controller. In an IK–tracking profile it resolves an EE target into base and arm targets; PID or L1 then tracks the base target. The MPC profile performs whole-body resolution within the optimizer and does not use the Pyroki front end.

## Public controller contract

`BaseController.compute(...)` returns `ControllerOutput`. Its stable fields include applied body-frame force and torque, optional arm targets, and optional telemetry for commanded and realized rotor thrust, desired and final wrench, rotor angles, proximity effects, and drag. Controllers expose `reset(...)` for episode-local state such as integrators or adaptive estimates.

The simulator-facing pipeline is:

```text
policy action -> ControlPipeline -> controller -> ControllerOutput -> RobotIO
```

Controller code should not contain task rewards, task success logic, or task reset geometry.

## Physical-effect order

For multirotors, `BaseController.compute(...)` applies inverse allocation and saturation, optional rotor actuator dynamics, proximity effects, forward allocation, drag, and optional wind before returning the realized wrench. See [Physics, Disturbances, and Randomization](../configure/physics-disturbances-and-randomization.md).

## Dependencies

Physical EE-target profiles require the `ext/pyroki` integration. Whole-body MPC additionally requires acados and its generated solver build. The EE oracle is the appropriate first check when those optional dependencies are not installed.

Exact reusable gains, IK constraints, MPC settings, rotor actuator fields, and aerodynamic defaults are listed in [Configuration Reference](configuration.md). Use [Controllers and Control Pipelines](../configure/controllers-and-control-pipelines.md) for selection and tuning guidance.
