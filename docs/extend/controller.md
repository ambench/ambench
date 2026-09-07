# Add a Controller

Controller implementations belong in `source/ambench/ambench/controllers/` and should remain reusable across tasks. Verify an existing task/profile combination first; the result is a controller selected through a `ControlPipelineCfg` and `RobotProfileCfg`, with applicable `ControllerOutput` telemetry preserved.

Start from `source/ambench/ambench/controllers/pid_6dof_ctrl.py` for a six-DoF wrench controller, `pid_4dof_ctrl.py` for underactuated tracking, `l1_adaptive_ctrl.py` for an adaptive extension, or `wholebody_mpc_ctrl.py` for the optimizer path. Use `controller_cfg.py` and `control_pipeline.py` as the shared integration contracts.

## 1. Define the controller surface

Wrench controllers subclass `BaseController`, implement `compute_desired_wrench(...)` or the supported MPC path, return the inherited `ControllerOutput` from `compute(...)`, and expose `reset(...)` for episode state. Use `ControllerCfg` for constructor parameters.

The controller receives resolved robot morphology, rotor layout, optional actuator state, articulation, scene, timestep, and physical-effect flags. Do not rediscover body or joint names from task strings.

## 2. Choose the pipeline boundary

If the policy emits EE targets, decide whether an existing Pyroki IK front end should generate base and arm targets or whether the controller resolves the whole body itself. If the policy emits base-plus-joint targets, use the direct pipeline and keep joint tracking in robot I/O.

Add a named `RobotProfileCfg` composition in `tasks/base/robot_profiles.py`. Task configs select that profile; they should not instantiate controllers locally.

## 3. Preserve telemetry

Populate `ControllerOutput` fields that apply to the new controller. At minimum, return applied `force_b` and `torque_b`; include desired/final wrench, arm targets, motor thrust, allocation, or proximity-effect values when available. Shared evaluation depends on this telemetry for tracking and limit-use comparisons.

## 4. Validate

Use a task-independent numerical check for controller math first. Then run a bounded EE-only or free-space tracking environment, reset repeatedly, and inspect error and saturation logs. Add contact tasks only after free-space tracking is stable.

For a new whole-body optimizer, validate feasibility, units, frame conventions, constraint enforcement, generated-code paths, and cleanup of solver state. Never commit generated build output or machine-specific solver paths.
