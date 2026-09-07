# Configure Controllers and Control Pipelines

`ControlPipelineCfg` combines a public `ActionMode`, a controller configuration, and optional Pyroki IK. `RobotProfileCfg` binds that pipeline to a robot specification. Select or tune the complete composition rather than treating the environment ID's controller token as an isolated switch.

## Maintained compositions

| Robot/profile family | Pipeline | Public action | Controller | IK |
| --- | --- | --- | --- | --- |
| EE oracle | `EndEffectorPosePipeline` | absolute EE | six-DoF PID or L1 adaptive | none |
| UAQuad | `ManipulatorIKPipeline` | absolute EE | four-DoF PID | Pyroki with roll/pitch fixed |
| UAHexa | `ManipulatorIKPipeline` | absolute EE | four-DoF PID | Pyroki with roll/pitch fixed |
| FAHexa | `ManipulatorIKPipeline` | absolute EE | six-DoF PID or L1 adaptive | Pyroki with base orientation fixed |
| FAHexa | `ManipulatorDirectPipeline` | absolute base plus joints | six-DoF PID or L1 adaptive | bypassed by the action path |
| FAHexa | `ManipulatorMPCPipeline` | absolute EE | whole-body MPC | none |
| OmniHexa | `ManipulatorIKPipeline` | absolute EE | six-DoF PID or L1 adaptive | Pyroki with free base orientation |
| OmniHexa | `ManipulatorDirectPipeline` | absolute base plus joints | six-DoF PID | none |

Not every task registers every reusable profile. Use the live registry rather than substituting tokens in an environment ID.

## Controller roles

Six-DoF PID tracks full base position and orientation for fully actuated platforms. Four-DoF PID uses nested position and attitude loops for underactuated platforms; the reusable profiles set its outer-loop decimation to `3`. L1 adaptive control wraps the six-DoF tracking structure with bounded adaptive compensation. Whole-body MPC jointly produces the base wrench and arm targets for the maintained FAHexa absolute-EE profile.

Pyroki is an IK front end, not a low-level wrench controller. It converts an EE target into base and arm targets for `ManipulatorIKPipeline`; PID or L1 then tracks the base target.

## Tune the owned parameters

Controller constructor parameters live in `ControllerCfg.params`. The reusable profiles in `source/ambench/ambench/tasks/base/robot_profiles.py` provide the maintained gains and constraints:

- `PID6DOFGains` holds position and rotation `kp`, `kd`, and `ki` values;
- `PID4DOFGains` separates XY, Z, and rotation gains, with `outer_loop_decimation` configured alongside it;
- L1 adds `low_pass_filter_bandwidth`, linear/angular sigma limits, and `adaptive_mix`;
- `PyrokiIKControllerConfig` owns fixed base axes, initial state, cost weights, base-position limits, collision setting, safety margin, and ordered arm joints;
- the MPC parameter map owns horizon `T`, node count `N`, state/input weights, position and joint bounds, output filtering, build directory, and model name.

Treat the values in the reusable profile as a starting configuration, not a universal protocol. Create a new profile for a new tuned combination and copy nested parameter mappings before changing them; mutating a shared profile can affect every task class that references it.

## Dependencies and validation

Physical EE-target IK profiles require the optional Pyroki installation. The whole-body MPC profile also requires acados and generates solver code under its configured build directory. Follow [Installation](../getting-started/installation.md) for those dependencies.

After changing a controller or IK constraint, validate the selected task with one environment and inspect controller telemetry before scaling. The public `ControllerOutput` contract and maintained controller inventory are in [Controllers](../reference/controllers.md). Physical limits and effect ordering are covered in [Physics, Disturbances, and Randomization](physics-disturbances-and-randomization.md).
