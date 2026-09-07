# Configuration Reference

This page collects the cross-cutting configuration fields shared by maintained AM-Bench environments. Values shown are current repository defaults or reusable-profile values, not a fixed protocol for every experiment.

## Base environment

`BaseEnvCfg` is defined in `source/ambench/ambench/tasks/base/base_env_cfg.py`.

| Field | Current default | Meaning and constraint |
| --- | --- | --- |
| `sim.dt` | `1 / 120` s | Physics time step |
| `decimation` | `1` | Physics steps per environment step; the default environment rate is 120 Hz |
| `sim.render_interval` | `decimation` | Render interval in physics steps |
| `scene.num_envs` | `1` | Number of cloned environments before a runtime override |
| `scene.env_spacing` | `5.0` m | Distance between cloned environment origins |
| `scene.replicate_physics` | `False` | Whether cloned environments share replicated physics state |
| `spawn_ground_plane` | `True` | Adds the shared ground plane when enabled |
| `scene_camera_cfg` | `None` | Optional external `CameraCfg`; recording and evaluation may supply one |
| `robot_profile` | required | Complete robot, action, controller, IK, and profile-camera composition |

The shared rigid-body material uses static and dynamic friction `1.0`, restitution `0.0`, and `multiply` combination modes. Individual tasks may define additional object materials.

## Noise and physical-effect switches

All five switches default to `False` in `BaseEnvCfg`.

| Field | Effect when enabled | Default model or constraint |
| --- | --- | --- |
| `enable_action_noise` | Applies `action_noise_model` | If unset, additive Gaussian mean `0.0`, standard deviation `0.0001` |
| `enable_observation_noise` | Applies `observation_noise_model` | If unset, additive Gaussian standard deviation `0.002` plus absolute-bias standard deviation `0.0001` |
| `enable_saturation` | Enforces rotor thrust limits | Required when rotor actuator dynamics are configured |
| `enable_aerodynamic_effects` | Enables ground effect, near-wall effect, and drag | Requires a physical profile with aerodynamic configuration |
| `enable_wind_effect` | Enables the implemented wind preset | Current controller implementation uses a constant world-frame force |

Task reset and prestartup randomization lives in each task's `EventCfg`; see [Tasks and Scenes](../configure/tasks-and-scenes.md) for the maintained task-specific surfaces.

## Robot and control composition

`RobotProfileCfg` in `source/ambench/ambench/tasks/base/robot_profiles.py` has four fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `robot` | `RobotSpecCfg` | Articulation, semantic bodies and joints, gripper, and optional multirotor capabilities |
| `control` | `ControlPipelineCfg` | Pipeline class, action mode, controller, and optional IK |
| `ee_camera` | `CameraCfg` | Required profile EE camera |
| `base_camera` | `CameraCfg \| None` | Optional base camera |

Profile cameras are RGB, `384 x 384`, update every render, and use clipping range `(0.01, 1000.0)` m. Camera prim paths and offsets differ by robot. The optional evaluation scene camera is configured separately.

`ControlPipelineCfg` in `source/ambench/ambench/controllers/control_pipeline.py` exposes `class_type`, `action_mode`, `controller`, and optional `ik`. Its action modes are:

| `ActionMode` | Dimension | Public layout |
| --- | ---: | --- |
| `ABSOLUTE_EE_POSE` | `8` | position (3), quaternion WXYZ (4), gripper (1) |
| `ABSOLUTE_BASE_JOINTS` | `8 + arm DoF` | base position (3), quaternion WXYZ (4), ordered arm joints, gripper (1) |

Use [Robots and Profiles](../configure/robots-and-profiles.md) for the maintained compositions and [Observations and Actions](interfaces.md) for the full environment interface.

## Robot specification fields

`source/ambench/ambench/robots/robot_cfg.py` defines the reusable morphology contract.

| Configuration | Fields and constraints |
| --- | --- |
| `RobotSpecCfg` | required `robot_id`, `asset`, `control_body_name`, and `base_body_name`; optional `ee_body_name`, ordered unique `arm_joint_names`, `end_effector`, `gripper`, `max_arm_reach`, and `multirotor` |
| `EndEffectorFrameCfg` | `tool_tip_offset_local=(0.18, 0.0, 0.0)` m and optional `command_to_link_quat_wxyz` |
| `GripperSpecCfg` | ordered `joint_names`, optional open/closed joint positions, and optional object-width converter; calibration arrays must match the joint count |
| `MultirotorSpecCfg` | required rotor `layout_provider` and `actuator`; `fully_actuated=True`; optional motor-arm joints, aerodynamics, and propeller visualization |

Concrete robot specifications override these values. Current physical profiles use per-rotor thrust limits `(0.0, 23.0)` N and reaction-torque ratio `0.02` m. UAQuad uses propeller radius `0.190` m; the hexarotor-based specifications use `0.152` m.

## Rotor actuator and aerodynamics

`RotorActuatorCfg` fields are validated when the robot specification is created.

| Field | Default | Constraint |
| --- | --- | --- |
| `thrust_limits` | required | Nonnegative, increasing `(minimum, maximum)` in N |
| `reaction_torque_ratio` | `0.02` m | Nonnegative reaction moment per unit thrust |
| `propeller_radius` | `0.152` m | Positive |
| `response_time_constant_s` | `None` | Positive seconds when configured |
| `normalized_acceleration_limit_per_s` | `None` | Positive normalized-speed rate when configured |

Actuation is instantaneous when both transient fields are `None`. Enabling either requires `enable_saturation=True`.

`AerodynamicCfg` defaults are:

| Field | Default |
| --- | --- |
| `max_raycast_distance` | `5.0` m |
| `ground_effect_b`, `ground_effect_k` | `1.0`, `0.171` |
| `wall_effect_a1`, `wall_effect_b1` | `0.05`, `0.34` |
| `wall_effect_a2`, `wall_effect_b2` | `0.02`, `0.25` |
| `drag_coefficients` | `(0.1, 0.1, 0.1)` |

See [Physics, Disturbances, and Randomization](../configure/physics-disturbances-and-randomization.md) for effect ordering and runtime validation.

## Controller and IK defaults

`ControllerCfg` contains a controller `class_type` and its constructor `params`. Reusable profile values are defined in `source/ambench/ambench/tasks/base/robot_profiles.py`.

| Parameter set | Current values |
| --- | --- |
| Six-DoF PID | position `kp=200`, `kd=120`, `ki=80`; rotation `kp=200`, `kd=120`, `ki=120` |
| OmniHexa six-DoF PID | same position gains; rotation `kp=240`, `kd=130`, `ki=130` |
| UAHexa four-DoF PID | XY `kp=10`, `kd=8`, `ki=6`; Z `kp=30`, `kd=10`, `ki=10`; rotation `kp=200`, `kd=120`, `ki=120` |
| UAQuad four-DoF PID | same XY/Z gains; rotation `kp=150`, `kd=20`, `ki=20` |
| Four-DoF outer loop | `outer_loop_decimation=3` |
| L1 adaptive additions | `low_pass_filter_bandwidth=0.3`, linear/angular `sigma_max=2.0`, `adaptive_mix=1.0`; PID integral gains are set to zero |

Maintained Pyroki configurations start the base at `(0, 0, 1)` m, use WXYZ identity orientation, disable collision, constrain `max_x=1.0` m and `min_z=0.0` m, and use `rest_arm=0.1`. FAHexa fixes all base-orientation axes with safety margin `0.85`; UAHexa fixes roll and pitch with margin `0.8`; UAQuad fixes roll and pitch with margin `0.15`; OmniHexa leaves orientation free with margin `0.3`.

The maintained whole-body MPC profile uses `T=0.8` s, `N=32`, position bounds `(-1.0, -5.0, 0.0)` to `(1.2, 2.0, 4.0)` m, four joint bounds of `-180` to `180` degrees, output gains `[1.0] * 6 + [0.5] * 4`, build directory `./tmp/acados_mpc_build`, and model name `fa_hexa_mpc`. Its `Q`, `R`, and `R_delta` arrays remain defined in the profile source and must retain dimensions compatible with the MPC model.
