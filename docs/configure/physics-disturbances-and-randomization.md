# Configure Physics, Disturbances, and Randomization

AM-Bench separates shared simulation timing, multirotor actuation and aerodynamic models, noise, wind, and task-owned domain randomization. Record each enabled layer explicitly when comparing runs.

## Shared effect switches

`BaseEnvCfg` exposes five independent switches. Their current shared defaults are all `False`.

| Field | Effect | Constraint |
| --- | --- | --- |
| `enable_action_noise` | Apply `action_noise_model` through the Isaac Lab environment step | If enabled without a model, the config creates additive Gaussian noise with mean `0.0` and standard deviation `0.0001` |
| `enable_observation_noise` | Apply `observation_noise_model` to the policy observation | If enabled without a model, the config creates additive Gaussian noise with standard deviation `0.002` and an additive-bias model with standard deviation `0.0001` |
| `enable_saturation` | Clamp allocated rotor thrust to the selected robot's limits | Required when transient rotor dynamics are configured |
| `enable_aerodynamic_effects` | Enable ground effect, near-wall effect, and body-frame drag | Also enables scene-query support before simulator construction |
| `enable_wind_effect` | Add the implemented constant world-frame wind force | Current implementation uses `[6.0, 6.0, 6.0]` N; it is a boolean preset, not a configurable vector |

Noise models may use other Isaac Lab noise configs, but their tensor shape must remain compatible with the selected action or policy observation.

## Multirotor actuation

Each physical `RobotSpecCfg` provides a `MultirotorSpecCfg` with a rotor-layout provider and `RotorActuatorCfg`. Current robot specs use a per-rotor thrust range of `(0.0, 23.0)` N and a reaction-torque ratio of `0.02` m. The propeller radius is `0.19` m for UAQuad and `0.152` m for the three hexarotor-based specifications.

`response_time_constant_s` and `normalized_acceleration_limit_per_s` are both `None` in the current robot specs, so actuation is instantaneous unless a derived specification enables one or both. When transient dynamics are enabled, both values must be positive when present and `enable_saturation` must also be `True`.

## Aerodynamic parameters and order

Physical robot specs attach an `AerodynamicCfg`. Its current defaults include a 5 m raycast limit, ground-effect coefficients `b=1.0` and `k=0.171`, two near-wall coefficient pairs, and body-axis drag coefficients `(0.1, 0.1, 0.1)`.

For a multirotor controller step, the implementation order is:

1. compute the desired body wrench;
2. perform inverse allocation;
3. apply steady thrust saturation, or let the transient actuator own saturation;
4. apply optional rotor response and normalized-acceleration limits;
5. apply ground and near-wall effects to rotor thrust vectors;
6. reconstruct the realized body wrench;
7. add body-frame drag and optional world-frame wind.

The EE oracle has no `MultirotorSpecCfg`, so rotor allocation, saturation, proximity effects, drag, and wind do not represent a physical aerial platform on that profile.

## Task domain randomization

Task randomization is configured through each task's `EventCfg`; there is no single global difficulty field. Current public tasks expose these event groups:

| Task | Current event surfaces |
| --- | --- |
| Cabinet Pick-and-Place | can XY reset pose |
| Frame Assembly | coordinated wall/peg reset pose; frame reset position |
| Lemon Harvesting | lemon/lime positions on the wall |
| NDT | no task `EventCfg` terms |
| Open Door | door reset pose; hinge actuator damping scale |
| Peg-in-Hole | wall reset pose; hole position; prestartup hole scale |
| Press Button | wall reset pose; button position; prestartup button scale |
| Pull Lever | wall/lever reset pose; lever and tip visual colors |
| Push Slider | wall/slider reset pose |
| Rotate Valve | wall/valve reset pose |
| Toss Ball | container XY reset pose |
| Wipe Window | window reset pose; stain positions; window friction |

`prestartup` events change stage-level properties before simulation begins and are not per-reset randomization. `reset` events run for the selected environment IDs during resets. When one object is attached to another, use the task's coordinated event function so dependent poses remain consistent.

## Validate one layer at a time

Begin with the selected registered default. Enable or change one effect, noise model, or event range, then run one bounded environment. Geometry scale, friction, contact thresholds, and aerodynamic raycasts require simulator-level validation; static config inspection cannot prove their runtime behavior.

For task fields and scene ownership, see [Tasks and Scenes](tasks-and-scenes.md). For controller telemetry such as commanded/realized thrust and proximity wrenches, see [Controllers](../reference/controllers.md).
