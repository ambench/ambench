# Configure Tasks and Scenes

Task configurations own episode timing, task assets, surrounding geometry, reset randomization, task observations, and success criteria. Select a registered configuration first, then inspect that task's config instead of assuming every task exposes the same fields.

## Shared environment settings

Every maintained task derives from `BaseEnvCfg` in `source/ambench/ambench/tasks/base/base_env_cfg.py`.

| Field | Current shared default | Effect |
| --- | --- | --- |
| `sim.dt` | `1 / 120` seconds | Physics step interval |
| `decimation` | `1` | Physics steps per environment step; the environment step interval is `sim.dt * decimation` |
| `scene.num_envs` | `1` | Number of cloned environments before a launcher override |
| `scene.env_spacing` | `5.0` metres | Spacing used when laying clones along the Y axis |
| `scene.replicate_physics` | `False` | Whether Isaac Lab replicates physics state across clones |
| `spawn_ground_plane` | `True` | Whether `BaseEnv` creates `/World/ground` |
| `episode_length_s` | task-specific | Time limit used to compute episode truncation |

Command-line tools may override values such as the environment count after loading the registered config. Record both the environment ID and launcher arguments when comparing runs.

## Maintained task configs

| Task ID prefix | Default config class | Episode limit | Representative task-owned settings |
| --- | --- | ---: | --- |
| `CabinetPickPlace` | `CabinetPickPlaceEnvDefaultCfg` | 60 s | cabinet and can poses; drawer height; placement and velocity thresholds |
| `FrameAssembly` | `FrameAssemblyEnvDefaultCfg` | 30 s | wall pose; peg separation/count; frame dimensions, mass, and placement tolerances |
| `LemonHarvesting` | `LemonHarvestingEnvDefaultCfg` | 30 s | wall and lemon/lime positions; container size; grasp/release thresholds |
| `NDT` | `NDTEnvDefaultCfg` | 20 s | scene and marker transforms; inspection goal; position tolerances and hold steps |
| `OpenDoor` | `OpenDoorEnvDefaultCfg` | 15 s | door pose and asset; open-angle threshold |
| `PegInHole` | `PegInHoleEnvDefaultCfg` | 20 s | wall/hole/peg geometry and poses; peg mass; insertion-depth threshold |
| `PressButton` | `PressButtonEnvDefaultCfg` | 20 s | wall/button geometry and poses; button radius and pressed threshold |
| `PullLever` | `PullLeverEnvDefaultCfg` | 20 s | wall/lever geometry and pose; minimum pull angle |
| `PushSlider` | `PushSliderEnvDefaultCfg` | 26 s | wall/slider pose and asset; pushed threshold |
| `RotateValve` | `RotateValveEnvDefaultCfg` | 20 s | wall/valve pose and properties; target angle and engagement ratio |
| `TossBall` | `TossBallEnvDefaultCfg` | 10 s | container size; ball radius/mass; release and base-position thresholds |
| `WipeWindow` | `WipeWindowEnvDefaultCfg` | 20 s | window/sponge geometry; stain count; contact-force, count, and radius thresholds |

These are current software defaults, not an immutable benchmark protocol. Read the selected class before overriding it, because derived registered configs may replace the robot profile or adjust task values.

## Success and progress criteria

Final success produces `terminated=True`. Named criteria are rollout progress signals; the evaluator marks a criterion complete if it becomes true at least once.

| Task | Final success | Named criteria |
| --- | --- | --- |
| Cabinet Pick Place | can is within the drawer height band and below the velocity threshold | drawer opened; can lifted; can placed on drawer |
| Frame Assembly | frame center is within the configured X and YZ peg tolerances | frame lifted; frame placed on pegs |
| Lemon Harvesting | lemon is inside the container region and the gripper is open | lemon grasped and detached; lemon delivered and released |
| NDT | EE remains within the inspection tolerances for the configured hold steps | inspection target held |
| Open Door | door joint exceeds the configured open angle | door engaged; door opened |
| Peg In Hole | peg tip passes the insertion depth and lateral-bound checks in the hole frame | none |
| Press Button | button joint reaches the pressed threshold | none |
| Pull Lever | lever joint exceeds the minimum pull angle | lever engaged; lever pulled |
| Push Slider | slider joint reaches the pushed threshold | slider engaged; slider pushed |
| Rotate Valve | valve joint reaches the target angle | valve engaged; valve rotated |
| Toss Ball | ball is released inside the container bounds while the robot remains behind the bin | ball released; ball delivered |
| Wipe Window | every stain has been removed by the configured contact checks | one removed criterion per stain |

Every maintained task calls the shared reset before restoring its own state. Articulated tasks restore their joints; object tasks restore or retain their configured/event-randomized poses as appropriate; attached-object tasks reattach the peg, ball, or sponge; and stateful tasks clear hold, grasp, contact, or visibility buffers.

## Change a task value

For a maintained variant, define a config subclass next to the task rather than mutating a shared class-level object. This example changes only Press Button task settings:

```python
from isaaclab.utils import configclass

from ambench.tasks.press_button.press_button_env_cfg import PressButtonEnvDefaultCfg


@configclass
class PressButtonLongEpisodeCfg(PressButtonEnvDefaultCfg):
    episode_length_s = 30
    button_pressed_threshold = 0.006
```

A new public variant also needs an explicit `register_env(...)` entry in the task package's `__init__.py`. Use [Environment IDs](../reference/environments.md) for the naming contract and [Add a Task](../extend/task.md) for the registration and validation path.

## Configure scene geometry and assets

Task configs use Isaac Lab asset configs such as `ArticulationCfg`, `RigidObjectCfg`, and scene helpers such as `WallSceneCfg`. Keep related derived values together: moving a wall may require recomputing the local offset or initial position of a button, hole, lever, slider, valve, or peg fixture.

Simple reusable surrounding geometry belongs in `source/ambench/ambench/scenes/`. Task-specific object counts, asset choices, poses, and success thresholds remain in the task config. Do not point a public config at a machine-local asset path.

## Verify success and observations

Config fields only define the inputs to task behavior. The task's `*_env.py` remains authoritative for `_get_observations()` and `_get_success()`. After changing geometry, thresholds, or event ranges, run one bounded environment and exercise the smallest relevant scripted policy before scaling the number of environments.

See [Physics, Disturbances, and Randomization](physics-disturbances-and-randomization.md) for the current event surfaces and [Actions, Observations, and Cameras](actions-observations-and-cameras.md) for the environment-level policy contract.
