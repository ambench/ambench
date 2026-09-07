# Extend an Existing Environment

Use this section when the change requires a new implementation. If an existing registered environment already exposes the required field, stay in [Configure AM-Bench](../configure/index.md) and change its task config or robot profile instead.

Before writing code, [find the closest registered environment](../getting-started/find-an-environment.md), identify which layer owns the behavior, and inspect the corresponding maintained implementation. Preserve the boundary between task logic, robot morphology, control, and learned-policy adapters; change one layer at a time.

| Goal | Guide | Primary package |
| --- | --- | --- |
| Add scene logic and success criteria | [Add a Task](task.md) | `ambench.tasks` |
| Add an articulation and morphology | [Add a Robot](robot.md) | `ambench.robots` |
| Add reusable control math | [Add a Controller](controller.md) | `ambench.controllers` |
| Add a training/evaluation integration | [Integrate a Policy](policy.md) | `ambench_learn.policies` |

## Source starting points

- Task registrations and implementations: `source/ambench/ambench/tasks/`
- Robot specification contract: `source/ambench/ambench/robots/robot_cfg.py`, and reusable robot profiles: `source/ambench/ambench/tasks/base/robot_profiles.py`
- Control pipeline contract: `source/ambench/ambench/controllers/control_pipeline.py`, and controller implementations: `source/ambench/ambench/controllers/`
- Learned-policy integrations: `source/ambench_learn/ambench_learn/policies/`, and the common evaluator: `source/ambench_learn/ambench_learn/eval/common.py`

## Design boundary

Tasks own objects, reset randomization, observations, rewards, and success. Robot specifications own articulation semantics, joints, links, cameras, and rotor geometry. Control pipelines translate public actions into `RobotCommand`; controllers compute reusable targets or wrenches. Learned-policy packages adapt canonical datasets and predictions without changing task internals.

Use `RobotProfileCfg` to compose a robot with a control pipeline. Add registered config classes for supported task/profile combinations instead of branching on robot names inside `BaseEnv`.

## Validation ladder

For every extension:

1. import the new module and confirm registration;
2. run one bounded headless environment;
3. exercise the new behavior with the smallest relevant scripted or policy check;
4. run formatting and focused tests;
5. document runtime dependencies and anything not verified.

Isaac-dependent checks require the activated Isaac Lab environment. GPU runs should begin with `nvidia-smi`, use one small environment, and avoid overlapping heavy simulator jobs.
