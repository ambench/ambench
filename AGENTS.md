# AGENTS.md

Instructions for coding agents working in `ambench`, an Isaac Lab extension for aerial manipulation.

## Start Here

Read local context before making structural changes:

1. `README.md`
2. `CODING_STYLE.md`
3. The nearest matching implementation in the same package
4. The public contributor guides:
   - [Extend AM-Bench](https://ambench.github.io/docs/extend/)
   - [Tasks](https://ambench.github.io/docs/extend/task/)
   - [Controllers](https://ambench.github.io/docs/extend/controller/)
   - [Robots](https://ambench.github.io/docs/extend/robot/)
   - [Policies](https://ambench.github.io/docs/extend/policy/)
   - [Workflows](https://ambench.github.io/docs/workflows/)

Prefer repository-consistent changes over clever rewrites.
Follow `CODING_STYLE.md` for code style.

## Quick Commands

Use the lightest command that proves the change works.

Efficient context gathering:

- Prefer `rg` or `rg --files` for codebase search.
- When reading several unrelated files, run independent reads in parallel where the agent runtime supports it.
- Avoid noisy chained shell output for context gathering; separate file reads are easier to review and summarize.

GPU capacity check:

- Before starting substantive work that may use Isaac Sim, rendering, or GPU-heavy validation, run `nvidia-smi` to inspect available GPUs, current memory pressure, and active processes on the machine you are using.
- If multiple GPUs are available, spread heavy runs across them when practical instead of stacking all jobs onto one device, leverage `CUDA_VISIBLE_DEVICES` to isolate workloads, and monitor VRAM usage to avoid overcommitment.
- If only one GPU is available, do not launch multiple heavy Isaac Sim or rendering workloads concurrently. Prefer one active heavy run at a time, because overlapping jobs can exhaust VRAM and destabilize the workstation.
- If desktop applications are already consuming significant VRAM, bias toward lighter validation settings such as fewer environments, headless mode, no extra camera sensors, or a single targeted repro script.

Python environment:

- Prefer the Isaac Lab virtual environment at `../IsaacLab/env_isaaclab`.
- Before running Isaac-dependent commands from `ambench/`, activate it once with `source ../IsaacLab/env_isaaclab/bin/activate`.
- After activation, use plain `python ...` for repo scripts.
- Running `../IsaacLab/env_isaaclab/bin/python` directly without activation may miss the Isaac Sim environment setup that activation provides.
- Do not use `../IsaacLab/_isaac_sim/python.sh` or other kit-Python entrypoints for normal repo validation from `ambench/`; they can miss the packages installed in `env_isaaclab` and produce misleading import/startup failures.
- For Isaac Lab or Isaac Sim Python scripts that touch `isaaclab`, `pxr`, or Omniverse app state, prefer files that use the standard `AppLauncher` bootstrap before the rest of the script runs.
- For ad hoc Isaac-related snippets, activate `env_isaaclab`, run plain `python`, and initialize `AppLauncher` before importing or using `pxr`/Isaac Sim APIs.

Install AM-Bench and its controller dependencies:

```bash
./scripts/setup/install.sh
export JAX_PLATFORMS=cpu
export ACADOS_SOURCE_DIR="$(pwd)/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
```

List registered environments:

First activate the environment once:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
```

Then run commands with `python`:

```bash
python scripts/environments/list_envs.py
```

Smoke-test an environment:

```bash
python scripts/environments/zero_agent.py --task <env-id>
```

`zero_agent.py` is a continuous runner. For bounded validation, wrap it in `timeout`, confirm that scene setup/reset/stepping started, and then ensure no child Isaac process remains before reporting completion.

Interactive inspection:

```bash
python scripts/environments/teleop_se3_agent.py --task <env-id>
```

Validate recorded demos:

```bash
python scripts/data/validate_lerobotdataset.py ...
```

Run formatting/hooks:

```bash
pre-commit run --all-files
```

If Isaac Sim, Isaac Lab, GUI, GPU, or datasets are unavailable, say exactly what you could not verify.

## Release Boundary

Release pruning removed the non-benchmark tasks and their assets, the internal
experiment and demo scripts, and the internal planning and tooling directories.
The task registry is now exactly the 12 task families described in the paper.
Do not reintroduce that material without a deliberate decision.

The registry is the authority on what ships: `python scripts/environments/list_envs.py`
prints it, and the [Environment Registry](https://ambench.github.io/docs/reference/environments/)
is the written reference.

Generated artifacts (datasets, checkpoints, videos, wandb runs, acados build
output) are covered by `.gitignore` and must stay untracked.

Areas that require deliberate review before changing their public support boundary:

- `source/ambench/ambench/utils/camera_utils.py`
  Reason: camera support and its public support boundary need file-level review.
- `source/ambench/ambench/utils/image_processing.py`
  Reason: image-processing support and dependencies need file-level review.
- `ext/`
  Reason: third-party dependencies are curated individually; do not treat the directory as a monolith.

## Tech Stack And Constraints

- Python `>=3.11`
- Isaac Lab project structure
- Isaac Sim runtime and GUI workflows
- Local packages:
  - `source/ambench`
  - `source/ambench_learn`
- External dependencies and submodules:
  - `ext/pyroki`
  - `ext/acados`
  - `ext/openpi`

Prefer Isaac Lab APIs over direct Isaac Sim APIs unless the task clearly needs lower-level Isaac Sim behavior.
Use `CODING_STYLE.md` as the enforced style guide for edits in this repo.

## Repository Map

Simulation package:

- `source/ambench/ambench/tasks/`: environments
- `source/ambench/ambench/robots/`: robot definitions/configs
- `source/ambench/ambench/controllers/`: controllers and controller utilities
- `source/ambench/ambench/policies/scripted/`: scripted policies
- `source/ambench/ambench/scenes/`: scene construction and object placement
- `source/ambench/ambench/disturbance/`: aerodynamic disturbance and randomization
- `source/ambench/ambench/recording/`: demonstration recording and dataset writing
- `source/ambench/ambench/evaluation/`: tracking metrics and evaluation utilities
- `source/ambench/ambench/assets/`: local runtime assets
- `source/ambench/ambench/utils/`: shared utilities

Policy learning package:

- `source/ambench_learn/ambench_learn/data/`: action semantics and dataset handling
- `source/ambench_learn/ambench_learn/policies/`: ACT, Diffusion Policy, and OpenPI integrations
- `source/ambench_learn/ambench_learn/eval/`: rollout execution and result aggregation
- `source/ambench_learn/ambench_learn/utils/`: shared utilities

Scripts:

- `scripts/environments/`: env listing and smoke tests
- `scripts/data/`: record, validate, and convert demos

## How To Work In This Repo

### Tasks

Most environment work belongs in:

- `source/ambench/ambench/tasks/<task_name>/`

Reusable scene components belong in:

- `source/ambench/ambench/scenes/`

Use this for task-agnostic scene spawners, room/wall helpers, and scene-level event terms. Keep task-specific asset lists, counts, placement parameters, and task rewards in the task package.

For direct RL tasks, follow the existing split:

- `__init__.py`: registration
- `<task_name>_env.py`: environment logic
- `<task_name>_env_cfg.py`: config
- optional `<task_name>_events.py`: custom events/domain randomization

Keep responsibilities separated:

- `BaseEnv` and the selected control pipeline: robot/camera setup, action preprocessing, and control application
- task `_setup_scene`: task-owned objects and surrounding geometry
- `_get_observations`: observations
- `_get_rewards`: rewards
- `_get_success`: final and named subtask success criteria
- `_reset_idx`: task state reset after calling `super()`

Prefer matching nearby tasks first:

- `cabinet_pick_place`
- `frame_assembly`
- `peg_in_hole`
- `open_door`
- `wipe_window`

Room and surrounding-scene conventions:

- Prefer parametric Isaac Lab scene spawners for simple rooms over checked-in fixed room USDs.
- Keep room geometry, collision, fallback materials, and surface names in the generic scene config.
- Keep visual domain-randomization source lists in the task config/defaults unless they are truly reusable.
- Keep visual material randomization separate from physics material, collision, friction, and geometry.
- Use one config-level room prim path as the source of truth; reset-time events should derive concrete env paths from that config instead of duplicating room names.
- For navigation-visible backgrounds, prefer actual room geometry, wall/floor materials, and USD scene geometry. Do not use visible dome/HDR backgrounds as the main navigation background because they do not provide translation parallax.

Asset strategy for clutter and distractors:

- Prefer referencing existing NVIDIA/Isaac Sim/Isaac Lab assets or useful sub-prim groups before importing generated assets.
- Prefer small local USD wrapper/reference files over copying large upstream assets into this repo.
- Git LFS is disabled in this repo, so ask before adding large binary assets or generated asset bundles.
- When deriving local wrapper assets from upstream scenes, document source scene paths, chosen prims, approximate sizes, and any known collision/material caveats.

### Controllers

Controller code belongs in:

- `source/ambench/ambench/controllers/`

Follow the existing controller shape:

- initialize gains and physical parameters
- return public `ControllerOutput` data from `compute(...)`
- expose `reset(...)`

Controller math should stay reusable across tasks. Do not bury task-specific reward or reset logic inside controller classes.
Compose controllers with an explicit `ControlPipelineCfg` and `RobotProfileCfg`; do not add controller-selection booleans to task configs.

Start with nearby references:

- `pid_6dof_ctrl.py`
- `wholebody_mpc_ctrl.py`
- `pyroki_ik_ctrl.py`
- `controllers/utils/`

### Robots

Robot code belongs in:

- `source/ambench/ambench/robots/`

Keep robot config separate from task logic. If a change touches runtime assets, keep the final assets under `source/ambench/ambench/assets/`.
Define morphology through `RobotSpecCfg` and let `RobotIO` resolve live body/joint handles. Do not introduce a `RobotType` enum or robot-name branches in `BaseEnv`.

### Policies And IL

Use this split:

- scripted policies: `source/ambench/ambench/policies/scripted/`
- learned policies and IL code: `source/ambench_learn/ambench_learn/`

Training and evaluation entrypoints are module entrypoints inside the policy
family, run as `python -m ambench_learn.policies.<family>.<train|eval>`, not
standalone launcher scripts.

Keep demo collection and replay logic in `scripts/data/` or dedicated data utilities, not embedded in environment classes.

## Preferred Patterns

Prefer concrete extensions of existing code over new abstractions.
Prefer inline local logic over one-off private helper methods when the logic is only used once and remains readable in place.

For repository bash entrypoints, prefer a short explicit configuration block near the top of the file followed by the main command invocation.
Avoid environment-variable default/override patterns such as `FOO="${FOO:-...}"` and avoid sourced preset layers for routine script configuration.
If multiple concrete launch variants are needed, prefer separate scripts or clearly separated commented config blocks over dynamic bash indirection.

Good task layout:

```text
source/ambench/ambench/tasks/my_task/
  __init__.py
  my_task_env.py
  my_task_env_cfg.py
  my_task_events.py
```

Good task profile pattern:

```python
@configclass
class MyTaskEnvFAHexaAbsPIDCfg(MyTaskEnvDefaultCfg):
    robot_profile = FA_HEXA_ABS_PID
```

Good reset pattern:

```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    # reset task state here
```

Avoid:

- mixing scene setup, reward logic, and controller internals in one block
- inventing a new directory layout for a task that matches existing direct task patterns
- adding hard-coded local paths for assets, logs, or datasets
- introducing excessive private helper methods, especially single-use `_{name}()` helpers that only wrap a small local branch or one call site
- splitting small, readable local control flow into separate private helpers solely for “cleanliness”

## Validation Expectations

Run the smallest meaningful validation for the change.

Typical order:

1. import or registration-level check
2. environment smoke test
3. data replay or policy-specific check if relevant
4. formatting or hook pass

Mention whether validation was:

- not run
- import-only
- smoke-test level
- partial functional test
- full task-specific verification

## Git And Change Hygiene

- Check for existing user changes before editing overlapping files.
- Keep diffs focused.
- Do not casually edit submodules under `ext/`.
- Do not revert unrelated local changes.
- Do not commit machine-specific overrides just to make one workstation run.

## Always / Ask / Never

Always:

- inspect nearby code before changing structure
- preserve Isaac Lab-first architecture
- keep task, controller, robot, and IL responsibilities separated
- state what you validated and what remains unverified

Ask first:

- changes to submodules in `ext/`
- large asset imports or asset format conversions
- long-running training jobs
- expensive downloads or destructive cleanup
- repo-wide refactors that touch multiple task families

Never:

- commit `datasets/`, checkpoints, videos, wandb outputs, or generated experiment artifacts
- hard-code absolute machine-specific paths
- rewrite large working modules without a concrete need
- use direct Isaac Sim APIs when an established Isaac Lab pattern already exists locally

Git LFS is disabled in this repo.

## Useful Reference Paths

- `README.md`
- `docs/extend/index.md`
- `docs/extend/robot.md`
- `docs/extend/controller.md`
- `docs/extend/task.md`
- `docs/extend/policy.md`
- `docs/workflows/index.md`
- `source/ambench/ambench/tasks/`
- `source/ambench/ambench/controllers/`
- `source/ambench/ambench/robots/`
- `source/ambench/ambench/policies/scripted/`
- `source/ambench_learn/ambench_learn/`

## When Unsure

- inspect the nearest similar implementation
- choose the simpler change
- preserve the current structure
- leave a concise summary of assumptions, validation, and remaining risks
