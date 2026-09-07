# Add a Task

This guide adds a direct task using the same split as the maintained benchmark families. Complete [Installation](../getting-started/installation.md) and [Verify Installation](../getting-started/first-run.md) first; the result is a task package with at least one registered environment ID and a bounded validation path.

## 1. Choose a nearby task

Match the dominant interaction before copying code:

- `press_button` or `push_slider` for simple contact;
- `peg_in_hole` or `frame_assembly` for alignment and insertion;
- `open_door` or `rotate_valve` for articulated objects;
- `wipe_window` or `ndt` for surface-constrained motion;
- `cabinet_pick_place` or `lemon_harvesting` for transport.

These are maintained packages under `source/ambench/ambench/tasks/`. Read the selected package's `__init__.py`, `*_env_cfg.py`, and `*_env.py` together before creating the new package.

## 2. Create the task package

```text
source/ambench/ambench/tasks/my_task/
  __init__.py
  my_task_env.py
  my_task_env_cfg.py
  my_task_events.py       # only for task-specific events
```

Subclass `BaseEnv`. Keep task-owned assets and surrounding geometry in `_setup_scene`, observations in `_get_observations`, rewards in `_get_rewards`, success and named subtask criteria in `_get_success`, and task state reset in `_reset_idx` after `super()._reset_idx(...)`.

## 3. Compose profiles in config

Define a task default config and small profile subclasses. Select an existing `RobotProfileCfg` constant such as `EE_ABS_PID`, `FA_HEXA_ABS_PID`, or `FA_HEXA_ABS_MPC`; do not add controller-selection booleans to the task.

Register only combinations the task actually supports in `__init__.py` with `register_env(...)`. Provide the task environment entry point, config entry point, and scripted-policy entry point.

## 4. Add a scripted policy

Place task-aware waypoint or state-machine logic in `source/ambench/ambench/policies/scripted/`. The policy should consume task observations and emit the task's configured public action. Keep success criteria authoritative in the environment, not in the policy.

## 5. Validate

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python scripts/environments/list_envs.py
timeout --signal=INT 30s python scripts/environments/zero_agent.py \
  --task MyTask-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

Then collect one scripted demonstration and inspect its video before scaling. Confirm that final success and every named subtask criterion reflect the intended task rather than a proxy such as proximity alone.
