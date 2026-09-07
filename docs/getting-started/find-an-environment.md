# Find an Environment

Use the live Gym registry to choose an environment ID that is available in the current checkout. Do not construct an ID by assuming that every task, robot, action mode, and controller combination is registered.

## List registered environments

From the AM-Bench repository root:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python scripts/environments/list_envs.py
```

The command starts Isaac Sim headlessly and prints one row per registered AM-Bench environment. Each row includes the environment ID, environment class, and configuration entry point.

## Choose an ID

Start with `PressButton-Am-EE-Abs-PID-Direct-v0` when verifying task registration and scene stepping. It uses the EE-only oracle and does not require Pyroki or acados.

For another experiment, select an exact ID from the command output. The usual pattern is:

```text
<Task>-Am-<Robot>-<Action>-<Controller>-Direct-v0
```

The tokens identify the task, robot profile, public action interface, and controller family. Some combinations add `BaseJoint-Abs` or a specialized suffix, and not every theoretical combination exists.

Use [Environment IDs](../reference/environments.md) to interpret each token and compare registered variants. After selecting an ID, pass it unchanged to [Run an Environment](../workflows/run-environment.md) or a supported policy evaluator.
