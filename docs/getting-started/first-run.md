# Verify Installation

This tutorial verifies package registration, scene creation, reset, and stepping with one bounded headless run. It uses the EE-only oracle so the check does not depend on Pyroki or acados.

!!! info "Prerequisites"

    Complete [Installation](installation.md), then run every command below from the AM-Bench repository root.

## 1. Activate the environment and inspect the GPU

```bash
source ../IsaacLab/env_isaaclab/bin/activate
nvidia-smi
```

Confirm that the selected GPU has enough free memory and that no unexpected Isaac Sim process is already running.

## 2. Verify environment registration

```bash
python scripts/environments/list_envs.py
```

The command starts Isaac Sim headlessly and prints the registered AM-Bench IDs, environment classes, and configuration entry points. Confirm that `PressButton-Am-EE-Abs-PID-Direct-v0` appears in the table.

If you need a different task or profile, use [Find an Environment](find-an-environment.md) before continuing.

## 3. Run the bounded smoke test

`zero_agent.py` is a continuous runner, so use `timeout` for this verification:

```bash
timeout --signal=INT 30s python scripts/environments/zero_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

The verification passes when:

1. Isaac Sim starts without a driver or renderer failure;
2. the task scene resets and prints its observation and action spaces;
3. the environment steps repeatedly until `timeout` sends the interrupt;
4. Isaac Sim shuts down and leaves no child process running.

!!! note "Validation boundary"

    The zero-action agent is not expected to solve the task. A successful smoke test proves that this environment can be created and stepped on the current machine; it does not validate task success, policy quality, optional physical-robot dependencies, or every registered environment.

See [Troubleshooting](../help/troubleshooting.md) if the process fails before stepping or does not shut down cleanly. Once the check passes, continue with [Run an Environment](../workflows/run-environment.md).
