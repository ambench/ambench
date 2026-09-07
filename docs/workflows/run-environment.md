# Run an Environment

This guide runs a registered AM-Bench environment independently of a learned policy. Use it to inspect scene startup, observations, actions, cameras, and simulator performance before adding data collection or evaluation.

Complete [Verify Installation](../getting-started/first-run.md), then choose an exact ID with [Find an Environment](../getting-started/find-an-environment.md). Run the commands below from the repository root with `env_isaaclab` active.

## Run a bounded headless session

Check GPU availability before starting Isaac Sim:

```bash
nvidia-smi
```

Run one environment for a bounded interval:

```bash
timeout --signal=INT 30s python scripts/environments/zero_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 \
  --headless \
  --device cuda:0
```

Replace `--task` only with an ID printed by `scripts/environments/list_envs.py`. The script prints the observation and action spaces, then steps continuously with a neutral action until interrupted. Task completion is not expected.

## Open a simulator window

Omit `--headless` when the machine has a working graphical session:

```bash
python scripts/environments/zero_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 \
  --device cuda:0
```

Close the Isaac Sim window or interrupt the process to stop the run. A remote machine requires a supported display or streaming setup; a successful headless run does not prove that GUI rendering is configured.

## Record a camera check

Enable video only after the basic headless session works:

```bash
timeout --signal=INT 30s python scripts/environments/zero_agent.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --num_envs 1 \
  --video \
  --camera_names ee_camera \
  --headless \
  --device cuda:0
```

The script prints `VIDEO_PATH` after a clean shutdown. Video confirms that the named camera and encoder work; it does not validate task success. Camera availability depends on the selected robot profile.

## Scale only after one environment works

Increase `--num_envs` only after the one-environment run is stable. More environments and cameras increase GPU memory use. On a single-GPU machine, keep one heavy Isaac Sim process active at a time and confirm that the previous process has exited before starting another.

Continue with [Collect Demonstrations](collect-demos.md) to record data, [Evaluate a Supported Policy](../evaluation/index.md), or [Troubleshooting](../help/troubleshooting.md) when startup or stepping fails.
