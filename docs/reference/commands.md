# Commands and Scripts

This reference lists maintained public entrypoints. Run repository Python commands from the AM-Bench root after activating the Isaac Lab environment:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
```

Commands that create an environment require Isaac Sim and a compatible NVIDIA GPU. Dataset-only validation and conversion do not launch Isaac Sim, but require their policy/data dependencies.

## Environment commands

| Entrypoint | Purpose | Important arguments and behavior |
| --- | --- | --- |
| `python scripts/environments/list_envs.py` | Print registered AM-Bench IDs, entrypoints, and config classes | Headless; launches Isaac Sim; no command-specific arguments |
| `python scripts/environments/zero_agent.py` | Continuously step an environment with neutral actions | required `--task`; optional `--num_envs`, `--video`, `--camera_names`, `--disable_fabric`, and AppLauncher options |
| `python scripts/environments/teleop_se3_agent.py` | Control an environment interactively | required `--task`; `--num_envs 1`, `--teleop_device keyboard`, and `--sensitivity 0.2` defaults; device increments are integrated into absolute EE targets |

`zero_agent.py` and teleoperation run until the application closes. Use the bounded commands in [Verify Installation](../getting-started/first-run.md) and [Run an Environment](../workflows/run-environment.md) when a finite smoke test is required.

## Recording and dataset commands

Script argument names in this section use their current underscore spelling.

| Entrypoint | Purpose | Required input | Primary output |
| --- | --- | --- | --- |
| `python scripts/data/record_demos_scripted.py` | Record task-aware scripted demonstrations | `--task` | timestamped session containing `env_cfg.yaml` and canonical `lerobot/` data |
| `python scripts/data/record_demos_teleop.py` | Record human teleoperation demonstrations | `--task` | same canonical session layout |
| `python scripts/data/validate_lerobotdataset.py` | Statically validate finalized canonical data | `--dataset_root`; optional `--repo_id` and `--target_hz` | console validation report and exit status |
| `python scripts/data/dp/lerobot_to_zarr.py` | Convert canonical data to UMI zarr | `--input_path`; optional `--output_path` | `.zarr.zip` or zarr directory |
| `python scripts/data/dp/validate_zarr.py` | Validate a DP zarr conversion | positional `zarr_path`; optional `--image_size` | console validation report and exit status |
| `python scripts/data/export_lerobot_to_openpi.py` | Export one or more canonical roots for the pinned OpenPI reader | `--dataset_roots`; optional `--repo_id`, `--output_root`, and prompt mapping | derived local LeRobot v2.1 dataset |

Both recorders accept `--dataset_root`, `--step_hz`, `--num_demos`, `--env_length_s`, `--camera_names`, `--video`, `--repo_id`, `--state_keys`, and `--task_prompt`. The scripted recorder additionally accepts `--num_envs`, `--inject_noise`, and `--save_failed_episodes`; teleoperation accepts `--teleop_device` and `--sensitivity`.

Conversion outputs are derived artifacts. Do not overwrite the canonical source dataset. See [Collect Demonstrations](../workflows/collect-demos.md), [Validate Datasets](../workflows/validate-data.md), and [Dataset Format](dataset-format.md).

## Learned-policy entrypoints

| Policy | Training or server entrypoint | Evaluation entrypoint |
| --- | --- | --- |
| ACT | `python -m ambench_learn.policies.act.train` | `python -m ambench_learn.policies.act.eval` |
| Diffusion Policy | `python train.py` from the pinned UMI directory with a Hydra config | `python -m ambench_learn.policies.dp.eval` |
| OpenPI | pinned `ext/openpi` compute-stats, PyTorch training, and policy-server modules | `python -m ambench_learn.policies.pi.eval` |

The three learned-policy evaluators use a shared kebab-case interface. All require `--task`; ACT and DP require `--checkpoint`, while OpenPI requires `--prompt` and a reachable server. See [Configure Evaluation](../evaluation/configure.md) for exact shared defaults and the individual [policy guides](../policies/index.md) for training and policy-specific commands.

## Completion evidence

Environment smoke tests must reach scene creation, reset, and stepping. Dataset validators must exit successfully and print their passing message. A learned-policy evaluation is complete only when `eval_summary.json` reports `status: "completed"` and the requested rollout count; see [Evaluation Outputs](evaluation.md).
