# Configure Evaluation

ACT, Diffusion Policy, and OpenPI share rollout, environment, disturbance, video, progress, and output controls. Policy-specific arguments remain on their own evaluators.

## Shared arguments

| Argument | Default | Effect and constraint |
| --- | --- | --- |
| `--task` | none | Required exact Gym environment ID |
| `--num-envs` | `1` | Parallel environments; must be at least 1; DP requires exactly 1 |
| `--num-rollouts` | `10` | Total completed rollout records requested; must be at least 1 |
| `--episode-length-s` | task config | Positive override of `env_cfg.episode_length_s` |
| `--seed` | unset | Base reset seed; later batches add their starting rollout index |
| `--disturbance` | off | Enables saturation, aerodynamic effects, and wind together |
| `--save-video` | off | Writes MP4 rollout videos under `<run-dir>/videos/` |
| `--video-camera-names` | `ee_camera scene_camera` | Space-separated sensor names; every requested sensor must exist |
| `--output-dir` | generated path | Exact run directory; otherwise uses `outputs/eval/<policy>/<task>/<timestamp>/` |
| `--progress-every` | `100` | Positive number of environment steps between progress lines |
| `--headless` | launcher default | Runs without the Isaac Sim window when supplied |
| `--device` | launcher default | Selects the Isaac simulation and policy device, such as `cuda:0` |

Argument names shown here use the public kebab-case interface.

## Policy-specific arguments

| Policy | Arguments |
| --- | --- |
| ACT | required `--checkpoint`; optional `--n-action-steps`, `--temporal-ensemble-coeff`, and `--policy-target-hz` |
| Diffusion Policy | required checkpoint file through `--checkpoint`; execution schedule comes from the saved Hydra configuration |
| OpenPI | required `--prompt`; `--host localhost`, `--port 8000`, `--n-action-steps 8`, optional `--policy-target-hz` and `--policy-id`, and `--connect-timeout-s 10.0` |

ACT accepts either a checkpoint directory containing `pretrained_model/` or that directory itself. DP requires a checkpoint file. OpenPI does not load a local checkpoint in the simulator process: its server owns the config and checkpoint, while `--policy-id` records a human-readable identifier in the evaluation metadata.

## Keep runs comparable

Keep the task ID, policy/checkpoint identity, action representation, policy rate, seed policy, rollout count, episode length, disturbance setting, requested cameras, and controller profile fixed. Use a separate output directory for every run and retain the exact command with the artifacts.

`--disturbance` is a bundled evaluation switch. To study individual physical effects, configure the environment fields directly as described in [Physics, Disturbances, and Randomization](../configure/physics-disturbances-and-randomization.md).
