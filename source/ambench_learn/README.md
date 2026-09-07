# `ambench_learn` — AM-Bench Policy Learning Extension

Imitation learning, data pipelines, and policy evaluation for
[AM-Bench](https://ambench.github.io/) aerial manipulation tasks.

## Contents

| Module | Purpose |
| --- | --- |
| `data/` | Action semantics, resampling, and LeRobot dataset handling |
| `pipeline/` | Data collection and conversion pipeline configuration |
| `policies/act/` | ACT training and evaluation |
| `policies/dp/` | Diffusion Policy training and evaluation |
| `policies/pi/` | OpenPI (π₀ / π₀.₅) evaluation adapters |
| `processor/` | Observation and action processing |
| `eval/` | Rollout execution and result aggregation |

## Installation

Install into an existing Isaac Lab environment, after `ambench`:

```bash
uv pip install -e source/ambench_learn
```

Policy-specific extras are available:

```bash
uv pip install -e "source/ambench_learn[act]"   # ACT
uv pip install -e "source/ambench_learn[dp]"    # Diffusion Policy
uv pip install -e "source/ambench_learn[all]"   # everything
```

See the [policy guides](https://ambench.github.io/docs/policies/) for training
and evaluation workflows.

## License

Apache-2.0. See [`LICENSE`](../../LICENSE) at the repository root.

This package vendors the Universal Manipulation Interface / Diffusion Policy
source tree under `ambench_learn/policies/dp/universal_manipulation_interface/`,
which is MIT licensed by the Columbia Artificial Intelligence and Robotics Lab.
See [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
