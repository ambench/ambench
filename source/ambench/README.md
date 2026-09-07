# `ambench` — AM-Bench Simulation Extension

Isaac Lab extension providing the aerial manipulation robots, tasks, scenes,
controllers, disturbance models, and data recording used by
[AM-Bench](https://ambench.github.io/).

## Contents

| Module | Purpose |
| --- | --- |
| `assets/` | USD, mesh, and texture assets for robots, objects, and scenes |
| `robots/` | Robot articulation configs and rotor actuator models |
| `tasks/` | Task families and their registered Gym environments |
| `scenes/` | Scene construction and object placement |
| `controllers/` | Low-level control pipelines (PID, L1, MPC) and IK |
| `disturbance/` | Aerodynamic disturbance and randomization models |
| `policies/scripted/` | Scripted expert policies used for demonstration collection |
| `recording/` | Demonstration recording and dataset writing |
| `evaluation/` | Tracking metrics and evaluation utilities |

## Installation

Install into an existing Isaac Lab environment:

```bash
uv pip install -e source/ambench
```

See the [Installation guide](https://ambench.github.io/docs/getting-started/installation/)
for the full procedure.

## License

Apache-2.0. See [`LICENSE`](../../LICENSE) and
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) at the repository root.
