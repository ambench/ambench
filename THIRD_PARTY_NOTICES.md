# Third-Party Notices

AM-Bench is distributed under the Apache License 2.0 (see [`LICENSE`](LICENSE)).
It redistributes and depends on the third-party components listed below, each of
which remains under its own license.

## Redistributed in this repository

These components are vendored into the source tree and ship with AM-Bench.

### Universal Manipulation Interface / Diffusion Policy

- **Path:** `source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface/`
- **License:** MIT — see the [`LICENSE`](source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface/LICENSE) in that directory
- **Copyright:** Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
- **Upstream:** <https://github.com/real-stanford/universal_manipulation_interface>,
  <https://github.com/real-stanford/diffusion_policy>

### imagecodecs-numcodecs

- **Path:** `source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface/diffusion_policy/codecs/imagecodecs_numcodecs.py`
- **License:** BSD-3-Clause (full text retained in the file header)
- **Copyright:** Copyright (c) 2021-2022, Christoph Gohlke

### Isaac Lab (derived files)

A small number of files began as Isaac Lab extension templates or example scripts
and remain under Isaac Lab's BSD-3-Clause license. Each carries both the original
Isaac Lab copyright and an AM-Bench copyright:

- `.pre-commit-config.yaml`
- `.vscode/tools/setup_vscode.py`
- `scripts/environments/list_envs.py`
- `scripts/environments/teleop_se3_agent.py`
- `scripts/environments/zero_agent.py`
- `source/ambench/setup.py`
- `source/ambench_learn/setup.py`

- **License:** BSD-3-Clause
- **Copyright:** Copyright (c) 2022-2025, The Isaac Lab Project Developers
- **Upstream:** <https://github.com/isaac-sim/IsaacLab>

## Fetched as submodules or external dependencies

These are not redistributed in this repository. They are fetched into `ext/` and
remain under their own licenses.

| Component | License | Upstream |
| --- | --- | --- |
| acados | BSD-2-Clause | <https://github.com/acados/acados> |
| Pyroki | MIT | <https://github.com/chungmin99/pyroki> |
| openpi | Apache-2.0 | <https://github.com/Physical-Intelligence/openpi> |

Pyroki is fetched from `https://github.com/ambench/pyroki`, a fork of the
upstream listed above. The fork adds two cost functions,
`smoothness_cost_constant` and `smoothness_cost_se3_const`, which the
whole-body IK controller uses. It remains MIT licensed.

The openpi submodule still points at a personal fork rather than the upstream
repository listed above. The license is unchanged, but the pinned revision
differs from upstream.

## Simulation assets

USD, mesh, and texture assets under `source/ambench/ambench/assets/` are
covered by AM-Bench's license except where a bundled asset carries its own terms.
Assets sourced from third-party libraries retain their original licenses.
