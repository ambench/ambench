# Development Setup

Optional tooling for working in the repository. None of it is required to run
AM-Bench; see [Installation](../getting-started/installation.md) for that.

## VSCode

Press `Ctrl+Shift+P`, select `Tasks: Run Task`, and run `setup_python_env`. You will be
prompted for the absolute path to your Isaac Sim installation. This writes
`.vscode/.python.env` with the Python paths for all Isaac Sim and Omniverse extensions,
which enables module indexing and completion.

For better code analysis, add the Isaac Lab sources to `"python.analysis.extraPaths"` in
`.vscode/settings.json`:

```json
"python.analysis.extraPaths": [
    "${workspaceFolder}/../IsaacLab/source/isaaclab",
    "${workspaceFolder}/../IsaacLab/source/isaaclab_assets",
    "${workspaceFolder}/../IsaacLab/source/isaaclab_mimic",
    "${workspaceFolder}/../IsaacLab/source/isaaclab_rl",
    "${workspaceFolder}/../IsaacLab/source/isaaclab_tasks",
]
```

## Code formatting

This repository uses pre-commit to format code automatically:

```bash
uv pip install pre-commit
pre-commit run --all-files
```

`CODING_STYLE.md` is the enforced style guide, and `AGENTS.md` describes the repository
conventions.

## Building the documentation

Public documentation source lives in `docs/` and is built with Zensical. To preview the
site locally, install the pinned version in the same Isaac Lab environment:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install "zensical==0.0.58"
zensical serve
```

Then open <http://127.0.0.1:8000/docs/>.

## Simulator performance

See the [Isaac Sim Performance Optimization Handbook](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html#cpu-governor-settings-on-linux)
for CPU governor settings. On a discrete NVIDIA GPU, also check the power mode,
performance state, and persistence service, plus any power settings specific to your
machine.

Before starting GPU-heavy work, run `nvidia-smi` to check available memory and active
processes. Prefer one small environment at a time; overlapping heavy simulator jobs can
exhaust VRAM.
