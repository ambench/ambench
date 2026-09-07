# Installation

This guide installs AM-Bench into an existing Isaac Lab Python environment. When it is complete, the `ambench` and `ambench_learn` packages are installed in `env_isaaclab`; runtime verification is the next, separate step.

## Requirements

- native Linux; Ubuntu 22.04 and 24.04 are the maintained host targets;
- an NVIDIA GPU and driver compatible with the selected Isaac Sim release;
- Isaac Sim 5.1 with an Isaac Lab checkout;
- Python 3.11 in the Isaac Lab environment;
- `uv` for editable package installation.

WSL, macOS, and CPU-only simulation are outside the maintained runtime boundary. Remote and containerized setups can work, but the host must provide compatible NVIDIA graphics, Vulkan access, filesystem mounts, and any required port forwarding.

## 1. Install Isaac Sim and Isaac Lab

Follow the [Isaac Lab binary installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html). Create the Python environment expected by that checkout.

For Isaac Sim 5.1, activate the environment and confirm that it uses Python 3.11:

```bash
source <path-to-IsaacLab>/env_isaaclab/bin/activate
python --version
```

Do not use Isaac Sim's Kit Python launcher for normal AM-Bench commands. Activating `env_isaaclab` also configures the Isaac runtime environment required by repository scripts.

## 2. Clone AM-Bench beside Isaac Lab

Clone the current source repository beside the Isaac Lab checkout:

```bash
cd <workspace-root>
git clone --recurse-submodules https://github.com/ambench/ambench.git
cd ambench
```

The commands in these docs assume this layout:

```text
<workspace-root>/
  IsaacLab/
  ambench/
```

If the directories are elsewhere, replace `../IsaacLab` with the path to your Isaac Lab checkout.

## 3. Install the local packages

From the AM-Bench repository root:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e source/ambench_learn
```

The simulation package provides environments, robots, controllers, recording, and scripted policies. The IL package provides learned-policy and dataset utilities.

AM-Bench is the product name; `ambench` and `ambench_learn` are the Python package names used by imports and editable installation paths.

Continue with [Verify Installation](first-run.md). That check starts Isaac Sim, confirms environment registration, creates one scene, and steps it for a bounded interval.

## Optional: LeRobot and policy extras

The canonical dataset boundary uses LeRobot 0.4.4. Install it without allowing its dependency resolver to replace Isaac Lab's validated Torch and NumPy stack:

```bash
uv pip install --no-deps "lerobot==0.4.4"
```

Install a learned-policy extra only when needed:

```bash
uv pip install -e "source/ambench_learn[act]"
uv pip install -e "source/ambench_learn[dp]"
```

OpenPI uses its own pinned environment for training and serving; see [OpenPI](../policies/openpi.md).

## Optional: Pyroki inverse kinematics

Physical manipulator variants that accept end-effector targets use the Pyroki integration:

```bash
git submodule update --init --recursive ext/pyroki
uv pip install -e ext/pyroki
export JAX_PLATFORMS=cpu
uv pip install "numpy==1.26.0" "jax==0.4.28" "jaxlib==0.4.28"
```

The maintained configuration runs JAX on CPU while Isaac Sim uses the NVIDIA GPU. GPU-backed JAX is not part of the validated setup.

## Optional: acados whole-body MPC

MPC variants require the `ext/acados` submodule and its Python interface:

```bash
git submodule update --init --recursive ext/acados
cd ext/acados
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
cd ../../..
uv pip install -e ext/acados/interfaces/acados_template
export ACADOS_SOURCE_DIR="$(pwd)/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
```

Generated acados build output is machine-local and must not be committed.
