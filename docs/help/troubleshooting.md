# Troubleshooting

Start at the earliest layer that fails. A task, controller, or policy change cannot repair an environment-activation, driver, simulator, registration, or dataset-format problem.

## Installation

### `ModuleNotFoundError: ambench` or `ambench_learn`

Activate the Isaac Lab environment and reinstall both editable packages from the repository root:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e source/ambench_learn
```

Do not invoke `../IsaacLab/env_isaaclab/bin/python` without activation. The activation script also configures the Isaac runtime environment.

### Isaac Sim, Isaac Lab, or `pxr` import fails

Use repository entrypoints that initialize `AppLauncher` before importing simulator APIs. Do not use the Kit Python launcher for ordinary AM-Bench commands.

If an official Isaac Lab example fails in the same activated environment, diagnose the NVIDIA driver, Vulkan access, Isaac Sim, and Isaac Lab installation before changing AM-Bench. Return to [Installation](../getting-started/installation.md) and [Verify Installation](../getting-started/first-run.md).

## Registration

### The task ID is not found

Do not construct an ID by substituting task, robot, action, or controller tokens. Print the current registry:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python scripts/environments/list_envs.py
```

Copy the complete ID, including capitalization, `Direct`, and `v0`. If the expected ID is absent, confirm that `ambench` is installed from the current checkout and compare it with the [Environment Registry](../reference/environments.md).

## Simulator startup

### Renderer startup fails or GPU memory is exhausted

Inspect available GPUs and active processes:

```bash
nvidia-smi
```

Run one heavy Isaac process at a time on a single-GPU machine. Begin with `--num_envs 1 --headless`, omit unnecessary cameras, and select a device such as `--device cuda:0`. GUI sessions and camera sensors increase graphics-memory use.

### Pyroki or JAX selects the wrong backend

The maintained integration uses JAX on CPU while Isaac Sim uses the NVIDIA GPU. Set the backend before starting Python:

```bash
export JAX_PLATFORMS=cpu
```

Confirm the NumPy, JAX, and JAXlib versions specified by [Installation](../getting-started/installation.md). Passing an EE-oracle task does not validate physical-manipulator IK.

### The acados solver cannot load

Confirm that acados was built, its Python interface is installed, and both runtime paths are set in the same shell:

```bash
export ACADOS_SOURCE_DIR="$(pwd)/ext/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
```

Generated solver code is machine-local. Rebuild it on the target machine instead of copying a build directory from another system.

## Scene and task behavior

### `Requested video camera(s) are unavailable`

The EE oracle exposes `ee_camera`. Physical profiles may also expose `base_camera`; `scene_camera` exists only when recording or evaluation configures it. Use the available sensor names printed by the evaluator, remove an unavailable name, or choose a profile that provides it.

### The scene starts but task behavior is wrong

Reproduce the behavior with one environment and the EE oracle before adding physical IK or base control. Check the exact task config, reset seed, action mode, object poses, event ranges, and success/subtask thresholds. A successful scene load does not prove that contacts, resets, or success conditions are valid; inspect a bounded video and compare the implementation with [Tasks and Scenes](../configure/tasks-and-scenes.md).

## Data

### `LeRobot dataset validation failed` or episode metadata is missing

Pass either the recorder session root or its `lerobot/` child. A finalized dataset contains `meta/info.json` and episode metadata under `meta/episodes/`. An interrupted recorder may leave an incomplete dataset that must not be used for training.

Run the exact validator command from [Validate Datasets](../workflows/validate-data.md). Do not repair a derived DP or OpenPI dataset in place; correct or replace the canonical source and regenerate the conversion.

### The dataset passes static validation but training behavior is wrong

Static validation does not prove demonstration quality, camera usefulness, task success correctness, or checkpoint compatibility. Inspect representative boundary frames and rollout videos, then confirm the recorded `action_semantics`, ordered `state_keys`, camera keys, task prompt, and logical policy rate.

## Policy

### ACT cannot find `pretrained_model`

Pass either the checkpoint directory containing `pretrained_model/` or the `pretrained_model/` directory itself. Confirm that the selected checkpoint contains the representation metadata expected by the ACT evaluator.

### `DP evaluation requires --num-envs 1`

Set `--num-envs 1`. The current UMI observation-history implementation is not vectorized.

### `Could not connect to OpenPI server`

Start the pinned OpenPI policy server and wait for `server listening`. Confirm `--host`, `--port`, routing or port forwarding, and firewall access before starting the Isaac client. The server and simulator may use separate Python environments; follow [OpenPI](../policies/openpi.md).

### A checkpoint loads but actions are wrong

Compare the exact task ID, environment action mode, source `action_semantics`, model action representation, policy rate, state-key order, camera keys, and WXYZ quaternion convention. Matching tensor dimensions do not establish semantic compatibility.

## Evaluation and shutdown

### Evaluation files exist but the run is incomplete

Open `eval_summary.json`. Require `status: "completed"` and confirm that `summary.num_rollouts` equals the requested count. `stopped`, `interrupted`, and `failed` runs may contain useful partial artifacts but are not complete evaluations. See [Evaluation Outputs](../reference/evaluation.md).

### The process does not exit after `timeout`

`zero_agent.py` is continuous. Use `timeout --signal=INT`, allow Isaac Sim to shut down, and confirm that no child Isaac process remains before starting another run. A forced kill can leave incomplete videos or datasets.

If the failure remains reproducible, follow [Report an Issue](report-an-issue.md).
