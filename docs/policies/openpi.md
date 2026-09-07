# OpenPI: PI0 and PI0.5

OpenPI uses two processes: the pinned `ext/openpi` checkout trains or serves a policy, and the Isaac Lab environment connects through `openpi-client`. Keep those Python environments separate.

## Initialize and install

```bash
git submodule update --init ext/openpi
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e source/ambench_learn
uv pip install -e ext/openpi/packages/openpi-client
uv pip install --no-deps "lerobot==0.4.4"
cd ext/openpi
uv sync
cd -
```

The simulator uses the activated Isaac Lab environment. Training and serving use `ext/openpi/.venv` through `uv run`. Do not install an unpinned remote `openpi-client` into Isaac Lab.

## Choose storage and GPUs

OpenPI weights, caches, and checkpoints are large. Set explicit locations:

```bash
export OPENPI_STORAGE_ROOT=<openpi-storage-root>
export OPENPI_DATA_HOME=<openpi-storage-root>/cache/openpi
export HF_HOME=<openpi-storage-root>/cache/huggingface
export HF_LEROBOT_HOME=<openpi-storage-root>/cache/huggingface/lerobot
export TORCH_HOME=<openpi-storage-root>/cache/torch
export JAX_COMPILATION_CACHE_DIR=<openpi-storage-root>/cache/jax
export UV_CACHE_DIR=<openpi-storage-root>/cache/uv
export OPENPI_ASSETS_BASE_DIR=<openpi-storage-root>/assets
export OPENPI_CHECKPOINT_BASE_DIR=<openpi-storage-root>/checkpoints
```

Select policy-server GPUs with `CUDA_VISIBLE_DEVICES`. Select the Isaac Sim device with `--device cuda:<id>`. The two processes may run on different hosts if the WebSocket port is reachable.

## Export the validated dataset

The canonical source is LeRobot 0.4.4/v3. The pinned OpenPI reader requires a derived local v2.1 layout:

```bash
python scripts/data/export_lerobot_to_openpi.py \
  --dataset_roots <session-root> \
  --repo_id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --output_root <openpi-dataset-root> \
  --target_hz 20 \
  --omit_base_image \
  --task_prompt "press the button"
```

Keep source and export in separate directories. For multiple tasks, pass multiple roots with `--task_prompt_map scripts/data/am_bench_language_instructions.json --require_task_prompt_map`. Use a BaseJoint OpenPI repository ID and config for BaseJoint exports.

Place the export under `$HF_LEROBOT_HOME/<repo-id>` or set `HF_LEROBOT_HOME` so the selected config resolves it.

## Prepare base weights

The PyTorch trainer requires a converted PI0 or PI0.5 base checkpoint:

```bash
cd ext/openpi
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir <pi0-or-pi05-jax-base-checkpoint> \
  --config_name <pi-config-name> \
  --output_path <pytorch-base-checkpoint>
cd -
```

Confirm that `<pytorch-base-checkpoint>/model.safetensors` exists. Follow the pinned OpenPI README for its Transformers version and patch step; those requirements belong to that checkout.

## Compute normalization statistics

Maintained EE config names are:

- `pi0_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative`
- `pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative`

BaseJoint variants use `multitask_base_joint_openpi_original_20hz_h50_base_joint_relative` in the name.

```bash
cd ext/openpi
uv run python -m scripts.compute_norm_stats \
  --config-name <pi-config-name> \
  --assets-base-dir <openpi-storage-root>/assets \
  --repo-id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --num-workers 2
cd -
```

Do not train or serve until the command writes `norm_stats.json` under the reported asset directory.

## Train

```bash
cd ext/openpi
CUDA_VISIBLE_DEVICES=0 uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  -m scripts.train_pytorch <pi-config-name> \
  --exp-name press_button_pi \
  --pytorch-weight-path <pytorch-base-checkpoint> \
  --assets-base-dir <openpi-storage-root>/assets \
  --checkpoint-base-dir <openpi-storage-root>/checkpoints \
  --data.repo-id am_bench/multitask_openpi_original_20hz_ee_local_relative \
  --batch-size 32 \
  --num-train-steps 10000 \
  --num-workers 2 \
  --save-interval 5000 \
  --no-resume \
  --no-overwrite \
  --no-wandb-enabled
cd -
```

For a smoke test, use `--num-train-steps 2 --save-interval 1`. A valid step contains `model.safetensors`, config metadata, and matching norm statistics under `assets/<repo-id>`.

## Serve

In the OpenPI terminal:

```bash
cd ext/openpi
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.serve_policy \
  --port 8000 \
  policy:checkpoint \
  --policy.config <pi-config-name> \
  --policy.dir <trained-checkpoint-step>
```

Wait for `server listening`. For a bounded smoke where first-request compilation exceeds client keepalive, restart with compilation disabled:

```bash
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0 \
  uv run python -m scripts.serve_policy \
  --port 8000 \
  policy:checkpoint \
  --policy.config <pi-config-name> \
  --policy.dir <trained-checkpoint-step>
```

This changes startup and throughput, not the loaded checkpoint.

## Evaluate

In a second terminal:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python -m ambench_learn.policies.pi.eval \
  --host localhost \
  --port 8000 \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --prompt "press the button" \
  --num-rollouts 10 \
  --num-envs 1 \
  --episode-length-s 20 \
  --n-action-steps 8 \
  --policy-target-hz 20 \
  --policy-id <pi-config-name>-<trained-checkpoint-step> \
  --save-video \
  --headless \
  --device cuda:0
```

A connection refusal means the server is not listening, the address is wrong, or remote port forwarding is missing. Use the matching BaseJoint environment and config together. Shared options are described in [Configure Evaluation](../evaluation/configure.md), and output follows the [common artifact contract](../reference/evaluation.md).
