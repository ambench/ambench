# Diffusion Policy

The maintained Diffusion Policy path converts canonical absolute-action LeRobot data to UMI zarr, validates the conversion, trains through Hydra, and evaluates with receding-horizon execution.

## Install

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e "source/ambench_learn[dp]"
uv pip install -e source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface
uv pip install --no-deps "lerobot==0.4.4"
```

## Convert and validate

```bash
python scripts/data/validate_lerobotdataset.py \
  --dataset_root <session-root> \
  --repo_id am_bench/press_button_ee_absolute \
  --target_hz 20

python scripts/data/dp/lerobot_to_zarr.py \
  --input_path <session-root> \
  --output_path <dataset-root>/press_button_ee.zarr.zip \
  --omit_base_image

python scripts/data/dp/validate_zarr.py \
  <dataset-root>/press_button_ee.zarr.zip \
  --image_size 224
```

Omit `--omit_base_image` only when the canonical dataset contains `observation.images.base_camera`. The converter rejects datasets recorded with the removed EE-delta semantics. For BaseJoint data, record the matching absolute state/action schema and select the BaseJoint task configuration during training.

## Train

Choose `umi_drone_ee_pos` for EE or `umi_drone_base_joint` for BaseJoint. Run from the UMI directory so Hydra resolves its config tree:

```bash
cd source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task=umi_drone_ee_pos \
  task.dataset_path=<absolute-path-to-dataset.zarr.zip> \
  task.obs_down_sample_steps=6 \
  task.action_horizon=16 \
  task.pose_repr.obs_pose_repr=relative \
  task.pose_repr.action_pose_repr=relative \
  training.num_epochs=40 \
  training.checkpoint_every=10 \
  training.device=cuda:0 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  exp_name=press_button_dp \
  logging.name=press_button_dp \
  logging.mode=disabled \
  hydra.run.dir=<absolute-output-dir>
cd -
```

For a bounded training smoke, add `training.num_epochs=1 training.max_train_steps=1 training.max_val_steps=1`.

## Evaluate

```bash
python -m ambench_learn.policies.dp.eval \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --checkpoint <absolute-output-dir>/checkpoints/latest.ckpt \
  --num-rollouts 10 \
  --num-envs 1 \
  --episode-length-s 20 \
  --save-video \
  --headless \
  --device cuda:0
```

!!! warning "Limitations"

    DP evaluation currently requires exactly one environment. A BaseJoint checkpoint must be paired with a BaseJoint task.

Shared options are described in [Configure Evaluation](../evaluation/configure.md), and output follows the [common artifact contract](../reference/evaluation.md).
