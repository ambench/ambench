# ACT

ACT trains directly from a validated canonical LeRobot dataset. The maintained model representations are `ee_local_relative` and `base_joint_relative`.

## Install

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e "source/ambench_learn[act]"
uv pip install --no-deps "lerobot==0.4.4"
python -c "from importlib.metadata import version; print(version('lerobot'))"
```

The final command must print `0.4.4`.

## Validate the source dataset

```bash
python scripts/data/validate_lerobotdataset.py \
  --dataset_root <session-root> \
  --repo_id am_bench/press_button_ee_absolute \
  --target_hz 20
```

Use an absolute-action source dataset. ACT constructs relative trajectories at the logical policy rate without rewriting the source.

## Train

```bash
python -m ambench_learn.policies.act.train \
  --dataset.repo_id=am_bench/press_button_ee_absolute \
  --dataset.root=<session-root>/lerobot \
  --dataset.use_imagenet_stats=true \
  --policy.type=act \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=<checkpoint-root> \
  --job_name=press_button_act \
  --batch_size=8 \
  --steps=20000 \
  --num_workers=4 \
  --save_freq=5000 \
  --wandb.enable=false \
  --policy_target_hz=20 \
  --policy_action_representation=ee_local_relative
```

For BaseJoint data, use the corresponding dataset and set `--policy_action_representation=base_joint_relative`. A saved checkpoint contains `pretrained_model` plus the representation metadata required by evaluation.

## Evaluate

```bash
python -m ambench_learn.policies.act.eval \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --checkpoint <checkpoint-root>/checkpoints/020000/pretrained_model \
  --num-rollouts 10 \
  --num-envs 1 \
  --n-action-steps 8 \
  --policy-target-hz 20 \
  --episode-length-s 20 \
  --save-video \
  --headless \
  --device cuda:0
```

Use a BaseJoint environment only with a BaseJoint checkpoint. ACT rejects temporal ensembling for relative representations because predictions anchored to different measured states cannot be averaged safely.

Shared rollout, seed, disturbance, video, and output options are described in [Configure Evaluation](../evaluation/configure.md). Evaluation output follows the [common artifact contract](../reference/evaluation.md).
