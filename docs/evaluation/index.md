# Evaluate a Supported Policy

Use this workflow to run a trained ACT, Diffusion Policy, or served OpenPI policy against a registered AM-Bench environment. The evaluator writes the same result and tracking artifacts for all three families.

## 1. Match the policy to the environment

Before launching Isaac Sim, confirm that the checkpoint or served policy agrees with:

- the exact [environment ID](../reference/environments.md);
- EE-absolute or BaseJoint-absolute environment action semantics;
- the policy action representation and logical policy rate;
- the state keys and cameras used during training.

A matching action dimension is not enough. Use the canonical dataset's `meta/info.json`, checkpoint metadata, and the selected policy guide.

## 2. Choose the evaluator

| Policy | Evaluation entrypoint | Required policy input | Constraint |
| --- | --- | --- | --- |
| [ACT](../policies/act.md) | `python -m ambench_learn.policies.act.eval` | `--checkpoint <checkpoint-or-pretrained_model>` | Relative checkpoints cannot use temporal ensembling |
| [Diffusion Policy](../policies/diffusion-policy.md) | `python -m ambench_learn.policies.dp.eval` | `--checkpoint <checkpoint.ckpt>` | Requires `--num-envs 1` |
| [PI0 / PI0.5](../policies/openpi.md) | `python -m ambench_learn.policies.pi.eval` | reachable `--host` and `--port`, plus `--prompt` | Start the pinned OpenPI server first |

Run the selected module with `--help` to inspect the current interface without starting Isaac Sim.

## 3. Run a bounded smoke evaluation

Start with one rollout, one environment, an explicit seed, and an explicit output directory. For example, with ACT:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python -m ambench_learn.policies.act.eval \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --checkpoint <act-checkpoint> \
  --num-rollouts 1 \
  --num-envs 1 \
  --seed 0 \
  --episode-length-s 20 \
  --output-dir <evaluation-root>/act-smoke \
  --save-video \
  --headless \
  --device cuda:0
```

Use the corresponding policy guide for its complete training and evaluation command. Use [Configure Evaluation](configure.md) for shared controls.

## 4. Confirm completion

The command should shut down cleanly and write `results.txt`, `eval_summary.json`, `tracking/tracking.jsonl`, and `tracking/analysis.json`. When video is enabled, inspect every requested camera under `videos/`.

In `eval_summary.json`, require `status: "completed"` and confirm that `summary.num_rollouts` equals the requested count. A generated file from a stopped, interrupted, or failed run is not evidence of completion. See [Evaluation Outputs](../reference/evaluation.md) for the complete contract.
