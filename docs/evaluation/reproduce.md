# Reproduce Research Workflows

Reproduction in AM-Bench means preserving the full data, policy, environment, and evaluation configuration—not copying a reported result from the paper. Use the project website for findings and this page for the reusable execution record.

## 1. Record the source revision

Start from a clean checkout and record the commit and submodule revisions:

```bash
git rev-parse HEAD
git submodule status
```

Keep the Python, Isaac Sim, Isaac Lab, CUDA, and GPU details with the run notes. Follow [Installation](../getting-started/installation.md) and verify the environment registry before training.

## 2. Preserve the canonical data boundary

Record or obtain a finalized canonical LeRobot dataset, then run [Validate Datasets](../workflows/validate-data.md) at the intended policy rate. Preserve:

- the source dataset and its `meta/info.json`;
- the collection `env_cfg.yaml`;
- the repository ID, action semantics, ordered state keys, cameras, task prompt, FPS, and episode count;
- representative rollout videos used for visual inspection.

Keep the canonical source immutable. Generate DP zarr or OpenPI layouts as derived artifacts using the commands in the relevant policy guide.

## 3. Train one supported policy branch

Use the maintained [ACT](../policies/act.md), [Diffusion Policy](../policies/diffusion-policy.md), or [OpenPI](../policies/openpi.md) guide. First run its bounded training smoke, reload the saved checkpoint in a clean process, and only then start the full training configuration.

Archive the exact training command, policy config, random seed, dataset identity, normalization data, checkpoint identity, and training logs outside the Git repository.

## 4. Evaluate with an explicit run record

Use [Evaluate a Supported Policy](index.md) and set the task ID, seed, rollout count, episode length, disturbance mode, cameras, output directory, device, and policy-specific execution settings explicitly. Confirm semantic compatibility before launch.

For every run, retain:

```text
source and submodule revisions
environment and dependency versions
canonical dataset metadata and derived-data recipe
training command, configuration, seed, and checkpoint identity
evaluation command and output directory
results.txt and eval_summary.json
tracking/tracking.jsonl and tracking/analysis.json
videos/ when enabled
```

Require `status: "completed"` and the requested rollout count before using a run in a comparison.

## Artifact availability

!!! info "Artifact availability"

    Datasets, checkpoints, videos, and generated evaluation outputs are not stored in this Git repository. Use artifacts linked by the [project website](https://ambench.github.io/) when available; otherwise the repository supports regenerating them from your own validated demonstrations. Do not substitute an undocumented artifact or configuration and describe the run as an exact reproduction.

The public docs intentionally do not repeat result tables. Compare a completed, configuration-matched run with the paper or website record.
