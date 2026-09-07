# Validate Datasets

Every finalized recording must pass static validation before conversion or training.

```bash
source ../IsaacLab/env_isaaclab/bin/activate
python scripts/data/validate_lerobotdataset.py \
  --dataset_root <session-root> \
  --repo_id am_bench/press_button_ee_absolute \
  --target_hz 20
```

`--dataset_root` accepts either the recorder session or its `lerobot` child. Use the repository ID stored during recording.

## Passing evidence

A successful command prints:

```text
LeRobot dataset validation passed
...
Benchmark resampling validation passed
```

The validator checks:

- positive episode, frame, and FPS metadata;
- finalized episode tables and consistent episode lengths;
- required state, action, and image features;
- declared feature shapes against loaded samples;
- readable first and last frames;
- AM-Bench action semantics;
- integer logical resampling at the requested policy rate.

## What validation does not prove

!!! note "Validation boundary"

    Static validation does not establish demonstration quality, task success correctness, camera usefulness, or checkpoint compatibility. Inspect representative videos and confirm behavior before training.

    AM-Bench does not expose a public exact-state dataset replay entrypoint. Exact simulator replay would require reset and scene state beyond the canonical policy dataset. The maintained gate is static validation plus visual inspection.

## Policy-specific derived formats

ACT reads the validated logical dataset view directly. DP converts it with `scripts/data/dp/lerobot_to_zarr.py` and validates that output with `scripts/data/dp/validate_zarr.py`. OpenPI uses `scripts/data/export_lerobot_to_openpi.py` because the pinned OpenPI reader expects an older local LeRobot layout. These conversions must never overwrite the canonical source.
