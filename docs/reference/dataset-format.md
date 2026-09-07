# Dataset Format

The canonical AM-Bench data boundary is a finalized LeRobot 0.4.4 dataset produced by the repository recorder. Policy-specific zarr or OpenPI layouts are derived artifacts, not source datasets.

## Session layout

```text
<session-root>/
  env_cfg.yaml
  lerobot/
    meta/info.json
    meta/episodes/
    data/
    images/ or videos/
  videos/                 # optional recorder videos
```

`env_cfg.yaml` records the task configuration used for collection. The `lerobot/` directory is the canonical dataset root accepted by the validator and training adapters.

## Required frame features

| Feature | Type | Meaning |
| --- | --- | --- |
| `observation.state` | float32 vector | Ordered concatenation of the recorder's `--state_keys` |
| `action` | float32 vector | Environment action at the recorded step |
| `task` | string | Language instruction stored for the episode/frame |
| `observation.images.<camera>` | RGB image | One feature for each recorded camera |

The default state keys are `ee_pos ee_quat gripper_width`. BaseJoint datasets normally use `base_pos base_quat arm_joint_pos gripper_width`.

## AM-Bench metadata

`meta/info.json` includes the standard LeRobot counts, FPS, robot type, and feature declarations. Its `ambench` block records:

- `action_semantics`: `ee_absolute` or `base_joint_absolute`;
- `state_keys`: the ordered keys concatenated into `observation.state`.

Datasets recorded before the package was renamed store this block under
`am_isaac` instead. Readers accept either key, so those datasets remain usable;
newly recorded datasets always use `ambench`.

Canonical relative-policy training uses absolute-action sources. EE absolute actions contain 8 values; the current four-joint BaseJoint platforms contain 12. Quaternions are WXYZ.

## Finalization and validation

Successful episodes receive finalized Parquet metadata under `meta/episodes/`. An interrupted collection that never finalized is not a valid training source. Run [Validate Datasets](../workflows/validate-data.md) before conversion or training.

Keep the canonical source immutable. DP conversion writes UMI zarr, and OpenPI export writes a pinned reader-compatible LeRobot layout plus norm-stat inputs. Regenerate either derived format from the canonical dataset when adapters change.
