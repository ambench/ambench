# Evaluation Outputs

ACT, Diffusion Policy, and OpenPI evaluators share one run lifecycle and artifact contract.

## Default location

```text
outputs/eval/<policy>/<task>/<timestamp>/
```

Use `--output-dir <path>` when an exact, inspectable location is required. The evaluator creates that directory before rollouts begin and finalizes partial results on interruption or failure where possible.

## Artifact layout

```text
<run-dir>/
  results.txt
  eval_summary.json
  tracking/
    tracking.jsonl
    analysis.json
  videos/                 # only with --save-video
```

| Artifact | Purpose |
| --- | --- |
| `results.txt` | Human-readable status, configuration, aggregate metrics, and per-rollout outcomes |
| `eval_summary.json` | Machine-readable status, metadata, summary, tracking pointers, and rollout records |
| `tracking/tracking.jsonl` | Step-level controller, target, state, allocation, and limit telemetry |
| `tracking/analysis.json` | Aggregated tracking and limit-use metrics |
| `videos/` | MP4 rollouts from every requested camera |

## Shared metrics

The summary reports completed rollouts, successes, success rate, available subtask completion, elapsed evaluation time, and throughput. Per-rollout entries include the termination reason and executed step count. For each named criterion in `info["success_criteria"]`, the evaluator records whether it became true at least once; subtask completion is the fraction of available criteria completed, or `N/A` when a task exposes none. Tracking summaries can include EE and base errors, tilt use, saturation, and controller-specific telemetry when the selected profile exposes them.

## Status and comparability

`status` is one of `completed`, `stopped`, `interrupted`, or `failed`. A valid JSON file does not imply a complete benchmark run; check status and completed-rollout count.

Compare success rates only when task ID, checkpoint semantics, rollout count, seed policy, episode length, disturbances, cameras, and evaluator settings match. Tracking metrics should be compared only when the same fields are available from both controller profiles.

## Common CLI

All learned-policy evaluators accept `--task`, `--num-envs`, `--num-rollouts`, `--episode-length-s`, `--seed`, `--disturbance`, `--save-video`, `--video-camera-names`, `--output-dir`, and `--progress-every`, plus Isaac Lab launcher options such as `--headless` and `--device`.
