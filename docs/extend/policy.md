# Integrate a Policy

Learned-policy integrations live under `source/ambench_learn/ambench_learn/policies/<family>/`. Start with a finalized dataset that passes [canonical validation](../workflows/validate-data.md). The result is a training path with checkpoint metadata and an evaluator that follows the shared CLI and artifact contract.

Use `source/ambench_learn/ambench_learn/policies/act/` for an in-process canonical-LeRobot pattern, `policies/dp/` for a derived-format integration, or `policies/pi/` for a separate policy-server pattern. Shared evaluation behavior belongs in `source/ambench_learn/ambench_learn/eval/`, not in a second policy-specific implementation.

## 1. Declare the data contract

Start from a validated canonical LeRobot dataset. Document the accepted `action_semantics`, ordered state keys, camera features, logical policy rate, history, prediction horizon, and model representation. If conversion is required, write a derived dataset without modifying the source.

Keep transforms under `ambench_learn.data` when they are shared. Policy-specific model preprocessing stays with the policy family.

## 2. Implement training and checkpoint metadata

The training entry point should save enough metadata to reject mismatched evaluation: policy rate, state and action representation, cameras, normalization, task or language conditioning, and source semantics. Paths should be command arguments rather than local constants.

## 3. Use the common evaluator

Add the shared CLI with `add_common_eval_args(...)`, validate arguments before Isaac Sim starts, and use `resolve_eval_output_dir`, `EvalRun`, and `run_evaluation_batches`. Convert model predictions back to the environment action representation before `env.step(...)`.

Match the common progress, video, and artifact behavior described in [Evaluation Outputs](../reference/evaluation.md). Add policy-only arguments only for genuine runtime differences, such as a remote server address.

## 4. Validate the complete path

Run these gates in order:

1. static dataset validation at the intended policy rate;
2. one or a few training steps that save a loadable checkpoint;
3. checkpoint reload in a clean process;
4. one rollout with one environment and explicit output directory;
5. inspection of `results.txt`, `eval_summary.json`, tracking analysis, and requested videos;
6. a larger evaluation only after the smoke path passes.

Training loss alone does not validate action conversion or simulator evaluation.
