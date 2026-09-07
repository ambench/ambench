# Supported Policies

AM-Bench provides task-aware scripted policies and three maintained learned-policy integrations.

| Policy | Best use | Training data | Evaluation process |
| --- | --- | --- | --- |
| [Scripted](scripted-policies.md) | task validation and demonstration collection | none | in simulator |
| [ACT](act.md) | compact task-specific imitation baseline | canonical LeRobot | in simulator process |
| [Diffusion Policy](diffusion-policy.md) | task-specific image-conditioned diffusion baseline | derived UMI zarr | in simulator process |
| [PI0 / PI0.5](openpi.md) | pretrained VLA adaptation and language-conditioned evaluation | pinned OpenPI export | separate policy server and simulator client |

## Shared contract

All learned policies begin from a validated canonical dataset and must match the environment's action semantics. Evaluators share rollout controls, progress output, artifact naming, tracking, and video behavior. Start with [Evaluate a Supported Policy](../evaluation/index.md), then use the family guide for training and policy-only arguments.

## Start small

Before a full experiment:

1. validate the source dataset at the intended policy rate;
2. run a bounded training smoke that saves a loadable checkpoint;
3. evaluate one rollout with one environment and video enabled;
4. inspect `results.txt`, `eval_summary.json`, tracking analysis, and all requested camera videos;
5. scale only after the complete pipeline passes.

Training success does not prove evaluation compatibility, and a successful one-rollout smoke does not measure policy quality.
