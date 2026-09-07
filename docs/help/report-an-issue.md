# Report an Issue

Report reproducible AM-Bench software and documentation problems through [GitHub Issues](https://github.com/ambench/ambench/issues). For research findings, authors, affiliations, or other project information, use the contact route on the [project website](https://ambench.github.io/).

## Before filing

Check [Troubleshooting](troubleshooting.md) and reproduce the problem with the smallest relevant command. For simulator problems, use one environment and the lightest applicable profile. For policy problems, separate checkpoint loading, environment startup, and the first inference step.

Do not attach datasets, checkpoints, credentials, private URLs, or machine-specific secrets. Replace personal storage paths while preserving the relevant directory structure.

## Include this information

### Problem and expected behavior

Give the issue a specific title. Describe what happened, what you expected, and the earliest failing layer: installation, registration, startup, scene/task behavior, data, policy, evaluation, or documentation.

### Exact reproduction

Include:

- the complete command, including the environment ID and relevant arguments;
- the smallest configuration override or code change required to reproduce it;
- whether the run is headless, uses cameras, requires a GUI, or connects to a policy server;
- whether the problem occurs every time and the seed when relevant.

Use a fenced code block for commands and a separate block for output.

### Environment and revision

Record:

```bash
source ../IsaacLab/env_isaaclab/bin/activate
git rev-parse HEAD
git submodule status
python --version
nvidia-smi
```

Also name the operating system, Isaac Sim version, Isaac Lab revision, installation method, GPU model, driver, CUDA device argument, and Python environment used. For dependency-specific failures, include the relevant package versions.

### Error and artifacts

Include the shortest log excerpt containing the first error and its traceback. For registration or scene problems, provide the exact task ID and config class. For data problems, provide the validator output and sanitized `meta/info.json` fields. For evaluation problems, include the evaluator command, policy/checkpoint identifier, `eval_summary.json` status and error, and whether partial tracking or video files were written.

Images or short videos can help with visible scene, camera, contact, or controller problems, but describe the behavior in text so the report remains searchable.

## Documentation issues

For a broken or inaccurate documentation page, include its URL or repository path, the heading, the command or identifier involved, and the current branch or commit. State the correction you verified against the repository when known.
