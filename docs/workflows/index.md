# Use AM-Bench

Choose a workflow by the result you need. Each page states its prerequisites, command, expected output, and validation boundary.

## Run and inspect an environment

[Run an Environment](run-environment.md) without a learned policy to check scene startup, observations, actions, cameras, and simulator performance.

## Collect and validate data

[Collect Demonstrations](collect-demos.md) records scripted or teleoperated episodes in the canonical LeRobot format. [Validate Datasets](validate-data.md) checks finalized metadata, episode boundaries, feature shapes, samples, action semantics, and logical resampling before conversion or training.

## Evaluate a supported policy

[Evaluate a Supported Policy](../evaluation/index.md) covers the shared workflow and routes to ACT, Diffusion Policy, and OpenPI. Use [Reproduce Research Workflows](../evaluation/reproduce.md) when preserving a complete experiment record.

## Extend the system

[Extend an Existing Environment](../extend/index.md) covers task, robot, controller, and policy integrations. Start from the nearest maintained implementation and validate one layer at a time.
