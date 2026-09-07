# Scripted Policies

Scripted policies are task-aware state machines used to validate task logic and collect successful demonstrations. They are not benchmark competitors against learned policies.

Implementations live in `source/ambench/ambench/policies/scripted/`. Each maintained task registers or exposes the scripted policy used by the generic recorder.

## Run through the recorder

```bash
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --dataset_root <dataset-root> \
  --repo_id am_bench/press_button_smoke \
  --state_keys ee_pos ee_quat gripper_width \
  --task_prompt "press the button" \
  --step_hz 120 \
  --num_envs 1 \
  --num_demos 1 \
  --env_length_s 20 \
  --camera_names ee_camera \
  --video \
  --headless \
  --device cuda:0
```

A successful episode is evidence that the selected reset, scene, controller, success criteria, and scripted behavior can cooperate for that rollout. It is not a robustness measurement. Inspect the video and repeat across seeds before collecting a large dataset.

For a BaseJoint environment, the recorder uses the robot profile's IK configuration to adapt scripted EE targets into absolute base and joint actions. See [Collect Demonstrations](../workflows/collect-demos.md) for the full recording contract.
