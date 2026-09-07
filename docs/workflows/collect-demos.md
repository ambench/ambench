# Collect Demonstrations

The scripted and teleoperation recorders write the same canonical LeRobot dataset format. Start with one environment and one episode, inspect the result, then scale collection.

## Prepare the environment

```bash
source ../IsaacLab/env_isaaclab/bin/activate
uv pip install -e source/ambench
uv pip install -e source/ambench_learn
uv pip install --no-deps "lerobot==0.4.4"
nvidia-smi
```

Choose a dataset root on a filesystem with enough capacity for images and optional videos. Do not write datasets into the Git repository history.

## Record an EE-absolute dataset

This is the canonical source for the maintained EE-relative ACT, DP, and OpenPI paths:

```bash
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-EE-Abs-PID-Direct-v0 \
  --dataset_root <dataset-root> \
  --repo_id am_bench/press_button_ee_absolute \
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

The recorder saves successful episodes by default. A timeout is discarded unless `--save_failed_episodes` is supplied for debugging. Saved failed episodes are written alongside successful ones but are not counted toward `--num_demos`, which always means an exact number of successful demonstrations.

## Record a BaseJoint-absolute dataset

Use a registered BaseJoint environment and include base pose, absolute arm joints, and gripper width in the state:

```bash
python scripts/data/record_demos_scripted.py \
  --task PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0 \
  --dataset_root <dataset-root> \
  --repo_id am_bench/press_button_base_joint_absolute \
  --state_keys base_pos base_quat arm_joint_pos gripper_width \
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

The scripted policy produces task-space targets; the recorder's BaseJoint adapter uses the environment's IK configuration to produce the absolute BaseJoint action.

## Record with teleoperation

Use teleoperation when no scripted policy is available:

```bash
python scripts/data/record_demos_teleop.py --help
```

Teleoperation normally requires a visible Isaac Sim session and a supported input device. Use one environment, omit `--headless`, and verify the device mapping before collecting a full session.

## Inspect before scaling

After a clean shutdown, retain the printed session path and check:

- `<session-root>/lerobot/meta/info.json` exists;
- the reported `action_semantics` and ordered `state_keys` match the intended policy interface;
- the number of finalized episodes matches the target;
- camera images and optional videos show the intended viewpoint;
- successful demonstrations complete the task rather than exploiting a success-condition bug.

Then run [Validate Datasets](validate-data.md). Only after that passes should you increase `--num_envs` or `--num_demos`.
