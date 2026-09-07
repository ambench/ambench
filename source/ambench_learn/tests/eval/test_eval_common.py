# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from ambench_learn.eval.common import (
    add_common_eval_args,
    build_common_eval_metadata,
    configure_env_cfg,
    resolve_eval_output_dir,
    run_evaluation_batches,
    validate_common_eval_args,
    validate_video_camera_names,
    wrap_video,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    return parser


def test_common_cli_uses_one_canonical_kebab_case_interface() -> None:
    args = _parser().parse_args([
        "--task",
        "Task-v0",
        "--num-envs",
        "2",
        "--num-rollouts",
        "3",
        "--save-video",
        "--episode-length-s",
        "4.5",
    ])

    assert args.num_envs == 2
    assert args.num_rollouts == 3
    assert args.save_video
    assert "--num-envs" in _parser().format_help()
    assert "--num_envs" not in _parser().format_help()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"num_envs": 0}, "--num-envs"),
        ({"num_rollouts": 0}, "--num-rollouts"),
        ({"episode_length_s": 0}, "--episode-length-s"),
        ({"progress_every": 0}, "--progress-every"),
    ],
)
def test_common_cli_validation(updates: dict, message: str) -> None:
    args = _parser().parse_args(["--task", "Task-v0"])
    for name, value in updates.items():
        setattr(args, name, value)

    with pytest.raises(ValueError, match=message):
        validate_common_eval_args(args)


def test_output_dir_override_is_exact(tmp_path) -> None:
    output_dir = resolve_eval_output_dir(policy="ACT", task="Task/v0", output_dir=tmp_path / "chosen")

    assert output_dir == (tmp_path / "chosen").resolve()


def test_common_metadata_records_runtime_interface(tmp_path) -> None:
    args = _parser().parse_args(["--task", "Task-v0", "--save-video"])
    args.device = "cuda:1"

    metadata = build_common_eval_metadata(args, output_dir=tmp_path)

    assert metadata["Device"] == "cuda:1"
    assert metadata["Video cameras"] == ["ee_camera", "scene_camera"]
    assert metadata["Progress interval (steps)"] == 100


def test_video_camera_validation_rejects_missing_requested_sensor() -> None:
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(sensors={"ee_camera": object()})))

    with pytest.raises(ValueError, match="scene_camera"):
        validate_video_camera_names(env, ["ee_camera", "scene_camera"])


def test_environment_configuration_applies_shared_overrides() -> None:
    config = SimpleNamespace(seed=0, recorders={"demo": object()})

    def parse_env_cfg(task: str, *, device: str, num_envs: int):
        assert (task, device, num_envs) == ("Task-v0", "cuda:1", 2)
        return config

    scene_camera = object()
    result = configure_env_cfg(
        parse_env_cfg=parse_env_cfg,
        task="Task-v0",
        device="cuda:1",
        num_envs=2,
        seed=7,
        episode_length_s=3.0,
        disturbance=True,
        scene_camera_cfg=scene_camera,
    )

    assert result.seed == 7
    assert result.episode_length_s == 3.0
    assert result.recorders == {}
    assert result.scene_camera_cfg is scene_camera
    assert result.enable_saturation and result.enable_aerodynamic_effects and result.enable_wind_effect


def test_video_wrapper_uses_canonical_settings(tmp_path) -> None:
    calls = []

    class Recorder:
        def __init__(self, env, **kwargs) -> None:
            self.unwrapped = env
            calls.append(kwargs)

    env = object()
    wrapped, video_dir = wrap_video(
        env,
        enabled=True,
        output_dir=tmp_path,
        camera_names=["ee_camera"],
        name_prefix="eval",
        record_video_cls=Recorder,
    )

    assert wrapped.unwrapped is env
    assert video_dir == tmp_path / "videos"
    assert calls == [{
        "video_folder": str(video_dir),
        "camera_names": ["ee_camera"],
        "name_prefix": "eval",
        "fps": 30,
        "frame_skip": 4,
        "output_format": "mp4",
    }]


def test_rollout_batches_include_partial_final_batch_and_finalize() -> None:
    class App:
        def is_running(self) -> bool:
            return True

        def is_exiting(self) -> bool:
            return False

    class Env:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Run:
        def __init__(self) -> None:
            self.rollout_records: list[dict] = []
            self.finished = False

        def finish(self, **kwargs):
            self.finished = True
            return kwargs

    env = Env()
    eval_run = Run()
    batches = []

    def run_batch(batch) -> None:
        batches.append((batch.start, batch.stop, batch.seed))
        eval_run.rollout_records.extend({"rollout_idx": index} for index in range(batch.start, batch.stop))

    payload = run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=App(),
        requested_rollouts=5,
        num_envs=2,
        run_batch=run_batch,
        base_seed=11,
    )

    assert batches == [(0, 2, 11), (2, 4, 13), (4, 5, 15)]
    assert payload["eval_wall_time_s"] >= 0.0
    assert payload["status"] == "completed"
    assert eval_run.finished and env.closed


def test_rollout_interruption_still_finalizes_and_closes() -> None:
    app = SimpleNamespace(is_running=lambda: True, is_exiting=lambda: False)
    env = SimpleNamespace(closed=False)
    env.close = lambda: setattr(env, "closed", True)
    eval_run = SimpleNamespace(rollout_records=[], finished=False, status=None)

    def finish(**kwargs):
        eval_run.finished = True
        eval_run.status = kwargs["status"]
        return kwargs

    eval_run.finish = finish

    def interrupt(_batch) -> None:
        raise KeyboardInterrupt

    run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=app,
        requested_rollouts=1,
        num_envs=1,
        run_batch=interrupt,
    )
    assert eval_run.finished and env.closed
    assert eval_run.status == "interrupted"


def test_rollout_exception_still_finalizes_and_closes() -> None:
    app = SimpleNamespace(is_running=lambda: True, is_exiting=lambda: False)
    env = SimpleNamespace(closed=False)
    env.close = lambda: setattr(env, "closed", True)
    eval_run = SimpleNamespace(rollout_records=[], finished=False, status=None)

    def finish(**kwargs):
        eval_run.finished = True
        eval_run.status = kwargs["status"]
        return kwargs

    eval_run.finish = finish

    def fail(_batch) -> None:
        raise RuntimeError("policy failure")

    with pytest.raises(RuntimeError, match="policy failure"):
        run_evaluation_batches(
            eval_run=eval_run,
            env=env,
            simulation_app=app,
            requested_rollouts=1,
            num_envs=1,
            run_batch=fail,
        )
    assert eval_run.finished and env.closed
    assert eval_run.status == "failed"


def test_application_stop_finalizes_without_starting_a_batch() -> None:
    app = SimpleNamespace(is_running=lambda: False, is_exiting=lambda: False)
    env = SimpleNamespace(closed=False)
    env.close = lambda: setattr(env, "closed", True)
    eval_run = SimpleNamespace(rollout_records=[], finished=False, status=None)

    def finish(**kwargs):
        eval_run.finished = True
        eval_run.status = kwargs["status"]
        return kwargs

    eval_run.finish = finish
    run_evaluation_batches(
        eval_run=eval_run,
        env=env,
        simulation_app=app,
        requested_rollouts=1,
        num_envs=1,
        run_batch=lambda _batch: pytest.fail("batch should not start"),
    )
    assert eval_run.finished and env.closed
    assert eval_run.status == "stopped"
