# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import ambench_learn.eval.run as run_module
from ambench_learn.eval.run import EvalRun


def test_eval_summary_flushes_console_output(monkeypatch) -> None:
    printed = {}

    def capture_print(*args, **kwargs) -> None:
        printed["args"] = args
        printed["kwargs"] = kwargs

    monkeypatch.setattr(run_module, "print", capture_print, raising=False)
    run_module.print_eval_summary(
        summary={
            "success_rate_percent": 0.0,
            "num_successes": 0,
            "num_rollouts": 1,
            "subtask_completion_percent": None,
        }
    )

    assert printed["kwargs"] == {"flush": True}


def test_eval_run_writes_canonical_artifacts(tmp_path) -> None:
    run = EvalRun(
        output_dir=tmp_path,
        policy="test",
        task="Task-v0",
        requested_rollouts=1,
        num_envs=1,
        action_semantics="ee_absolute",
        action_representation="ee_local_relative",
        tracking_enabled=False,
    )
    run.start_rollout(0)
    run.finish_rollout(
        0,
        success=True,
        executed_steps=3,
        subtask_completion=1.0,
        termination_reason="success",
    )
    payload = run.finish(eval_wall_time_s=0.5, print_summary=False)

    assert run.result_path.name == "results.txt" and run.result_path.is_file()
    assert run.eval_summary_path.name == "eval_summary.json" and run.eval_summary_path.is_file()
    saved = json.loads(run.eval_summary_path.read_text())
    assert saved == payload
    assert saved["summary"]["num_rollouts"] == 1
    assert saved["summary"]["success_rate"] == 1.0
    assert saved["status"] == "completed"
    assert saved["rollouts"][0]["termination_reason"] == "success"


def test_tracking_lifecycle_has_one_start_and_finish_boundary(tmp_path, monkeypatch) -> None:
    events = []

    class FakeTrackingLog:
        def __init__(self, path, metadata) -> None:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch()

        def start_rollout(self, **kwargs) -> None:
            events.append(("start", kwargs))

        def write_timestep(self, **kwargs) -> None:
            events.append(("timestep", kwargs))

        def finish_rollout(self, **kwargs) -> None:
            events.append(("finish", kwargs))

    monkeypatch.setattr(run_module, "TrackingLog", FakeTrackingLog)
    monkeypatch.setattr(run_module, "summarize_records", lambda records: {"samples": len(records)})
    monkeypatch.setattr(
        run_module,
        "finalize_tracking_log",
        lambda log, output_dir, extra_summary: ({"samples": 1}, Path(output_dir) / "analysis.json"),
    )

    run = EvalRun(output_dir=tmp_path, policy="test", task="Task-v0")
    monkeypatch.setattr(run, "_collector", lambda env, env_index: SimpleNamespace(collect=lambda obs, timestep: {}))
    run.start_rollout(0, batch_env_index=0)
    run.record_timestep(env=object(), raw_obs={}, rollout_idx=0, timestep=1, batch_env_index=0)
    run.finish_rollout(0, success=False, executed_steps=1, subtask_completion=0.0, batch_env_index=0)

    assert [event[0] for event in events] == ["start", "timestep", "finish"]
    assert events[-1][1]["termination_reason"] == "unknown"
    assert run.tracking_log_path == tmp_path / "tracking" / "tracking.jsonl"


def test_progress_output_uses_canonical_rollout_and_reward_labels(tmp_path, capsys) -> None:
    run = EvalRun(
        output_dir=tmp_path,
        policy="ACT",
        task="Task-v0",
        requested_rollouts=4,
        tracking_enabled=False,
    )

    run.print_progress(
        rollout_batch="1-2",
        step=1,
        active=(2, 2),
        reward=1.25,
        tracking_env_index=0,
        progress_every=100,
    )
    run.print_progress(step=2, progress_every=100)

    assert capsys.readouterr().out == "rollouts=1-2/4 step=1 active=2/2 reward_mean=1.25 tracking_env=0\n"
