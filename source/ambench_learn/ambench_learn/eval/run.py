# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical eval-run lifecycle for ACT, DP, and OpenPI."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ambench.evaluation.tracking.log import TrackingLog, json_safe
from ambench.evaluation.tracking.metrics import (
    finalize_tracking_log,
    format_tracking_console_lines,
    format_tracking_progress,
    format_tracking_summary_lines,
    summarize_records,
)


def format_metric(value: Any, precision: int = 6) -> str:
    """Format a scalar metric for user-facing text."""
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(scalar):
        return "N/A"
    return f"{scalar:.{precision}f}"


def summarize_eval_rollouts(
    rollout_records: Sequence[Mapping[str, Any]],
    *,
    eval_wall_time_s: float | None = None,
    subtask_completion: float | None = None,
) -> dict[str, Any]:
    """Summarize policy eval rollouts independent of policy implementation."""
    successes = [bool(record.get("success", False)) for record in rollout_records]
    num_rollouts = len(rollout_records)
    num_successes = int(sum(successes))
    if subtask_completion is None:
        subtask_values = [
            float(record["subtask_completion"])
            for record in rollout_records
            if record.get("subtask_completion") is not None
        ]
        subtask_completion = float(np.mean(subtask_values)) if subtask_values else None
    seconds_per_rollout = None
    rollouts_per_minute = None
    if eval_wall_time_s is not None and num_rollouts > 0:
        seconds_per_rollout = float(eval_wall_time_s) / num_rollouts
        rollouts_per_minute = 60.0 / seconds_per_rollout if seconds_per_rollout > 0 else 0.0
    return {
        "num_rollouts": num_rollouts,
        "num_successes": num_successes,
        "success_rate": float(num_successes / num_rollouts) if num_rollouts else 0.0,
        "success_rate_percent": float(100.0 * num_successes / num_rollouts) if num_rollouts else 0.0,
        "subtask_completion": subtask_completion,
        "subtask_completion_percent": None if subtask_completion is None else float(subtask_completion * 100.0),
        "eval_wall_time_s": eval_wall_time_s,
        "seconds_per_rollout": seconds_per_rollout,
        "rollouts_per_minute": rollouts_per_minute,
        "successes": successes,
    }


def format_eval_text_report(
    *,
    metadata: Mapping[str, Any],
    rollout_records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    tracking_summary: Mapping[str, Any] | None = None,
    tracking_summary_path: str | Path | None = None,
    status: str = "completed",
    error: str | None = None,
) -> str:
    """Build a text report for an eval run."""
    lines = [f"Status: {status}"]
    if error is not None:
        lines.append(f"Error: {error}")
    lines.extend(f"{key}: {value}" for key, value in metadata.items())
    lines.extend([
        f"Completed rollouts: {summary['num_rollouts']}",
        f"Success rate: {summary['success_rate_percent']:.2f}%",
        f"Subtask completion: {_format_optional_percent(summary.get('subtask_completion_percent'))}",
    ])
    if summary.get("eval_wall_time_s") is not None:
        lines.extend([
            f"Eval wall time (s): {summary['eval_wall_time_s']:.3f}",
            f"Seconds per rollout: {format_metric(summary.get('seconds_per_rollout'), 3)}",
            f"Rollouts per minute: {format_metric(summary.get('rollouts_per_minute'), 3)}",
        ])
    lines.extend(["", "Per-rollout results:"])
    for index, record in enumerate(rollout_records):
        rollout_idx = record.get("rollout_idx", index)
        completion = record.get("subtask_completion")
        completion_text = "N/A" if completion is None else f"{float(completion) * 100.0:.2f}%"
        env_text = "" if "batch_env_index" not in record else f" env={record['batch_env_index']}"
        lines.append(
            f"  Rollout {rollout_idx}: success={bool(record.get('success', False))}"
            f"{env_text} steps={record.get('executed_steps')} "
            f"termination_reason={record.get('termination_reason', 'unknown')} "
            f"subtask_completion={completion_text}"
        )
    if tracking_summary is not None:
        lines.extend(format_tracking_summary_lines(tracking_summary, tracking_summary_path))
    lines.append("")
    return "\n".join(lines)


def write_eval_text_report(path: str | Path, **kwargs: Any) -> Path:
    """Write a text report for an eval run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_eval_text_report(**kwargs), encoding="utf-8")
    return path


def print_eval_summary(
    *,
    summary: Mapping[str, Any],
    result_file: str | Path | None = None,
    tracking_log_path: str | Path | None = None,
    tracking_summary: Mapping[str, Any] | None = None,
    video_folder: str | Path | None = None,
    title: str = "Evaluation Results",
    status: str = "completed",
    error: str | None = None,
) -> None:
    """Print a compact user-facing eval summary."""
    lines = [
        "",
        "=" * 60,
        title,
        "=" * 60,
        f"Status: {status}",
        f"Success rate: {summary['success_rate_percent']:.2f}% ({summary['num_successes']}/{summary['num_rollouts']})",
        f"Subtask completion: {_format_optional_percent(summary.get('subtask_completion_percent'))}",
    ]
    if summary.get("eval_wall_time_s") is not None:
        lines.append(
            f"Wall time: {summary['eval_wall_time_s']:.2f}s "
            f"({format_metric(summary.get('seconds_per_rollout'), 2)}s/rollout, "
            f"{format_metric(summary.get('rollouts_per_minute'), 2)} rollouts/min)"
        )
    if tracking_summary is not None:
        lines.extend(format_tracking_console_lines(tracking_summary))
    if tracking_log_path is not None:
        lines.append(f"Saved tracking JSONL to {tracking_log_path}")
    if result_file is not None:
        lines.append(f"Saved results to {result_file}")
    if video_folder is not None:
        lines.append(f"VIDEO_PATH={Path(video_folder).resolve()}")
    if error is not None:
        lines.append(f"Error: {error}")
    lines.extend(["=" * 60, ""])
    print("\n".join(lines), flush=True)


def _format_optional_percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


class EvalRun:
    """Own eval artifacts, rollout records, and optional tracking diagnostics."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        policy: str,
        task: str,
        policy_source: str | Path | None = None,
        requested_rollouts: int | None = None,
        num_envs: int | None = None,
        action_semantics: str | None = None,
        action_representation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        tracking_enabled: bool = True,
        tracking_dir: str | Path | None = None,
        result_filename: str = "results.txt",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.task = task
        self.action_semantics = action_semantics
        self.metadata = {
            key: value
            for key, value in {
                "Task": task,
                "Policy": policy,
                "Policy source": policy_source,
                "Requested rollouts": requested_rollouts,
                "Num envs": num_envs,
                "Action semantics": action_semantics,
                "Action representation": action_representation,
                **dict(metadata or {}),
            }.items()
            if value is not None
        }
        self.result_path = self.output_dir / result_filename
        self.eval_summary_path = self.output_dir / "eval_summary.json"
        self.tracking_dir = Path(tracking_dir) if tracking_dir is not None else self.output_dir / "tracking"
        self.tracking_log = None
        if tracking_enabled:
            tracking_metadata = {
                **self.metadata,
                "task": task,
                "policy": policy,
            }
            self.tracking_log = TrackingLog(self.tracking_dir / "tracking.jsonl", metadata=tracking_metadata)
        self.rollout_records: list[dict[str, Any]] = []
        self.requested_rollouts = requested_rollouts
        self._tracking_collectors: dict[tuple[int, int], Any] = {}
        self._tracking_records: dict[int, list[dict[str, Any]]] = {}

    @property
    def tracking_enabled(self) -> bool:
        return self.tracking_log is not None

    @property
    def tracking_log_path(self) -> Path | None:
        return None if self.tracking_log is None else self.tracking_log.path

    def print_progress(
        self,
        *,
        step: int,
        reward: float | None = None,
        rollout_batch: str | None = None,
        active: tuple[int, int] | None = None,
        tracking_record: Mapping[str, Any] | None = None,
        tracking_env_index: int | None = None,
        progress_every: int = 1,
        prefix: str | None = None,
    ) -> None:
        """Print one compact rollout progress line."""
        if step != 1 and step % progress_every != 0:
            return
        parts = []
        if prefix is not None:
            parts.append(prefix)
        if rollout_batch is not None:
            total_text = "" if self.requested_rollouts is None else f"/{self.requested_rollouts}"
            parts.append(f"rollouts={rollout_batch}{total_text}")
        parts.append(f"step={int(step)}")
        if active is not None:
            parts.append(f"active={active[0]}/{active[1]}")
        if reward is not None:
            parts.append(f"reward_mean={float(reward):.2f}")
        if tracking_env_index is not None:
            parts.append(f"tracking_env={tracking_env_index}")
        tracking_text = format_tracking_progress(tracking_record, action_semantics=self.action_semantics)
        if tracking_text:
            parts.append(tracking_text)
        print(" ".join(parts), flush=True)

    @staticmethod
    def print_rollout_finished(record: Mapping[str, Any], *, total: int | None = None) -> None:
        """Print one rollout completion line."""
        rollout_idx = int(record.get("rollout_idx", 0))
        total_text = "" if total is None else f"/{total}"
        env_text = "" if "batch_env_index" not in record else f" batch_env_index={record['batch_env_index']}"
        completion = record.get("subtask_completion")
        completion_text = "N/A" if completion is None else f"{float(completion) * 100.0:.1f}%"
        print(
            f"rollout={rollout_idx + 1}{total_text} finished "
            f"success={bool(record.get('success', False))} "
            f"steps={record.get('executed_steps')} "
            f"reason={record.get('termination_reason', 'unknown')} "
            f"subtask_completion={completion_text}{env_text}",
            flush=True,
        )

    def print_startup(self) -> None:
        """Print the same startup summary for every learned-policy evaluator."""

        lines = ["", "=" * 60, "Evaluation Setup", "=" * 60]
        lines.extend(f"{key}: {value}" for key, value in self.metadata.items())
        lines.extend(["=" * 60, ""])
        print("\n".join(lines), flush=True)

    def start_rollout(
        self,
        rollout_idx: int,
        *,
        batch_env_index: int | None = None,
        **metadata: Any,
    ) -> None:
        """Start a rollout in the eval artifact stream."""
        for collector in self._tracking_collectors.values():
            collector.reset()
        if self.tracking_log is not None:
            self.tracking_log.start_rollout(
                rollout_idx=rollout_idx,
                batch_env_index=batch_env_index,
                **metadata,
            )
        self._tracking_records.setdefault(rollout_idx, [])

    def record_timestep(
        self,
        *,
        env: Any,
        raw_obs: Mapping[str, Any],
        rollout_idx: int,
        timestep: int,
        env_index: int = 0,
        batch_env_index: int | None = None,
    ) -> dict[str, Any] | None:
        """Collect and write one timestep tracking record."""
        if self.tracking_log is None:
            return None
        collector = self._collector(env, env_index)
        record = collector.collect(raw_obs, timestep=timestep)
        self._tracking_records.setdefault(rollout_idx, []).append(record)
        self.tracking_log.write_timestep(
            rollout_idx=rollout_idx,
            batch_env_index=batch_env_index,
            record=record,
        )
        return record

    def finish_rollout(
        self,
        rollout_idx: int,
        *,
        success: bool,
        executed_steps: int,
        subtask_completion: float | None,
        termination_reason: str = "unknown",
        batch_env_index: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finalize one rollout and return the canonical rollout record."""
        tracking_records = self._tracking_records.get(rollout_idx, [])
        tracking_summary = summarize_records(tracking_records) if self.tracking_log is not None else None
        if self.tracking_log is not None:
            self.tracking_log.finish_rollout(
                rollout_idx=rollout_idx,
                batch_env_index=batch_env_index,
                success=success,
                executed_steps=executed_steps,
                subtask_completion=subtask_completion,
                summary=tracking_summary,
                termination_reason=termination_reason,
            )
        record = dict(extra or {})
        record.update({
            "rollout_idx": rollout_idx,
            "success": bool(success),
            "executed_steps": int(executed_steps),
            "subtask_completion": subtask_completion,
            "termination_reason": termination_reason,
        })
        if batch_env_index is not None:
            record["batch_env_index"] = int(batch_env_index)
        if tracking_summary is not None and self.tracking_log is not None:
            record["tracking_log"] = str(self.tracking_log.path)
            record["tracking"] = tracking_summary
        self.rollout_records.append(record)
        return record

    def finish(
        self,
        *,
        eval_wall_time_s: float | None = None,
        video_folder: str | Path | None = None,
        print_summary: bool = True,
        status: str = "completed",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Finalize eval artifacts and return the eval summary payload."""
        summary = summarize_eval_rollouts(self.rollout_records, eval_wall_time_s=eval_wall_time_s)
        tracking_summary = None
        tracking_summary_path = None
        if self.tracking_log is not None:
            tracking_summary, tracking_summary_path = finalize_tracking_log(
                self.tracking_log,
                self.tracking_dir,
                extra_summary={
                    "subtask_completion": summary["subtask_completion"],
                    "status": status,
                    "error": error,
                },
            )
        write_eval_text_report(
            self.result_path,
            metadata=self.metadata,
            rollout_records=self.rollout_records,
            summary=summary,
            tracking_summary=tracking_summary,
            tracking_summary_path=tracking_summary_path,
            status=status,
            error=error,
        )
        payload = json_safe({
            "status": status,
            "error": error,
            "metadata": self.metadata,
            "summary": summary,
            "tracking": {
                "tracking_jsonl": None if self.tracking_log_path is None else str(self.tracking_log_path),
                "analysis_json": None if tracking_summary_path is None else str(tracking_summary_path),
                "summary": tracking_summary,
            },
            "rollouts": self.rollout_records,
        })
        self.eval_summary_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if print_summary:
            print_eval_summary(
                summary=summary,
                result_file=self.result_path,
                tracking_log_path=self.tracking_log_path,
                tracking_summary=tracking_summary,
                video_folder=video_folder,
                status=status,
                error=error,
                title="Evaluation Results" if status == "completed" else f"Evaluation {status.capitalize()}",
            )
        return payload

    def _collector(self, env: Any, env_index: int) -> Any:
        from ambench.evaluation.tracking.collect import TrackingCollector

        key = (id(env.unwrapped), int(env_index))
        if key not in self._tracking_collectors:
            self._tracking_collectors[key] = TrackingCollector(env=env, env_idx=int(env_index))
        return self._tracking_collectors[key]
