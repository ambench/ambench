# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Summary and report helpers for canonical tracking JSONL logs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .log import TrackingLog, iter_tracking_events, json_safe


def _finite(values: Iterable[float | None]) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def _summary(values: Iterable[float | None]) -> dict[str, float | int]:
    array = _finite(values)
    if array.size == 0:
        return {key: np.nan for key in ("mean", "min", "p50", "p90", "p95", "p99", "max")} | {"count": 0}
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "count": int(array.size),
    }


def _sample_summary(values: Iterable[float | None]) -> dict[str, float | int]:
    array = _finite(values)
    if array.size == 0:
        return {"mean": np.nan, "std": np.nan, "stderr": np.nan, "count": 0}
    std = float(np.std(array, ddof=1)) if array.size >= 2 else np.nan
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "stderr": std / float(np.sqrt(array.size)) if array.size >= 2 else np.nan,
        "count": int(array.size),
    }


def _get(record: Mapping[str, Any], path: str, default: Any = np.nan) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _format_metric(value: Any, precision: int = 6) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(scalar):
        return "N/A"
    return f"{scalar:.{precision}f}"


METRIC_PATHS = {
    "ee_pos_error_norm_m": "error.ee_pos_norm_m",
    "ee_rot_error_rad": "error.ee_rot_rad",
    "base_pos_error_norm_m": "error.base_pos_norm_m",
    "base_rot_error_rad": "error.base_rot_rad",
    "joint_l2_error_rad": "error.joint_l2_rad",
    "joint_abs_max_error_rad": "error.joint_abs_max_rad",
    "joint_mean_abs_error_rad": "error.joint_mean_abs_rad",
    "gripper_width_error_m": "error.gripper_width_m",
    "normalized_ee_error": "error.normalized_ee_error",
    "normalized_base_deviation": "error.normalized_base_deviation",
    "base_tilt_rad": "control.base_tilt_rad",
    "base_cmd_step_pos_norm_m": "control.base_cmd_step_pos_norm_m",
    "base_cmd_step_rot_rad": "control.base_cmd_step_rot_rad",
    "joint_cmd_step_l2_rad": "control.joint_cmd_step_l2_rad",
    "joint_cmd_step_max_abs_rad": "control.joint_cmd_step_max_abs_rad",
    "motor_max_thrust": "control.motor_max_thrust",
}


def summarize_records(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one rollout's timestep records."""
    data_for_stats = records[:-1] if len(records) > 1 else records
    metrics = {key: _summary(_get(record, path) for record in data_for_stats) for key, path in METRIC_PATHS.items()}
    saturation_values = []
    for record in data_for_stats:
        motor_saturated = _get(record, "control.motor_saturated")
        if motor_saturated is not None:
            saturation_values.append(motor_saturated)
    dt = _get(data_for_stats[0], "dt", None) if data_for_stats else None
    return {
        "num_tracking_steps": int(len(records)),
        "execution_time": float(len(data_for_stats) * dt) if dt is not None and np.isfinite(dt) else None,
        "metrics": metrics,
        "avg_tracking_error_pos": metrics["ee_pos_error_norm_m"]["mean"],
        "avg_tracking_error_rot_geodesic": metrics["ee_rot_error_rad"]["mean"],
        "avg_base_tracking_error_pos": metrics["base_pos_error_norm_m"]["mean"],
        "avg_base_tracking_error_rot_geodesic": metrics["base_rot_error_rad"]["mean"],
        "avg_normalized_ee_error": metrics["normalized_ee_error"]["mean"],
        "avg_normalized_base_deviation": metrics["normalized_base_deviation"]["mean"],
        "max_tilt_utilization": metrics["base_tilt_rad"]["max"],
        "saturation_rate": float(np.mean(saturation_values)) if saturation_values else np.nan,
    }


def summarize_tracking_log(path: str | Path) -> dict[str, Any]:
    """Summarize a complete tracking JSONL file."""
    metadata = {}
    records_by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rollout_summaries = {}
    for event in iter_tracking_events(path):
        event_type = event["event"]
        if event_type == "run_metadata":
            metadata = event.get("metadata", {})
        elif event_type == "timestep":
            records_by_rollout[int(event["rollout_idx"])].append(event["record"])
        elif event_type == "rollout_summary":
            rollout_summaries[int(event["rollout_idx"])] = event

    rollouts = []
    for rollout_idx in sorted(records_by_rollout):
        saved = rollout_summaries.get(rollout_idx, {})
        computed = summarize_records(records_by_rollout[rollout_idx])
        rollouts.append({
            "rollout_idx": rollout_idx,
            "batch_env_index": saved.get("batch_env_index"),
            "success": bool(saved.get("success", False)),
            "executed_steps": int(saved.get("executed_steps", len(records_by_rollout[rollout_idx]))),
            "subtask_completion": saved.get("subtask_completion"),
            "summary": computed,
        })

    successes = [rollout["success"] for rollout in rollouts]
    all_records = [record for records in records_by_rollout.values() for record in records]
    all_summary = summarize_records(all_records)
    successful_rollouts = [rollout for rollout in rollouts if rollout["success"]]
    successful_records = [
        record for rollout in successful_rollouts for record in records_by_rollout.get(int(rollout["rollout_idx"]), [])
    ]
    successful_summary = summarize_records(successful_records)
    return {
        "metadata": metadata,
        "num_rollouts": len(rollouts),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "num_successes": int(sum(successes)),
        "all_rollouts": all_summary,
        "successful_rollouts_only": successful_summary,
        "per_rollout_sample": {
            "avg_tracking_error_pos": _sample_summary(
                rollout["summary"]["avg_tracking_error_pos"] for rollout in rollouts
            ),
            "avg_base_tracking_error_pos": _sample_summary(
                rollout["summary"]["avg_base_tracking_error_pos"] for rollout in rollouts
            ),
            "avg_normalized_ee_error": _sample_summary(
                rollout["summary"]["avg_normalized_ee_error"] for rollout in rollouts
            ),
            "avg_normalized_base_deviation": _sample_summary(
                rollout["summary"]["avg_normalized_base_deviation"] for rollout in rollouts
            ),
            "max_tilt_utilization": _sample_summary(rollout["summary"]["max_tilt_utilization"] for rollout in rollouts),
            "saturation_rate": _sample_summary(rollout["summary"]["saturation_rate"] for rollout in rollouts),
        },
        "rollouts": rollouts,
    }


def write_tracking_reports(
    path: str | Path,
    output_dir: str | Path | None = None,
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write derived JSON and text reports for a tracking JSONL file."""
    path = Path(path)
    output_dir = path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_tracking_log(path) if summary is None else dict(summary)
    (output_dir / "analysis.json").write_text(
        json.dumps(json_safe(summary), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"Tracking log: {path}",
        f"Num rollouts: {summary['num_rollouts']}",
        f"Success rate: {summary['success_rate'] * 100.0:.2f}%",
        "",
        "All-rollout metrics:",
        f"  Avg EE position error: {summary['all_rollouts']['avg_tracking_error_pos']:.6f} m",
        f"  Avg base position error: {summary['all_rollouts']['avg_base_tracking_error_pos']:.6f} m",
        f"  Avg normalized EE error: {summary['all_rollouts']['avg_normalized_ee_error']:.6f}",
        f"  Avg normalized base deviation: {summary['all_rollouts']['avg_normalized_base_deviation']:.6f}",
        f"  Max tilt utilization: {summary['all_rollouts']['max_tilt_utilization']:.6f} rad",
        f"  Saturation rate: {summary['all_rollouts']['saturation_rate'] * 100.0:.2f}%",
    ]
    (output_dir / "tracking_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def finalize_tracking_log(
    tracking_log: TrackingLog,
    output_dir: str | Path,
    *,
    extra_summary: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Finish a tracking JSONL run and write derived tracking reports."""
    output_dir = Path(output_dir)
    summary = summarize_tracking_log(tracking_log.path)
    if extra_summary is not None:
        summary.update({key: value for key, value in extra_summary.items() if value is not None})
    tracking_log.finish_run(summary)
    write_tracking_reports(tracking_log.path, output_dir, summary=summary)
    return summary, output_dir / "analysis.json"


def format_tracking_summary_lines(
    tracking_summary: Mapping[str, Any],
    tracking_summary_path: str | Path | None = None,
) -> list[str]:
    """Format tracking details for an eval text report."""
    all_rollouts = tracking_summary.get("all_rollouts", {})
    metrics = all_rollouts.get("metrics", {})
    lines = [
        "",
        "Tracking:",
        f"Tracking summary: {tracking_summary_path}",
        "Tracking p95 over all rollouts:",
    ]
    for key in ("ee_pos_error_norm_m", "base_pos_error_norm_m", "base_rot_error_rad", "joint_l2_error_rad"):
        lines.append(f"  {key}: {_format_metric(metrics.get(key, {}).get('p95'))}")
    lines.extend([
        "Embodiment metrics over all rollouts:",
        f"  Avg normalized base deviation: {_format_metric(all_rollouts.get('avg_normalized_base_deviation'))}",
        f"  Max tilt utilization: {_format_metric(all_rollouts.get('max_tilt_utilization'))} rad",
        f"  Saturation rate: {_format_metric(100.0 * all_rollouts.get('saturation_rate', np.nan), 2)}%",
    ])
    return lines


def format_tracking_console_lines(tracking_summary: Mapping[str, Any]) -> list[str]:
    """Format a compact tracking block for final console output."""
    all_rollouts = tracking_summary.get("all_rollouts", {})
    return [
        "Tracking:",
        f"  Avg normalized base deviation: {_format_metric(all_rollouts.get('avg_normalized_base_deviation'))}",
        f"  Max tilt utilization: {_format_metric(all_rollouts.get('max_tilt_utilization'))} rad",
        f"  Saturation rate: {_format_metric(100.0 * all_rollouts.get('saturation_rate', np.nan), 2)}%",
    ]


def format_tracking_progress(
    record: Mapping[str, Any] | None,
    *,
    action_semantics: str | None = None,
) -> str:
    """Format a compact progress suffix from one tracking timestep record."""
    if record is None:
        return ""
    error = record.get("error", {})
    control = record.get("control", {})
    if action_semantics == "base_joint_absolute":
        return (
            f"base_err={float(error.get('base_pos_norm_m', math.nan)) * 100.0:.1f}cm "
            f"rot_err={math.degrees(float(error.get('base_rot_rad', math.nan))):.1f}deg "
            f"joint_l2={float(error.get('joint_l2_rad', math.nan)):.3f}rad"
        )
    return (
        f"ee_err={float(error.get('ee_pos_norm_m', math.nan)) * 100.0:.1f}cm "
        f"base_err={float(error.get('base_pos_norm_m', math.nan)) * 100.0:.1f}cm "
        f"tilt={float(control.get('base_tilt_rad', math.nan)):.3f}rad"
    )
