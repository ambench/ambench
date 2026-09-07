# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Posthoc plotting helpers for canonical tracking JSONL logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .log import iter_tracking_events


def _collect_series(path: Path, metric_path: str) -> dict[int, list[float]]:
    series: dict[int, list[float]] = {}
    for event in iter_tracking_events(path):
        if event["event"] != "timestep":
            continue
        value: Any = event.get("record", {})
        for part in metric_path.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is None:
            continue
        series.setdefault(int(event["rollout_idx"]), []).append(float(value))
    return series


def plot_tracking_log(path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Generate simple derived metric plots from a tracking JSONL file."""
    path = Path(path)
    output_dir = path.parent / "plots" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for metric_name, metric_path in {
        "ee_pos_error_norm_m": "error.ee_pos_norm_m",
        "base_pos_error_norm_m": "error.base_pos_norm_m",
        "base_tilt_rad": "control.base_tilt_rad",
    }.items():
        series = _collect_series(path, metric_path)
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for rollout_idx, values in series.items():
            ax.plot(values, label=f"rollout {rollout_idx}")
        ax.set_title(metric_name)
        ax.set_xlabel("tracking step")
        ax.grid(True)
        ax.legend()
        output = output_dir / f"{metric_name}.png"
        fig.tight_layout()
        fig.savefig(output)
        plt.close(fig)
        outputs.append(output)
    return outputs
