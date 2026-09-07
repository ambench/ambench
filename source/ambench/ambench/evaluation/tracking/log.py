# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""JSONL read/write helpers for tracking diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRACKING_EVENT_TYPES = frozenset({
    "run_metadata",
    "rollout_start",
    "timestep",
    "rollout_summary",
    "run_summary",
})


def _validate_event_type(event_type: Any, *, path: Path | None = None, line_no: int | None = None) -> str:
    if not isinstance(event_type, str) or event_type not in TRACKING_EVENT_TYPES:
        location = "" if path is None or line_no is None else f" in {path}:{line_no}"
        raise ValueError(f"Tracking event{location} has unsupported event type: {event_type!r}.")
    return event_type


def json_safe(value: Any) -> Any:
    """Return a strict-JSON-compatible representation of common numeric objects."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def iter_tracking_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield strict JSON events from a tracking JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"Tracking event in {path}:{line_no} must be a JSON object.")
            if "event" not in event:
                raise ValueError(f"Tracking event in {path}:{line_no} is missing required 'event' field.")
            _validate_event_type(event["event"], path=path, line_no=line_no)
            yield event


class TrackingLog:
    """Append-only writer for canonical tracking JSONL events."""

    def __init__(self, path: str | Path, metadata: Mapping[str, Any] | None = None, *, reset: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")
        if metadata is not None:
            self.write_event("run_metadata", metadata=dict(metadata))

    def write_event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event_type = _validate_event_type(event_type)
        event = {
            "event": event_type,
            **payload,
        }
        event = json_safe(event)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
        return event

    def start_rollout(self, *, rollout_idx: int, batch_env_index: int | None = None, **metadata: Any) -> dict[str, Any]:
        return self.write_event(
            "rollout_start",
            rollout_idx=int(rollout_idx),
            batch_env_index=None if batch_env_index is None else int(batch_env_index),
            metadata=metadata,
        )

    def write_timestep(
        self,
        *,
        rollout_idx: int,
        record: Mapping[str, Any],
        batch_env_index: int | None = None,
    ) -> dict[str, Any]:
        return self.write_event(
            "timestep",
            rollout_idx=int(rollout_idx),
            batch_env_index=None if batch_env_index is None else int(batch_env_index),
            record=dict(record),
        )

    def finish_rollout(
        self,
        *,
        rollout_idx: int,
        success: bool,
        executed_steps: int,
        subtask_completion: float | None,
        summary: Mapping[str, Any],
        batch_env_index: int | None = None,
        termination_reason: str | None = None,
    ) -> dict[str, Any]:
        return self.write_event(
            "rollout_summary",
            rollout_idx=int(rollout_idx),
            batch_env_index=None if batch_env_index is None else int(batch_env_index),
            success=bool(success),
            executed_steps=int(executed_steps),
            subtask_completion=subtask_completion,
            termination_reason=termination_reason,
            summary=dict(summary),
        )

    def finish_run(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return self.write_event("run_summary", summary=dict(summary))
