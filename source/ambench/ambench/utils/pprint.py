# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Pretty-print utilities."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

__all__ = ["pprint"]

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"
_ANSI_YELLOW = "\033[33m"
_ANSI_INFO = "\033[0m"
_ANSI_RED = "\033[31m"


def _to_numpy(value: Any) -> Any:
    """Convert torch tensors to CPU numpy arrays; leave other types as-is."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Flatten nested dict/list/tuple structures into (path, leaf_value) pairs."""
    value = _to_numpy(value)

    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            child_prefix = f"{prefix}.{key_str}" if prefix else key_str
            yield from _flatten(child, child_prefix)
        return

    if isinstance(value, (list, tuple)):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _flatten(child, child_prefix)
        return

    yield prefix, value


def _format_leaf(value: Any, decimals: int = 4) -> str:
    """Format a leaf value into a compact string."""
    value = _to_numpy(value)
    precision = max(0, int(decimals))

    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating):

            def float_fmt(x: Any) -> str:
                return f"{float(x):.{precision}f}"

            return np.array2string(
                value,
                separator=", ",
                max_line_width=140,
                formatter={"float_kind": float_fmt},
            )
        return np.array2string(value, separator=", ", max_line_width=140)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def _format_bool_like(value: Any) -> str:
    """Format a bool or array-of-bools with a color cue."""
    value = _to_numpy(value)

    if isinstance(value, np.ndarray):
        text = np.array2string(value, separator=", ", max_line_width=140)
        try:
            any_true = bool(np.any(value))
        except Exception:
            any_true = False
        return _colorize(text, _ANSI_RED if any_true else _ANSI_INFO)

    try:
        truthy = bool(value)
    except Exception:
        truthy = False
    return _colorize(str(value), _ANSI_RED if truthy else _ANSI_INFO)


def _colorize(text: str, color: str) -> str:
    """Wrap text with ANSI color."""
    return f"{color}{text}{_ANSI_RESET}"


def _flatten_pprint_obs(obs: Any) -> Iterator[tuple[str, Any]]:
    """Flatten observations with a friendlier default for the common single-env policy dict."""
    if isinstance(obs, dict):
        policy = obs.get("policy")
        if isinstance(policy, list) and len(policy) == 1 and isinstance(policy[0], dict):
            yield from _flatten(policy[0])
            return
    yield from _flatten(obs)


def pprint(
    obs: Any | None = None,
    reward: Any | None = None,
    terminated: Any | None = None,
    truncated: Any | None = None,
    *,
    info: Any | None = None,
    success_criteria: Any | None = None,
    indent: int = 2,
    decimals: int = 4,
    logging_level: str | None = None,
) -> None:
    """Pretty-print rollout outputs.

    Args:
        obs: Observation, often a nested dict/list/tuple of tensors.
        reward: Reward value.
        terminated: Termination signal.
        truncated: Truncation signal.
        info: Step info dict, optionally containing ``success_criteria``.
        success_criteria: Success-criteria dict to print directly.
        indent: Indentation spaces for section bodies.
        decimals: Number of decimals for floating-point formatting.
    """
    if logging_level is None:
        log_level = logging.INFO
    else:
        log_level = logging._nameToLevel.get(logging_level.upper(), logging.INFO)
    lines: list[str] = []
    pad = " " * max(0, int(indent))

    if obs is not None:
        flattened_obs = list(_flatten_pprint_obs(obs))
        if flattened_obs:
            lines.append(_colorize("Obs", _ANSI_INFO))
            for path, leaf in flattened_obs:
                label = path if path else "(obs)"
                lines.append(f"{pad}{label}: {_format_leaf(leaf, decimals=decimals)}")
        else:
            lines.append(f"{_colorize('Obs', _ANSI_INFO)}\n{pad}{_format_leaf(obs, decimals=decimals)}")

    summary_parts: list[str] = []
    if reward is not None:
        summary_parts.append(f"reward={_format_leaf(reward, decimals=decimals)}")
    if terminated is not None:
        summary_parts.append(f"terminated={_format_bool_like(terminated)}")
    if truncated is not None:
        summary_parts.append(f"truncated={_format_bool_like(truncated)}")
    if summary_parts:
        lines.append(_colorize("Status", _ANSI_YELLOW))
        lines.append(f"{pad}" + "  ".join(summary_parts))

    if success_criteria is None and isinstance(info, dict):
        success_criteria = info.get("success_criteria")
    if success_criteria:
        lines.append(_colorize("Success Criteria", _ANSI_YELLOW))
        for name, value in success_criteria.items():
            lines.append(f"{pad}{name}: {_format_bool_like(value)}")

    block = "\n".join(lines)
    if block:
        logger.log(log_level, "\n%s", block)
