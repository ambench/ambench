# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class PropellerVizCfg:
    """Configuration for visual-only propeller animation."""

    enabled: bool = True
    relative_prim_paths: tuple[str, ...] = ()
    max_angular_speed: float = 180.0
    local_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
