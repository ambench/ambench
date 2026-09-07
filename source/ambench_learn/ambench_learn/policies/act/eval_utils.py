# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Policy-specific ACT evaluation helpers that do not launch Isaac Sim."""

from __future__ import annotations

from typing import Any

from ambench_learn.data.action_semantics import (
    resolve_policy_action_representation_for_dataset,
)
from ambench_learn.policies.act.se_relative_processor import (
    BaseJointAbsoluteTrajectoryProcessorStep,
    EELocalAbsoluteTrajectoryProcessorStep,
)


def resolve_act_action_representation(action_semantics: str, policy_cfg: Any, postprocessor: Any) -> str:
    """Resolve ACT representation from enabled decoder steps and checkpoint config."""

    decoder_representations = []
    if any(isinstance(step, EELocalAbsoluteTrajectoryProcessorStep) and step.enabled for step in postprocessor.steps):
        decoder_representations.append("ee_local_relative")
    if any(isinstance(step, BaseJointAbsoluteTrajectoryProcessorStep) and step.enabled for step in postprocessor.steps):
        decoder_representations.append("base_joint_relative")
    if len(set(decoder_representations)) > 1:
        raise ValueError(
            f"ACT checkpoint has multiple enabled relative decoders: {sorted(set(decoder_representations))}."
        )
    requested = (
        decoder_representations[0] if decoder_representations else getattr(policy_cfg, "action_representation", None)
    )
    return resolve_policy_action_representation_for_dataset(
        action_semantics,
        requested,
        policy_name="ACT",
    )
