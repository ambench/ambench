# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from lerobot.processor import TransitionKey
from lerobot.utils.constants import OBS_STATE

from ambench_learn.data.action_semantics import to_ee_local_relative_trajectory
from ambench_learn.policies.act.se_relative_processor import (
    EELocalRelativeTrajectoryProcessorStep,
)


def test_relative_processor_anchors_action_chunk_to_latest_observation() -> None:
    state_history = torch.tensor([[
        [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.02],
        [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 0.03],
    ]])
    actions = torch.tensor([[
        [5.0, 7.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.5],
        [6.0, 8.0, 10.0, 1.0, 0.0, 0.0, 0.0, -0.5],
    ]])
    processor = EELocalRelativeTrajectoryProcessorStep(enabled=True)

    result = processor({
        TransitionKey.OBSERVATION: {OBS_STATE: state_history[:, -1]},
        TransitionKey.ACTION: actions,
    })

    expected = to_ee_local_relative_trajectory(actions, state_history[:, -1])
    torch.testing.assert_close(result[TransitionKey.ACTION], expected)
    torch.testing.assert_close(processor.resolve_decode_anchor(), state_history[:, -1])
