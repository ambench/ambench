# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

FA_HEXA_PROP_POSITIONS = torch.tensor(
    [
        [0.34175, 0.1585, -0.058],
        [-0.0335, 0.375, -0.058],
        [-0.30825, 0.2165, -0.058],
        [-0.30825, -0.2165, -0.058],
        [-0.0335, -0.375, -0.058],
        [0.34175, -0.1585, -0.058],
    ],
    dtype=torch.float32,
)

FA_HEXA_PROP_RPYS = torch.tensor(
    [
        [-0.464, -0.252, 0.060],
        [0.0, 0.524, 0.0],
        [0.464, -0.252, -0.060],
        [-0.464, -0.252, 0.060],
        [0.0, 0.524, 0.0],
        [0.464, -0.252, -0.060],
    ],
    dtype=torch.float32,
)

FA_HEXA_PROP_SPIN_DIRS = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)


UA_HEXA_PROP_POSITIONS = torch.tensor(
    [
        [0.32495, 0.1875, -0.065],
        [0.0, 0.375, -0.065],
        [-0.32505, 0.1875, -0.065],
        [-0.32505, -0.1875, -0.065],
        [0.0, -0.375, -0.065],
        [0.32495, -0.1875, -0.065],
    ],
    dtype=torch.float32,
)

UA_HEXA_PROP_SPIN_DIRS = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
