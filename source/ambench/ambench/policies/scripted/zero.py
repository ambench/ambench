# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""No-op scripted policy for controller and recording smoke tests."""

import torch


class ZeroPolicy:
    """Return zero actions for any registered environment.

    Compatible with the record_demos_scripted.py interface:
    - __init__(env, inject_noise)
    - reset()
    - advance(obs, env_id) -> action tensor (1, action_dim)
    """

    def __init__(self, env, inject_noise: bool = False, **kwargs):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = self.env.device

    def reset(self):
        """Nothing to reset."""

    def advance(self, obs, env_id: int = 0) -> torch.Tensor:
        """Return one zero action."""
        action_dim = self.env.cfg.action_space
        return torch.zeros(1, action_dim, device=self.device)
