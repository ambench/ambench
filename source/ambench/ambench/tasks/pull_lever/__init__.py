# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="PullLever-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)

register_env(
    task_id="PullLever-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)


register_env(
    task_id="PullLever-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.pull_lever_env:PullLever",
    env_cfg_entry_point=f"{__name__}.pull_lever_env_cfg:PullLeverEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.pull_lever:PullLeverPolicy",
)
