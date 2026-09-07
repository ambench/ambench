# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="LemonHarvesting-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)

register_env(
    task_id="LemonHarvesting-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


register_env(
    task_id="LemonHarvesting-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicy",
)


# Agile-motion evaluation variant.
register_env(
    task_id="LemonHarvesting-Am-EE-Abs-PID-Direct-Fast-v0",
    entry_point=f"{__name__}.lemon_harvesting_env:LemonHarvesting",
    env_cfg_entry_point=f"{__name__}.lemon_harvesting_env_cfg:LemonHarvestingEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.lemon_harvesting:LemonHarvestingPolicyFast",
)
