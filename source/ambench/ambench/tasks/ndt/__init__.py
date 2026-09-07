# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="NDT-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)

register_env(
    task_id="NDT-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)


register_env(
    task_id="NDT-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.ndt_env:NDT",
    env_cfg_entry_point=f"{__name__}.ndt_env_cfg:NDTEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.NDT:NDTPolicy",
)
