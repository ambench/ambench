# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="WipeWindow-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)

register_env(
    task_id="WipeWindow-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)


register_env(
    task_id="WipeWindow-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.wipe_window_env:WipeWindow",
    env_cfg_entry_point=f"{__name__}.wipe_window_env_cfg:WipeWindowEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.wipe_window:WipeWindowPolicy",
)
