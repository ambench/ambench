# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="RotateValve-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)

register_env(
    task_id="RotateValve-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvFAHexaBaseJointAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)

register_env(
    task_id="RotateValve-Am-FAHexa-BaseJoint-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvFAHexaBaseJointAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)

register_env(
    task_id="RotateValve-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


register_env(
    task_id="RotateValve-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicy",
)


# Agile-motion evaluation variant.
register_env(
    task_id="RotateValve-Am-EE-Abs-PID-Direct-Fast-v0",
    entry_point=f"{__name__}.rotate_valve_env:RotateValve",
    env_cfg_entry_point=f"{__name__}.rotate_valve_env_cfg:RotateValveEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.rotate_valve:RotateValvePolicyFast",
)
