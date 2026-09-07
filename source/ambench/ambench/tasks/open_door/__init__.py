# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="OpenDoor-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)

register_env(
    task_id="OpenDoor-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)


register_env(
    task_id="OpenDoor-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.open_door_env:OpenDoor",
    env_cfg_entry_point=f"{__name__}.open_door_env_cfg:OpenDoorEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.open_door:OpenDoorPolicy",
)
