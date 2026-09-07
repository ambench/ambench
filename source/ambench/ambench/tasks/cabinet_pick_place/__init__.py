# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="CabinetPickPlace-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)

register_env(
    task_id="CabinetPickPlace-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)


register_env(
    task_id="CabinetPickPlace-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.cabinet_pick_place_env:CabinetPickPlace",
    env_cfg_entry_point=f"{__name__}.cabinet_pick_place_env_cfg:CabinetPickPlaceEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.cabinet_pick_place:CabinetPickPlacePolicy",
)
