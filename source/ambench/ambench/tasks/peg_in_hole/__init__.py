# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="PegInHole-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)

register_env(
    task_id="PegInHole-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)


register_env(
    task_id="PegInHole-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.peg_in_hole_env:PegInHole",
    env_cfg_entry_point=f"{__name__}.peg_in_hole_env_cfg:PegInHoleEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.peg_in_hole:PegInHolePolicy",
)
