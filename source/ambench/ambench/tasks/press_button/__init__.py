# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="PressButton-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)

register_env(
    task_id="PressButton-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvFAHexaBaseJointAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)

register_env(
    task_id="PressButton-Am-FAHexa-BaseJoint-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvFAHexaBaseJointAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)

register_env(
    task_id="PressButton-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


register_env(
    task_id="PressButton-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicy",
)


# -----------------------------------
# Agile motion
register_env(
    task_id="PressButton-Am-EE-Abs-PID-Direct-Fast-v0",
    entry_point=f"{__name__}.press_button_env:PressButton",
    env_cfg_entry_point=f"{__name__}.press_button_env_cfg:PressButtonEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.press_button:PressButtonPolicyFast",
)
