# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="PushSlider-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)

register_env(
    task_id="PushSlider-Am-FAHexa-BaseJoint-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvFAHexaBaseJointAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)

register_env(
    task_id="PushSlider-Am-FAHexa-BaseJoint-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvFAHexaBaseJointAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)

register_env(
    task_id="PushSlider-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


register_env(
    task_id="PushSlider-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicy",
)


# -----------------------------------
# Agile motion
register_env(
    task_id="PushSlider-Am-EE-Abs-PID-Direct-Fast-v0",
    entry_point=f"{__name__}.push_slider_env:PushSlider",
    env_cfg_entry_point=f"{__name__}.push_slider_env_cfg:PushSliderEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.push_slider:PushSliderPolicyFast",
)
