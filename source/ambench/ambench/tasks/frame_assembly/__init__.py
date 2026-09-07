# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="FrameAssembly-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)

register_env(
    task_id="FrameAssembly-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)


register_env(
    task_id="FrameAssembly-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.frame_assembly_env:FrameAssembly",
    env_cfg_entry_point=f"{__name__}.frame_assembly_env_cfg:FrameAssemblyEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.frame_assembly:FrameAssemblyPolicy",
)
