# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from ambench.tasks._registration import register_env

##
# Register Gym environments.
##

register_env(
    task_id="TossBall-Am-EE-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvEEAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-EE-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvEEAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-OmniHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvOmniHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-FAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvFAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-FAHexa-Abs-L1-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvFAHexaAbsL1Cfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-UAHexa-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvUAHexaAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)

register_env(
    task_id="TossBall-Am-FAHexa-Abs-MPC-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvFAHexaAbsMPCCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)


register_env(
    task_id="TossBall-Am-UAQuad-Abs-PID-Direct-v0",
    entry_point=f"{__name__}.toss_ball_env:TossBall",
    env_cfg_entry_point=f"{__name__}.toss_ball_env_cfg:TossBallEnvUAQuadAbsPIDCfg",
    scripted_policy_entry_point="ambench.policies.scripted.toss_ball:TossBallPolicy",
)
