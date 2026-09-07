# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Composed robot, controller, action, IK, and camera profiles."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ambench.controllers.control_pipeline import (
    ActionMode,
    ControlPipelineCfg,
    EndEffectorPosePipeline,
    ManipulatorDirectPipeline,
    ManipulatorIKPipeline,
    ManipulatorMPCPipeline,
)
from ambench.controllers.controller_cfg import ControllerCfg
from ambench.controllers.l1_adaptive_ctrl import L1AdaptiveController
from ambench.controllers.pid_4dof_ctrl import PID4DOFController, PID4DOFGains
from ambench.controllers.pid_6dof_ctrl import PID6DOFController, PID6DOFGains
from ambench.controllers.pyroki_ik_ctrl import PyrokiIKControllerConfig
from ambench.controllers.wholebody_mpc_ctrl import WholeBodyMPCController
from ambench.robots.eef import EndEffectorRobotSpecCfg
from ambench.robots.fa_hexa import URDF_FILE_PATH as FA_HEXA_URDF
from ambench.robots.fa_hexa import FAHexaRobotSpecCfg
from ambench.robots.omni_hexa import URDF_FILE_PATH as OMNI_HEXA_URDF
from ambench.robots.omni_hexa import OmniHexaRobotSpecCfg
from ambench.robots.robot_cfg import RobotSpecCfg
from ambench.robots.ua_hexa import UAHexaRobotSpecCfg
from ambench.robots.ua_quad import URDF_FILE_PATH as UA_QUAD_URDF
from ambench.robots.ua_quad import UAQuadRobotSpecCfg


def _camera(
    prim_path: str,
    *,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=384,
        width=384,
        data_types=["rgb"],
        depth_clipping_behavior="zero",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=19.6,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="ros"),
    )


DEFAULT_EE_CAMERA = _camera(
    "/World/envs/env_.*/Robot/ee_link/ee_camera",
    position=(0.08, 0.0, 0.04),
    rotation=(0.5, -0.5, 0.5, -0.5),
)
DEFAULT_BASE_CAMERA = _camera(
    "/World/envs/env_.*/Robot/base_link/base_camera",
    position=(0.1, 0.0, 0.06),
    rotation=(0.5, -0.5, 0.5, -0.5),
)
OMNI_EE_CAMERA = _camera(
    "/World/envs/env_.*/Robot/end_effector/ee_camera",
    position=(-0.04, 0.0, -0.03),
    rotation=(0.707, 0.0, 0.0, -0.707),
)
QUAD_EE_CAMERA = _camera(
    "/World/envs/env_.*/Robot/ee_link/ee_camera",
    position=(0.011, -0.08, 0.0),
    rotation=(0.0, 0.0, 1.0, 0.0),
)
QUAD_BASE_CAMERA = _camera(
    "/World/envs/env_.*/Robot/base_link/base_camera",
    position=(0.15, 0.0, 0.05),
    rotation=(0.5, -0.5, 0.5, -0.5),
)


@configclass
class RobotProfileCfg:
    """Complete robot and control selection for a task config."""

    robot: RobotSpecCfg = MISSING
    control: ControlPipelineCfg = MISSING
    ee_camera: CameraCfg = DEFAULT_EE_CAMERA
    base_camera: CameraCfg | None = None


_PID6_GAINS = PID6DOFGains(
    kp_pos=200.0,
    kd_pos=120.0,
    ki_pos=80.0,
    kp_rot=200.0,
    kd_rot=120.0,
    ki_rot=120.0,
)
_OMNI_PID6_GAINS = PID6DOFGains(
    kp_pos=200.0,
    kd_pos=120.0,
    ki_pos=80.0,
    kp_rot=240.0,
    kd_rot=130.0,
    ki_rot=130.0,
)
_PID4_GAINS = PID4DOFGains(
    kp_pos_xy=10.0,
    kp_pos_z=30.0,
    kd_pos_xy=8.0,
    kd_pos_z=10.0,
    ki_pos_xy=6.0,
    ki_pos_z=10.0,
    kp_rot=200.0,
    kd_rot=120.0,
    ki_rot=120.0,
)
_QUAD_PID4_GAINS = PID4DOFGains(
    kp_pos_xy=10.0,
    kp_pos_z=30.0,
    kd_pos_xy=8.0,
    kd_pos_z=10.0,
    ki_pos_xy=6.0,
    ki_pos_z=10.0,
    kp_rot=150.0,
    kd_rot=20.0,
    ki_rot=20.0,
)
_L1_PARAMS = {
    "low_pass_filter_bandwidth": 0.3,
    "sigma_max_lin": 2.0,
    "sigma_max_ang": 2.0,
    "adaptive_mix": 1.0,
}


def _pid6(gains: PID6DOFGains) -> ControllerCfg:
    return ControllerCfg(class_type=PID6DOFController, params={"gains": gains})


def _l1(gains: PID6DOFGains) -> ControllerCfg:
    l1_gains = PID6DOFGains(
        kp_pos=gains.kp_pos,
        kd_pos=gains.kd_pos,
        ki_pos=0.0,
        kp_rot=gains.kp_rot,
        kd_rot=gains.kd_rot,
        ki_rot=0.0,
    )
    return ControllerCfg(class_type=L1AdaptiveController, params={"gains": l1_gains, "l1_params": _L1_PARAMS})


def _pid4(gains: PID4DOFGains) -> ControllerCfg:
    return ControllerCfg(
        class_type=PID4DOFController,
        params={"gains": gains, "outer_loop_decimation": 3},
    )


def _hexa_ik(
    fix_orientation: tuple[bool, bool, bool], safety_margin: float, smoothness: float
) -> PyrokiIKControllerConfig:
    return PyrokiIKControllerConfig(
        urdf_path=FA_HEXA_URDF,
        target_link_name="ee_link",
        fix_base_position=(False, False, False),
        fix_base_orientation=fix_orientation,
        initial_base_position=(0.0, 0.0, 1.0),
        initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
        enable_collision=False,
        cost_weights={
            "smoothness_base_pos": smoothness,
            "smoothness_base_ori": 0.1,
            "rest_arm": 0.1,
        },
        base_pos_limits={"max_x": 1.0, "min_z": 0.0},
        safety_margin=safety_margin,
        arm_joint_names=list(FAHexaRobotSpecCfg().arm_joint_names),
    )


def _omni_ik() -> PyrokiIKControllerConfig:
    spec = OmniHexaRobotSpecCfg()
    return PyrokiIKControllerConfig(
        urdf_path=OMNI_HEXA_URDF,
        target_link_name=spec.ee_body_name,
        fix_base_position=(False, False, False),
        fix_base_orientation=(False, False, False),
        initial_base_position=(0.0, 0.0, 1.0),
        initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
        enable_collision=False,
        cost_weights={
            "smoothness_base_pos": 1.0,
            "smoothness_base_ori": 1.0,
            "rest_arm": 0.1,
        },
        base_pos_limits={"max_x": 1.0, "min_z": 0.0},
        safety_margin=0.3,
        arm_joint_names=list(spec.arm_joint_names),
        initial_joint_positions=dict(spec.asset.init_state.joint_pos),
    )


def _quad_ik() -> PyrokiIKControllerConfig:
    return PyrokiIKControllerConfig(
        urdf_path=UA_QUAD_URDF,
        target_link_name="ee_link",
        fix_base_position=(False, False, False),
        fix_base_orientation=(True, True, False),
        initial_base_position=(0.0, 0.0, 1.0),
        initial_base_orientation=(1.0, 0.0, 0.0, 0.0),
        enable_collision=False,
        cost_weights={
            "smoothness_base_pos": 1.0,
            "smoothness_base_ori": 0.1,
            "rest_arm": 0.1,
        },
        base_pos_limits={"max_x": 1.0, "min_z": 0.0},
        safety_margin=0.15,
        arm_joint_names=list(UAQuadRobotSpecCfg().arm_joint_names),
    )


def _profile(
    robot: RobotSpecCfg,
    pipeline: type,
    action_mode: ActionMode,
    controller: ControllerCfg,
    *,
    ik: PyrokiIKControllerConfig | None = None,
    ee_camera: CameraCfg = DEFAULT_EE_CAMERA,
    base_camera: CameraCfg | None = None,
) -> RobotProfileCfg:
    return RobotProfileCfg(
        robot=robot,
        control=ControlPipelineCfg(
            class_type=pipeline,
            action_mode=action_mode,
            controller=controller,
            ik=ik,
        ),
        ee_camera=ee_camera,
        base_camera=base_camera,
    )


def _pose_profiles(
    robot: RobotSpecCfg,
    pipeline: type,
    pid_gains: PID6DOFGains,
    *,
    ik: PyrokiIKControllerConfig | None = None,
    ee_camera: CameraCfg = DEFAULT_EE_CAMERA,
    base_camera: CameraCfg | None = None,
) -> tuple[RobotProfileCfg, RobotProfileCfg]:
    common = {
        "robot": robot,
        "pipeline": pipeline,
        "ik": ik,
        "ee_camera": ee_camera,
        "base_camera": base_camera,
    }
    return (
        _profile(action_mode=ActionMode.ABSOLUTE_EE_POSE, controller=_pid6(pid_gains), **common),
        _profile(action_mode=ActionMode.ABSOLUTE_EE_POSE, controller=_l1(pid_gains), **common),
    )


EE_ABS_PID, EE_ABS_L1 = _pose_profiles(
    EndEffectorRobotSpecCfg(),
    EndEffectorPosePipeline,
    _PID6_GAINS,
)
FA_HEXA_ABS_PID, FA_HEXA_ABS_L1 = _pose_profiles(
    FAHexaRobotSpecCfg(),
    ManipulatorIKPipeline,
    _PID6_GAINS,
    ik=_hexa_ik((True, True, True), 0.85, 1.0),
    base_camera=DEFAULT_BASE_CAMERA,
)
OMNI_HEXA_ABS_PID, OMNI_HEXA_ABS_L1 = _pose_profiles(
    OmniHexaRobotSpecCfg(),
    ManipulatorIKPipeline,
    _OMNI_PID6_GAINS,
    ik=_omni_ik(),
    ee_camera=OMNI_EE_CAMERA,
    base_camera=DEFAULT_BASE_CAMERA,
)

FA_HEXA_BASE_JOINT_ABS_PID = _profile(
    FAHexaRobotSpecCfg(),
    ManipulatorDirectPipeline,
    ActionMode.ABSOLUTE_BASE_JOINTS,
    _pid6(_PID6_GAINS),
    ik=_hexa_ik((True, True, True), 0.85, 1.0),
    base_camera=DEFAULT_BASE_CAMERA,
)
FA_HEXA_BASE_JOINT_ABS_L1 = _profile(
    FAHexaRobotSpecCfg(),
    ManipulatorDirectPipeline,
    ActionMode.ABSOLUTE_BASE_JOINTS,
    _l1(_PID6_GAINS),
    ik=_hexa_ik((True, True, True), 0.85, 1.0),
    base_camera=DEFAULT_BASE_CAMERA,
)
OMNI_HEXA_BASE_JOINT_ABS_PID = _profile(
    OmniHexaRobotSpecCfg(),
    ManipulatorDirectPipeline,
    ActionMode.ABSOLUTE_BASE_JOINTS,
    _pid6(_OMNI_PID6_GAINS),
    ee_camera=OMNI_EE_CAMERA,
    base_camera=DEFAULT_BASE_CAMERA,
)

UA_HEXA_ABS_PID = _profile(
    UAHexaRobotSpecCfg(),
    ManipulatorIKPipeline,
    ActionMode.ABSOLUTE_EE_POSE,
    _pid4(_PID4_GAINS),
    ik=_hexa_ik((True, True, False), 0.8, 8.0),
    base_camera=DEFAULT_BASE_CAMERA,
)
UA_QUAD_ABS_PID = _profile(
    UAQuadRobotSpecCfg(),
    ManipulatorIKPipeline,
    ActionMode.ABSOLUTE_EE_POSE,
    _pid4(_QUAD_PID4_GAINS),
    ik=_quad_ik(),
    ee_camera=QUAD_EE_CAMERA,
    base_camera=QUAD_BASE_CAMERA,
)

_MPC_PARAMS = {
    "T": 0.8,
    "N": 32,
    "Q": [6000.0] * 3 + [200.0] * 3 + [20.0] * 3 + [50.0] * 3 + [20.0] * 3 + [10.0] * 4,
    "R": [0.02] * 3 + [0.1] * 3 + [1.0] * 4,
    "R_delta": [0.01] * 3 + [0.01] * 3 + [200.0] * 4,
    "pos_min": (-1.0, -5.0, 0.0),
    "pos_max": (1.2, 2.0, 4.0),
    "joint_min": (-180.0, -180.0, -180.0, -180.0),
    "joint_max": (180.0, 180.0, 180.0, 180.0),
    "default_arm_angle": (0.0, 0.0, 0.0, 0.0),
    "output_filter_gain": [1.0] * 6 + [0.5] * 4,
    "compile_dir": "./tmp/acados_mpc_build",
    "model_name": "fa_hexa_mpc",
}
FA_HEXA_ABS_MPC = _profile(
    FAHexaRobotSpecCfg(),
    ManipulatorMPCPipeline,
    ActionMode.ABSOLUTE_EE_POSE,
    ControllerCfg(class_type=WholeBodyMPCController, params={"mpc_params": _MPC_PARAMS}),
    base_camera=DEFAULT_BASE_CAMERA,
)
