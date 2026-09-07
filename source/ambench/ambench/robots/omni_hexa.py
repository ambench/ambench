# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Omni-Hexa robot configuration.

Hexacopter with tiltable motor arms (initialized at :data:`INIT_MOTOR_ARM_ANGLES`), a 3-DOF manipulator,
and a gripper. Thruster layout for control allocation is derived from ``urdf/omni_scorpion.urdf``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from ambench.robots.robot_cfg import (
    AerodynamicCfg,
    EndEffectorFrameCfg,
    GripperSpecCfg,
    MultirotorSpecCfg,
    RobotSpecCfg,
    RotorActuatorCfg,
    RotorLayout,
)
from ambench.robots.visuals.propeller_cfg import PropellerVizCfg
from ambench.utils.assets import LOCAL_ASSET_DIR

USD_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/omni_scorpion.usd"
URDF_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/urdf/omni_scorpion.urdf"

PROPELLER_SPIN_DIRS = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
PROPELLER_VIZ_RELATIVE_PATHS = tuple(f"prop_{idx}/visuals" for idx in range(1, 7))

# Initial rotor-arm joint angles (rad) at reset and for variable-tilt allocation geometry.
INIT_MOTOR_ARM_ANGLES: tuple[float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

GRIPPER_OPEN_POS: float = -0.5
GRIPPER_CLOSED_POS: float = 0.0
INIT_SERVO1_POS: float = -3.14159265359 / 4.0
INIT_SERVO2_POS: float = 3.14159265359 / 4.0 * 3.0
INIT_SERVO3_POS: float = 0.0

# Command-frame -> URDF ``end_effector`` link frame at all-zero joint configuration.
# Ry(+90 deg): maps tool +Z to body +X (toward wall).
EE_CMD_ORIENTATION_OFFSET_WXYZ: tuple[float, float, float, float] = (
    0.7071067811865476,
    0.0,
    0.7071067811865476,
    0.0,
)

# Per-arm URDF joint data: (origin_xyz, origin_rpy, axis_xyz, prop_xyz in motor_arm frame).
_MOTOR_ARM_SPECS: tuple[
    tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
    ...,
] = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, -1.5707963267948966), (0.5, -0.8660254037844386, 0.0), (0.089682, -0.153329, 0.03)),
    ((0.0, 0.003, 0.0), (0.0, 0.0, -1.5707963267948966), (1.0, 0.0, 0.0), (0.181138, 0.0, 0.03)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, -1.5707963267948966), (0.5, 0.8660254037844386, 0.0), (0.090192, 0.156211, 0.03)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, -1.5707963267948966), (-0.5, 0.8660254037844386, 0.0), (-0.089399, 0.154892, 0.03)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, -1.5707963267948966), (-1.0, 0.0, 0.0), (-0.178796, 0.0, 0.03)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, -1.5707963267948966), (-0.5, -0.8660254037844386, 0.0), (-0.087683, -0.150797, 0.03)),
)

# --- Omni-Hexa articulation configuration ---
OMNI_HEXA_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_FILE_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "base_motor_arm1": INIT_MOTOR_ARM_ANGLES[0],
            "base_motor_arm2": INIT_MOTOR_ARM_ANGLES[1],
            "base_motor_arm3": INIT_MOTOR_ARM_ANGLES[2],
            "base_motor_arm4": INIT_MOTOR_ARM_ANGLES[3],
            "base_motor_arm5": INIT_MOTOR_ARM_ANGLES[4],
            "base_motor_arm6": INIT_MOTOR_ARM_ANGLES[5],
            "servo1": INIT_SERVO1_POS,
            "servo2": INIT_SERVO2_POS,
            "servo3": INIT_SERVO3_POS,
            "servo4": GRIPPER_OPEN_POS,
        },
        joint_vel={},
    ),
    actuators={
        "motor_arms": ImplicitActuatorCfg(
            joint_names_expr=["base_motor_arm[1-6]"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
        ),
        "servo_manipulator": ImplicitActuatorCfg(
            joint_names_expr=["servo[1-3]"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["servo4"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
    },
)


def _homogeneous_from_origin(
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 4x4 transform from URDF joint origin xyz/rpy (roll, pitch, yaw)."""
    transform = torch.eye(4, device=device, dtype=dtype)
    rpy_zyx = torch.tensor([origin_rpy[2], origin_rpy[1], origin_rpy[0]], device=device, dtype=dtype)
    transform[:3, :3] = math_utils.matrix_from_euler(rpy_zyx, convention="ZYX")
    transform[:3, 3] = torch.tensor(origin_xyz, device=device, dtype=dtype)
    return transform


def _rotation_about_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix for a revolute joint."""
    axis = axis / torch.linalg.norm(axis)
    quat = math_utils.quat_from_angle_axis(angle.unsqueeze(0), axis.unsqueeze(0))
    return math_utils.matrix_from_quat(quat)[0]


def _motor_geometry_at_angles(
    motor_arm_angles: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prop positions, thrust directions, and body-frame rotations from URDF FK.

    Returns:
        motor_positions_b: [6, 3]
        motor_dirs_b: [6, 3]
        motor_rotations_b: [6, 3, 3] prop frame rotation in base_link
    """
    dtype = torch.float32
    motor_positions = torch.zeros((6, 3), device=device, dtype=dtype)
    motor_dirs = torch.zeros((6, 3), device=device, dtype=dtype)
    motor_rotations = torch.zeros((6, 3, 3), device=device, dtype=dtype)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    for arm_idx, (origin_xyz, origin_rpy, axis_xyz, prop_xyz) in enumerate(_MOTOR_ARM_SPECS):
        origin_tf = _homogeneous_from_origin(origin_xyz, origin_rpy, device=device, dtype=dtype)
        axis = torch.tensor(axis_xyz, device=device, dtype=dtype)
        joint_rot = _rotation_about_axis(axis, motor_arm_angles[arm_idx])
        arm_rot = origin_tf[:3, :3] @ joint_rot
        arm_tf = origin_tf.clone()
        arm_tf[:3, :3] = arm_rot

        prop_pos_h = arm_tf @ torch.tensor([*prop_xyz, 1.0], device=device, dtype=dtype)
        motor_positions[arm_idx] = prop_pos_h[:3]
        thrust_dir = arm_rot @ z_axis
        motor_dirs[arm_idx] = thrust_dir / torch.linalg.norm(thrust_dir)
        motor_rotations[arm_idx] = arm_rot

    return motor_positions, motor_dirs, motor_rotations


def _motor_tilt_basis_at_reference(
    reference_angles: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Motor positions and cos/sin thrust bases at a trim reference for variable-tilt allocation."""
    motor_positions_b, cos_dirs_b, motor_rotations_b = _motor_geometry_at_angles(reference_angles, device=device)
    sin_dirs_b = torch.zeros((6, 3), device=device, dtype=torch.float32)
    for arm_idx, (_origin_xyz, _origin_rpy, axis_xyz, _prop_xyz) in enumerate(_MOTOR_ARM_SPECS):
        axis_local = torch.tensor(axis_xyz, device=device, dtype=torch.float32)
        axis_b = motor_rotations_b[arm_idx] @ axis_local
        axis_b = axis_b / torch.linalg.norm(axis_b)
        sin_dirs_b[arm_idx] = torch.linalg.cross(axis_b, cos_dirs_b[arm_idx])
        sin_dirs_b[arm_idx] = sin_dirs_b[arm_idx] / torch.linalg.norm(sin_dirs_b[arm_idx])

    return motor_positions_b, cos_dirs_b, sin_dirs_b


def rotor_layout(
    device: str | torch.device = "cpu",
) -> RotorLayout:
    """Return the variable-tilt rotor layout at the configured hover trim."""
    device = torch.device(device) if isinstance(device, str) else device
    reference_angles = torch.tensor(INIT_MOTOR_ARM_ANGLES, device=device, dtype=torch.float32)
    motor_positions_b, cos_dirs_b, sin_dirs_b = _motor_tilt_basis_at_reference(reference_angles, device=device)
    spin_dirs = torch.tensor(PROPELLER_SPIN_DIRS, device=device, dtype=torch.float32)
    return RotorLayout(
        positions_b=motor_positions_b,
        thrust_axes_b=cos_dirs_b,
        spin_directions=spin_dirs,
        tilt_cos_axes_b=cos_dirs_b,
        tilt_sin_axes_b=sin_dirs_b,
    )


def omni_ik_initial_joint_positions() -> dict[str, float]:
    """Full actuated joint defaults for Pyroki ``initial_joint_cfg`` / rest pose."""
    positions = {f"base_motor_arm{i + 1}": INIT_MOTOR_ARM_ANGLES[i] for i in range(6)}
    positions["servo4"] = GRIPPER_OPEN_POS
    positions["servo1"] = INIT_SERVO1_POS
    positions["servo2"] = INIT_SERVO2_POS
    positions["servo3"] = INIT_SERVO3_POS
    return positions


@configclass
class OmniHexaRobotSpecCfg(RobotSpecCfg):
    """Specification for the variable-tilt Omni-Hexa."""

    robot_id: str = "omni_hexa"
    asset: ArticulationCfg = OMNI_HEXA_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    control_body_name: str = "base_link"
    base_body_name: str = "base_link"
    ee_body_name: str = "end_effector"
    arm_joint_names: tuple[str, ...] = ("servo1", "servo2", "servo3")
    end_effector: EndEffectorFrameCfg = EndEffectorFrameCfg(
        tool_tip_offset_local=(0.0, 0.0, 0.0445),
        command_to_link_quat_wxyz=EE_CMD_ORIENTATION_OFFSET_WXYZ,
    )
    gripper: GripperSpecCfg = GripperSpecCfg(
        joint_names=("servo4",),
        open_joint_positions=(GRIPPER_OPEN_POS,),
        closed_joint_positions=(GRIPPER_CLOSED_POS,),
    )
    max_arm_reach: float = 0.5
    multirotor: MultirotorSpecCfg = MultirotorSpecCfg(
        layout_provider=rotor_layout,
        actuator=RotorActuatorCfg(
            thrust_limits=(0.0, 23.0),
            reaction_torque_ratio=0.02,
            propeller_radius=0.152,
        ),
        fully_actuated=True,
        aerodynamics=AerodynamicCfg(),
        propeller_viz=PropellerVizCfg(
            relative_prim_paths=PROPELLER_VIZ_RELATIVE_PATHS,
            local_axis=(0.0, 0.0, 1.0),
        ),
        motor_arm_joint_names=(
            "base_motor_arm1",
            "base_motor_arm2",
            "base_motor_arm3",
            "base_motor_arm4",
            "base_motor_arm5",
            "base_motor_arm6",
        ),
    )
