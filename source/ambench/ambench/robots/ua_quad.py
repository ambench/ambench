# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
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

USD_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/quad_scorpion.usd"
URDF_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/urdf/quad_scorpion.urdf"

PROPELLER_SPIN_DIRS = (-1.0, 1.0, -1.0, 1.0)
PROPELLER_VIZ_RELATIVE_PATHS = tuple(f"base_link/visuals/prop_mount_{idx}/prop_spin_{idx}" for idx in range(4))

# Fixed transform from the task command frame (+X through the gripper) to the
# UAQuad ``ee_link`` frame.
EE_CMD_ORIENTATION_OFFSET_WXYZ = (0.5, -0.5, -0.5, 0.5)

# Center of the closed finger volume, measured from the authored USD geometry.
EE_TOOL_TIP_OFFSET_LOCAL = (0.011, -0.032, -0.075)


def gripper_joint_pos_from_object_width(object_width: float) -> float:
    """Return the symmetric finger position that fits an object at the grasp center."""
    closed_inner_overlap = 0.006
    return 0.5 * (object_width + closed_inner_overlap)


# --- UA-Quad configuration ---
UA_QUAD_CONFIG = ArticulationCfg(
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
            "arm_link1_yaw_joint": 0.0,
            "arm_link2_pitch_joint": 0.0,
            "arm_link3_pitch_joint": 0.0,
            "arm_link4_pitch_joint": 0.0,
            "arm_lfinger_joint": 0.001,
            "arm_rfinger_joint": 0.001,
        },
        joint_vel={},  # required
    ),
    actuators={
        "arm_link1_yaw_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link1_yaw_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link2_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link2_pitch_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link3_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link3_pitch_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link4_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link4_pitch_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_lfinger_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_lfinger_joint"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
        "arm_rfinger_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_rfinger_joint"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
    },
)


def rotor_layout(
    device: str | torch.device = "cpu",
) -> RotorLayout:
    """Return the UA-Quad rotor layout."""

    device = torch.device(device) if isinstance(device, str) else device

    # Motor distance from center: 30cm = 0.3m
    motor_distance = 0.3
    # For square layout, distance from center to corner = distance / sqrt(2)
    motor_offset = motor_distance / (2.0**0.5)  # ≈ 0.212m

    # Thruster positions in body frame
    # Motor 1: left top, Motor 2: left bottom, Motor 3: right bottom, Motor 4: right top
    motor_positions_b = torch.tensor(
        [
            [motor_offset, motor_offset, 0.05],  # Motor 0: left top
            [-motor_offset, motor_offset, 0.05],  # Motor 1: left bottom
            [-motor_offset, -motor_offset, 0.05],  # Motor 2: right bottom
            [motor_offset, -motor_offset, 0.05],  # Motor 3: right top
        ],
        device=device,
        dtype=torch.float32,
    )

    # All motors point downward (z-axis)
    motor_dirs_b = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # Motor 1
            [0.0, 0.0, 1.0],  # Motor 2
            [0.0, 0.0, 1.0],  # Motor 3
            [0.0, 0.0, 1.0],  # Motor 4
        ],
        device=device,
        dtype=torch.float32,
    )

    # Spin directions: Alternating pattern for torque cancellation
    # Motor 0 (left top): CCW, Motor 1 (left bottom): CW
    # Motor 2 (right bottom): CCW, Motor 3 (right top): CW
    spin_dirs = torch.tensor(PROPELLER_SPIN_DIRS, device=device, dtype=torch.float32)

    return RotorLayout(
        positions_b=motor_positions_b,
        thrust_axes_b=motor_dirs_b,
        spin_directions=spin_dirs,
    )


@configclass
class UAQuadRobotSpecCfg(RobotSpecCfg):
    """Specification for the under-actuated quadrotor platform."""

    robot_id: str = "ua_quad"
    asset: ArticulationCfg = UA_QUAD_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    control_body_name: str = "base_link"
    base_body_name: str = "base_link"
    ee_body_name: str = "ee_link"
    arm_joint_names: tuple[str, ...] = (
        "arm_link1_yaw_joint",
        "arm_link2_pitch_joint",
        "arm_link3_pitch_joint",
        "arm_link4_pitch_joint",
    )
    end_effector: EndEffectorFrameCfg = EndEffectorFrameCfg(
        tool_tip_offset_local=EE_TOOL_TIP_OFFSET_LOCAL,
        command_to_link_quat_wxyz=EE_CMD_ORIENTATION_OFFSET_WXYZ,
    )
    gripper: GripperSpecCfg = GripperSpecCfg(
        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),
        joint_position_from_object_width=gripper_joint_pos_from_object_width,
    )
    max_arm_reach: float = 0.67
    multirotor: MultirotorSpecCfg = MultirotorSpecCfg(
        layout_provider=rotor_layout,
        actuator=RotorActuatorCfg(
            thrust_limits=(0.0, 23.0),
            reaction_torque_ratio=0.02,
            propeller_radius=0.190,
        ),
        fully_actuated=False,
        aerodynamics=AerodynamicCfg(),
        propeller_viz=PropellerVizCfg(
            relative_prim_paths=PROPELLER_VIZ_RELATIVE_PATHS,
            local_axis=(0.0, 0.0, 1.0),
        ),
    )
