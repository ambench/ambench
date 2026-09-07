# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

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
from ambench.robots.visuals.propeller_specs import (
    FA_HEXA_PROP_POSITIONS,
    FA_HEXA_PROP_RPYS,
    FA_HEXA_PROP_SPIN_DIRS,
)
from ambench.utils.assets import LOCAL_ASSET_DIR

USD_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/hexa_scorpion.usd"
URDF_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/urdf/hexa_scorpion.urdf"

PROPELLER_VIZ_RELATIVE_PATHS = tuple(
    f"base_link/visuals/prop_mount_{idx}/prop_spin_{idx}" for idx in range(FA_HEXA_PROP_SPIN_DIRS.numel())
)


# --- FA-Hexa configuration ---
FA_HEXA_CONFIG = ArticulationCfg(
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
            "arm_link1_pitch_joint": 0.0,
            "arm_link2_pitch_joint": 0.0,
            "arm_link3_pitch_joint": 0.0,
            "arm_link4_roll_joint": 0.0,
            "arm_lfinger_joint": 0.001,
            "arm_rfinger_joint": 0.001,
        },
        joint_vel={},  # required
    ),
    actuators={
        "arm_link1_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link1_pitch_joint"],
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
        "arm_link4_roll_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link4_roll_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_lfinger_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_lfinger_joint"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
        "arm_rfinger_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_rfinger_joint"],
            stiffness=100.0,
            damping=20.0,
            effort_limit_sim=1000.0,
            velocity_limit_sim=5.0,
        ),
    },
)


def rotor_layout(
    device: str | torch.device = "cpu",
) -> RotorLayout:
    """Return the FA-Hexa rotor layout derived from its URDF."""

    device = torch.device(device) if isinstance(device, str) else device

    # Thruster positions in body frame (from URDF)
    motor_positions_b = FA_HEXA_PROP_POSITIONS.to(device=device)

    # Thruster orientations (rpy) from URDF
    motor_rpys = FA_HEXA_PROP_RPYS.to(device=device)

    # Compute thruster directions: Rotate [0, 0, 1] (z-axis) by rpy rotation
    motor_dirs_b = torch.zeros((6, 3), device=device, dtype=torch.float32)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)

    # Stack all RPYs and compute rotation matrices in batch
    rpy_intrinsic = motor_rpys[:, [2, 1, 0]]  # Reorder to [yaw, pitch, roll] for intrinsic ZYX
    R_matrices = math_utils.matrix_from_euler(rpy_intrinsic, convention="ZYX")  # [6, 3, 3]

    # Rotate z-axis by each rotation matrix
    z_axis_expanded = z_axis.unsqueeze(0).expand(6, -1).unsqueeze(-1)  # [6, 3, 1]
    motor_dirs_b = torch.bmm(R_matrices, z_axis_expanded).squeeze(-1)  # [6, 3]

    # Normalize directions
    motor_dirs_b = motor_dirs_b / torch.norm(motor_dirs_b, dim=1, keepdim=True)

    # Spin directions: Based on prop_arm naming (cw/ccw)
    # Thruster 0: cw -> +1, Thruster 1: ccw -> -1, Thruster 2: cw -> +1
    # Thruster 3: ccw -> -1, Thruster 4: cw -> +1, Thruster 5: ccw -> -1
    spin_dirs = FA_HEXA_PROP_SPIN_DIRS.to(device=device)

    return RotorLayout(
        positions_b=motor_positions_b,
        thrust_axes_b=motor_dirs_b,
        spin_directions=spin_dirs,
    )


@configclass
class FAHexaRobotSpecCfg(RobotSpecCfg):
    """Specification for the fully actuated hexarotor platform."""

    robot_id: str = "fa_hexa"
    asset: ArticulationCfg = FA_HEXA_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    control_body_name: str = "base_link"
    base_body_name: str = "base_link"
    ee_body_name: str = "ee_link"
    arm_joint_names: tuple[str, ...] = (
        "arm_link1_pitch_joint",
        "arm_link2_pitch_joint",
        "arm_link3_pitch_joint",
        "arm_link4_roll_joint",
    )
    end_effector: EndEffectorFrameCfg = EndEffectorFrameCfg()
    gripper: GripperSpecCfg = GripperSpecCfg(
        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),
    )
    max_arm_reach: float = 1.24
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
    )
