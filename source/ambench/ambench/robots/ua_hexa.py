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
from ambench.robots.visuals.propeller_specs import (
    UA_HEXA_PROP_POSITIONS,
    UA_HEXA_PROP_SPIN_DIRS,
)
from ambench.utils.assets import LOCAL_ASSET_DIR

USD_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/hexa_scorpion_uac.usd"
PROPELLER_VIZ_RELATIVE_PATHS = tuple(
    f"base_link/visuals/prop_mount_{idx}/prop_spin_{idx}" for idx in range(UA_HEXA_PROP_SPIN_DIRS.numel())
)

# --- UA-Hexa configuration ---
UA_HEXA_CONFIG = ArticulationCfg(
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
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link2_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link2_pitch_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link3_pitch_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link3_pitch_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=5.0,
        ),
        "arm_link4_roll_joint": ImplicitActuatorCfg(
            joint_names_expr=["arm_link4_roll_joint"],
            stiffness=100000.0,
            damping=1000.0,
            effort_limit_sim=20.0,
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
    """Return the UA-Hexa rotor layout derived from its URDF."""

    device = torch.device(device) if isinstance(device, str) else device

    # Thruster positions in body frame (from URDF)
    # URDF thruster positions:
    motor_positions_b = UA_HEXA_PROP_POSITIONS.to(device=device)

    # All thrusters have rpy="0 0 0" in URDF, so they point downward (z-axis)
    # Thruster directions: all point in -z direction (downward)
    motor_dirs_b = torch.tensor([[0.0, 0.0, 1.0]] * UA_HEXA_PROP_POSITIONS.shape[0], device=device, dtype=torch.float32)

    # Spin directions: Based on prop_arm naming (cw/ccw)
    spin_dirs = UA_HEXA_PROP_SPIN_DIRS.to(device=device)

    return RotorLayout(
        positions_b=motor_positions_b,
        thrust_axes_b=motor_dirs_b,
        spin_directions=spin_dirs,
    )


@configclass
class UAHexaRobotSpecCfg(RobotSpecCfg):
    """Specification for the under-actuated hexarotor platform."""

    robot_id: str = "ua_hexa"
    asset: ArticulationCfg = UA_HEXA_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
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
        fully_actuated=False,
        aerodynamics=AerodynamicCfg(),
        propeller_viz=PropellerVizCfg(
            relative_prim_paths=PROPELLER_VIZ_RELATIVE_PATHS,
            local_axis=(0.0, 0.0, 1.0),
        ),
    )
