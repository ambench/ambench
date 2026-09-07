# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from ambench.robots.robot_cfg import EndEffectorFrameCfg, GripperSpecCfg, RobotSpecCfg
from ambench.utils.assets import LOCAL_ASSET_DIR

USD_FILE_PATH = f"{LOCAL_ASSET_DIR}/robots/end_effector.usd"

# --- End Effector configuration ---
END_EFFECTOR_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_FILE_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "arm_lfinger_joint": 0.001,
            "arm_rfinger_joint": 0.001,
        },
        joint_vel={},  # required
    ),
    actuators={
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


@configclass
class EndEffectorRobotSpecCfg(RobotSpecCfg):
    """Specification for the free end-effector test articulation."""

    robot_id: str = "end_effector"
    asset: ArticulationCfg = END_EFFECTOR_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    control_body_name: str = "ee_link"
    base_body_name: str = "ee_link"
    ee_body_name: str = "ee_link"
    end_effector: EndEffectorFrameCfg = EndEffectorFrameCfg()
    gripper: GripperSpecCfg = GripperSpecCfg(
        joint_names=("arm_lfinger_joint", "arm_rfinger_joint"),
    )
