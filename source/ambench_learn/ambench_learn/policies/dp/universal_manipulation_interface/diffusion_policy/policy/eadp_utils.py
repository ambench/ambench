import isaaclab.utils.math as math_utils
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_trajectory_cost_with_gradient(env, trajectory, mpc_planner, u_prev):
    """
    Compute the gradient of the tracking cost for a given reference EE trajectory.
    
    Args: 
        env: Isaac Lab environment 
        trajectory (np.ndarray): Waypoint array of shape (N, 7) [x,y,z, qw,qx,qy,qz].

    Returns:
        tuple: (total_cost, tracking_cost, gradient, mpc_x_opt)
            - total_cost (float): Total MPC cost.
            - tracking_cost (float): Tracking component of MPC cost.
            - gradient (np.ndarray): Gradient of the tracking cost w.r.t. the initial state.
            - mpc_x_opt (np.ndarray): MPC optimized states for end-effector position extraction.
    """
    robot = env.unwrapped.robot
    # Assume an aerial manipulator with a four-joint arm.
    # TODO: Generalize for EEF only robot
    base_link_idx = robot.find_bodies("base_link")[0][0]
    arm_joint_ids, _ = robot.find_joints("arm_link[1-4]_.*_joint")
    base_state = robot.data.body_link_state_w[:, base_link_idx, :]
    arm_state = robot.data.joint_pos[:, arm_joint_ids]

    pos = base_state[:, 0:3]
    quat = base_state[:, 3:7]
    quat_np = quat.detach().cpu().numpy()
    current_base_ori = np.array([quat_np[:, 3], quat_np[:, 0], quat_np[:, 1], quat_np[:, 2]])
    base_euler = R.from_quat(current_base_ori.T).as_euler('zyx', degrees=False)[::-1]
    lin_vel = base_state[:, 7:10]
    ang_vel_w = base_state[:, 10:13]
    ang_vel_b = math_utils.quat_apply_inverse(quat, ang_vel_w)

    ee_pos_ref = trajectory[:, 0:3] 
    ee_quat_ref = trajectory[:, 3:7]

    # Convert everything to numpy
    pos = pos.detach().cpu().numpy().squeeze()
    lin_vel = lin_vel.detach().cpu().numpy().squeeze()
    arm_state = arm_state.detach().cpu().numpy().squeeze()
    ang_vel_b = ang_vel_b.detach().cpu().numpy().squeeze()
    base_euler = base_euler.squeeze()
    # DEBUG: check states shapes
    print(f'pos shape: {pos.shape}, lin_vel shape: {lin_vel.shape}, arm_state shape: {arm_state.shape}, ang_vel_b shape: {ang_vel_b.shape}, base_euler shape: {base_euler.shape}')
    
    # Run MPC planner with tracking cost extraction
    total_cost, tracking_cost, tracking_gradient, mpc_x_opt, mpc_u_opt = mpc_planner.optimize_with_tracking_gradient(
        pos, lin_vel, arm_state, base_euler, ang_vel_b, 
        ee_pos_ref, ee_quat_ref, 
        u_prev)

    return total_cost, tracking_cost, tracking_gradient, mpc_u_opt
