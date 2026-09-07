# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from umi.common.pose_util import (
    mat_to_pose,
    mat_to_pose10d,
    pose10d_to_mat,
    pose_to_mat,
)

from ambench.utils.image_processing import prepare_rgb_image
from ambench_learn.data.action_semantics import (
    BASE_JOINT_ABSOLUTE,
    EE_ABSOLUTE,
)


def resolve_dp_execution_schedule(action_horizon: int, obs_down_sample_steps: int) -> tuple[int, int, int]:
    """Return low-rate execution, query, and expanded prediction horizons."""
    if action_horizon < 2:
        raise ValueError(f"DP action_horizon must be >= 2, got {action_horizon}.")
    if obs_down_sample_steps < 1:
        raise ValueError(f"DP obs_down_sample_steps must be >= 1, got {obs_down_sample_steps}.")

    execution_horizon = action_horizon // 2
    query_frequency = execution_horizon * obs_down_sample_steps
    prediction_horizon_steps = action_horizon * obs_down_sample_steps
    return execution_horizon, query_frequency, prediction_horizon_steps


def resolve_dp_eval_action_semantics(action_mode: str, env_cfg) -> str:
    """Map a DP checkpoint action mode onto canonical Isaac env action semantics."""
    from ambench.controllers.control_pipeline import ActionMode

    configured_mode = env_cfg.robot_profile.control.action_mode
    configured_semantics = {
        ActionMode.ABSOLUTE_EE_POSE: EE_ABSOLUTE,
        ActionMode.ABSOLUTE_BASE_JOINTS: BASE_JOINT_ABSOLUTE,
    }.get(configured_mode)
    if action_mode == "base_joint":
        if configured_semantics != BASE_JOINT_ABSOLUTE:
            raise ValueError(
                "DP base_joint checkpoint requires an env with "
                f"action_semantics={BASE_JOINT_ABSOLUTE!r}. Got {configured_semantics!r}."
            )
        return BASE_JOINT_ABSOLUTE
    if action_mode == "ee_pose":
        if configured_semantics != EE_ABSOLUTE:
            raise ValueError(
                "DP ee_pose checkpoint requires an env with "
                f"action_semantics={EE_ABSOLUTE!r}. Got {configured_semantics!r}."
            )
        return EE_ABSOLUTE
    raise ValueError(f"Unsupported DP action mode: {action_mode!r}.")


def quat_wxyz_to_axis_angle(quat_wxyz):
    if len(quat_wxyz.shape) == 1:
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        r = R.from_quat(quat_xyzw)
        return r.as_rotvec()
    else:
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        r = R.from_quat(quat_xyzw)
        return r.as_rotvec()


def extract_robot_state(obs):
    """Extract measured robot state from an Isaac Lab policy observation."""

    if not isinstance(obs, dict):
        raise TypeError(f"Expected obs to be a dict with key 'policy', got {type(obs)}")
    if "policy" not in obs:
        raise KeyError(f"obs has keys {list(obs.keys())}, expected 'policy' key")

    policy_obs = obs["policy"][0].copy()

    for key, value in policy_obs.items():
        if torch.is_tensor(value):
            policy_obs[key] = value.detach().cpu().numpy()
        else:
            policy_obs[key] = np.array(value)

    if isinstance(policy_obs, dict):
        ee_pos = np.array(policy_obs.get("ee_pos", np.zeros(3)), dtype=np.float32)
        ee_quat_wxyz = np.array(policy_obs.get("ee_quat", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float32)
        gripper_width = np.array(policy_obs["gripper_width"], dtype=np.float32)

        if "arm_joint_pos" in policy_obs:
            arm_joint_pos = np.array(policy_obs["arm_joint_pos"], dtype=np.float32)
            if arm_joint_pos.ndim > 1:
                arm_joint_pos = arm_joint_pos.flatten()
        else:
            arm_joint_pos = np.zeros(4, dtype=np.float32)

        base_pos = policy_obs.get("base_pos", np.zeros(3))
        base_pos = np.array(base_pos, dtype=np.float32)

        base_quat_wxyz = policy_obs.get("base_quat", np.array([1.0, 0.0, 0.0, 0.0]))
        base_quat_wxyz = np.array(base_quat_wxyz, dtype=np.float32)

    return ee_pos, ee_quat_wxyz, gripper_width, arm_joint_pos, base_pos, base_quat_wxyz


def prepare_umi_image_observation(
    image: torch.Tensor | np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize an RGB image to ``(height, width)`` and return HWC ``uint8``."""

    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Expected a single image or batch size 1, got shape {image.shape}.")
        image = image[0]

    if image.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW RGB image, got shape {image.shape}.")
    channels_first = image.shape[0] in (3, 4)
    channels_last = image.shape[-1] in (3, 4)
    if channels_first == channels_last:
        raise ValueError(f"Expected an unambiguous HWC or CHW RGB image, got shape {image.shape}.")
    if channels_first:
        image = np.moveaxis(image, 0, -1)

    image = prepare_rgb_image(image)
    target_height, target_width = (int(value) for value in target_size)
    if target_height < 1 or target_width < 1:
        raise ValueError(f"Expected a positive target size, got {target_size}.")
    if image.shape[:2] != (target_height, target_width):
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return image


class ObservationBufferManager:
    def __init__(self, img_obs_horizon, low_dim_obs_horizon, image_keys=None):
        self.img_obs_horizon = img_obs_horizon
        self.low_dim_obs_horizon = low_dim_obs_horizon
        if image_keys is None:
            image_keys = ["camera0_rgb"]
        self.img_buffers = {key: [] for key in image_keys}
        self.robot_pos_buffer = []
        self.robot_rot_buffer = []
        self.robot_gripper_buffer = []
        self.robot_joint_pos_buffer = []
        self.robot_base_pos_buffer = []
        self.robot_base_rot_buffer = []

    def add_observation(
        self,
        img_dict,
        robot_pos,
        robot_rot,
        robot_gripper,
        robot_joint_pos=None,
        robot_base_pos=None,
        robot_base_rot=None,
    ):
        for key, img in img_dict.items():
            if key in self.img_buffers:
                self.img_buffers[key].append(img)

        self.robot_pos_buffer.append(robot_pos)
        self.robot_rot_buffer.append(robot_rot)
        self.robot_gripper_buffer.append(robot_gripper)

        if robot_joint_pos is not None:
            self.robot_joint_pos_buffer.append(robot_joint_pos)
        if robot_base_pos is not None:
            self.robot_base_pos_buffer.append(robot_base_pos)
        if robot_base_rot is not None:
            self.robot_base_rot_buffer.append(robot_base_rot)

        # Maintain horizon
        for key in self.img_buffers:
            while len(self.img_buffers[key]) > self.img_obs_horizon:
                self.img_buffers[key].pop(0)

        while len(self.robot_pos_buffer) > self.low_dim_obs_horizon:
            self.robot_pos_buffer.pop(0)
        while len(self.robot_rot_buffer) > self.low_dim_obs_horizon:
            self.robot_rot_buffer.pop(0)
        while len(self.robot_gripper_buffer) > self.low_dim_obs_horizon:
            self.robot_gripper_buffer.pop(0)
        while len(self.robot_joint_pos_buffer) > self.low_dim_obs_horizon:
            self.robot_joint_pos_buffer.pop(0)
        while len(self.robot_base_pos_buffer) > self.low_dim_obs_horizon:
            self.robot_base_pos_buffer.pop(0)
        while len(self.robot_base_rot_buffer) > self.low_dim_obs_horizon:
            self.robot_base_rot_buffer.pop(0)

    def get_stacked_observation(self):
        # Pad if not enough history
        # Check first buffer as reference
        first_key = next(iter(self.img_buffers))
        if len(self.img_buffers[first_key]) == 0:
            return None

        def pad_list(buffer, target_len):
            if len(buffer) >= target_len:
                return np.stack(buffer[-target_len:])
            else:
                # Pad with first element
                padding = [buffer[0]] * (target_len - len(buffer))
                return np.stack(padding + buffer)

        obs_dict = {
            "robot0_eef_pos": pad_list(self.robot_pos_buffer, self.low_dim_obs_horizon),
            "robot0_eef_rot_axis_angle": pad_list(self.robot_rot_buffer, self.low_dim_obs_horizon),
            "robot0_gripper_width": pad_list(self.robot_gripper_buffer, self.low_dim_obs_horizon),
        }

        for key, buffer in self.img_buffers.items():
            obs_dict[key] = pad_list(buffer, self.img_obs_horizon)

        if len(self.robot_joint_pos_buffer) > 0:
            obs_dict["robot0_joint_pos"] = pad_list(self.robot_joint_pos_buffer, self.low_dim_obs_horizon)

        if len(self.robot_base_pos_buffer) > 0:
            obs_dict["robot0_base_pos"] = pad_list(self.robot_base_pos_buffer, self.low_dim_obs_horizon)

        if len(self.robot_base_rot_buffer) > 0:
            obs_dict["robot0_base_rot_axis_angle"] = pad_list(self.robot_base_rot_buffer, self.low_dim_obs_horizon)

        return obs_dict


def interpolate_action_sequence(actions, target_horizon, output_type=None):
    """
    Interpolate between action waypoints to create smooth trajectories.

    Args:
        actions: Array of shape (N, D).
                 If D=7: [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
                 If D=8: [x, y, z, qw, qx, qy, qz, gripper]
        target_horizon: Total number of interpolated steps to generate.

    Returns:
        List of interpolated actions with the same dimension as input.
    """
    if len(actions) == 0:
        return []

    actions = np.array(actions)
    N, D = actions.shape

    # Determine format
    if D == 7:
        input_type = "axis_angle"
    elif D == 8:
        input_type = "quat"
    else:
        raise ValueError(f"Unsupported action dimension: {D}. Expected 7 (Axis-Angle) or 8 (Quat).")

    # if output_type is None, keep same as input
    if output_type is None:
        output_type = input_type
    assert output_type in ["axis_angle", "quat"], "output_type must be 'axis_angle' or 'quat'"

    # Convert actions to sim format (Pos + Quat + Gripper)
    sim_actions = []
    for action in actions:
        xyz = action[:3]
        if input_type == "axis_angle":
            axis_angle = action[3:6]
            grip = action[6:]
            # Convert axis-angle to quaternion
            if np.linalg.norm(axis_angle) > 1e-6:
                quat_xyzw = R.from_rotvec(axis_angle).as_quat()
                quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            else:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        else:  # quat
            quat_wxyz = action[3:7]
            grip = action[7:]

        sim_action = np.concatenate([xyz, quat_wxyz, grip])  # [x, y, z, qw, qx, qy, qz, gripper]
        sim_actions.append(sim_action)

    waypoints = sim_actions

    if len(waypoints) < 2:
        # If we have less than 2 waypoints, just repeat the single waypoint
        if len(waypoints) == 1:
            return [actions[0]] * target_horizon
        else:
            return []

    # Create evenly spaced interpolation across the entire trajectory
    interpolated_buffer = []

    # Create parameter values from 0 to 1 across the target horizon
    for i in range(target_horizon):
        t_global = i / (target_horizon - 1) if target_horizon > 1 else 0.0  # t ∈ [0, 1]

        # Find which segment this t_global falls into
        segment_length = 1.0 / (len(waypoints) - 1)
        segment_idx = min(int(t_global / segment_length), len(waypoints) - 2)

        # Local t within the current segment
        t_local = (t_global - segment_idx * segment_length) / segment_length
        t_local = np.clip(t_local, 0.0, 1.0)

        # Get the waypoints for this segment
        start_action = waypoints[segment_idx]
        end_action = waypoints[segment_idx + 1]

        # Extract components
        start_pos = start_action[:3]
        start_quat_wxyz = start_action[3:7]
        start_gripper = start_action[7:]

        end_pos = end_action[:3]
        end_quat_wxyz = end_action[3:7]
        end_gripper = end_action[7:]
        if np.dot(start_quat_wxyz, end_quat_wxyz) < 0.0:
            end_quat_wxyz = -end_quat_wxyz

        # Linear interpolation for position and gripper
        interp_pos = start_pos + t_local * (end_pos - start_pos)
        interp_gripper = start_gripper + t_local * (end_gripper - start_gripper)

        # SLERP for quaternion interpolation
        # Convert to scipy quaternion format [x, y, z, w]
        start_quat_xyzw = np.array([start_quat_wxyz[1], start_quat_wxyz[2], start_quat_wxyz[3], start_quat_wxyz[0]])
        end_quat_xyzw = np.array([end_quat_wxyz[1], end_quat_wxyz[2], end_quat_wxyz[3], end_quat_wxyz[0]])

        # Create rotation objects
        start_rot = R.from_quat(start_quat_xyzw)
        end_rot = R.from_quat(end_quat_xyzw)

        # SLERP interpolation
        slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
        interp_rot = slerp(t_local)
        interp_quat_xyzw = interp_rot.as_quat()

        # Convert back to [w, x, y, z] format
        interp_quat_wxyz = np.array(
            [interp_quat_xyzw[3], interp_quat_xyzw[0], interp_quat_xyzw[1], interp_quat_xyzw[2]]
        )

        # Convert back to original format
        if output_type == "axis_angle":
            interp_axis_angle = interp_rot.as_rotvec()
            interp_action = np.concatenate([interp_pos, interp_axis_angle, interp_gripper])
        else:
            interp_action = np.concatenate([interp_pos, interp_quat_wxyz, interp_gripper])

        interpolated_buffer.append(interp_action)

    return interpolated_buffer


def get_real_base_joint_action(action, env_obs, action_pose_repr="abs"):
    """Decode 14D relative base-joint DP actions into 12D absolute env commands."""

    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"Expected base_joint policy actions with shape (T, 14). Got {action.shape}.")

    base_pose_mat = pose_to_mat(
        np.concatenate(
            [env_obs["robot0_base_pos"][-1], env_obs["robot0_base_rot_axis_angle"][-1]],
            axis=-1,
        )
    )
    action_base_pose_mat = pose10d_to_mat(action[:, :9])
    decoded_base_mat = convert_pose_mat_rep(
        action_base_pose_mat,
        base_pose_mat=base_pose_mat,
        pose_rep=action_pose_repr,
        backward=True,
    )
    decoded_base_pose_axis_angle = mat_to_pose(decoded_base_mat)
    decoded_base_quat_xyzw = R.from_rotvec(decoded_base_pose_axis_angle[:, 3:6]).as_quat()
    decoded_base_quat_wxyz = decoded_base_quat_xyzw[:, [3, 0, 1, 2]]
    decoded_base_pose = np.concatenate(
        [decoded_base_pose_axis_angle[:, :3], decoded_base_quat_wxyz],
        axis=-1,
    )
    joint_targets = action[:, 9:13] + env_obs["robot0_joint_pos"][-1]
    gripper = action[:, 13:14]
    return np.concatenate([decoded_base_pose, joint_targets, gripper], axis=-1)


def get_real_base_joint_obs_dict(env_obs, shape_meta, obs_pose_repr="abs"):
    """Build policy observations for the base-joint DP task config."""

    obs_dict_np = {}
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        if obs_type == "rgb":
            image = env_obs[key]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            obs_dict_np[key] = np.moveaxis(image, -1, 1)
        elif key in ("robot0_joint_pos", "robot0_gripper_width"):
            obs_dict_np[key] = env_obs[key]

    base_pose_mat = pose_to_mat(
        np.concatenate(
            [env_obs["robot0_base_pos"], env_obs["robot0_base_rot_axis_angle"]],
            axis=-1,
        )
    )
    obs_base_pose_mat = convert_pose_mat_rep(
        base_pose_mat,
        base_pose_mat=base_pose_mat[-1],
        pose_rep=obs_pose_repr,
        backward=False,
    )
    obs_base_pose = mat_to_pose10d(obs_base_pose_mat)
    obs_dict_np["robot0_base_pos"] = obs_base_pose[:, :3]
    obs_dict_np["robot0_base_rot_axis_angle"] = obs_base_pose[:, 3:]
    return obs_dict_np


def interpolate_base_joint_action_sequence(actions, target_horizon):
    """Interpolate 12D Base+joints actions to raw-rate simulation commands."""

    if len(actions) == 0:
        return []

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError(f"Expected Base+joints actions with shape (N, 12). Got {actions.shape}.")

    if len(actions) < 2:
        return [actions[0]] * target_horizon

    interpolated_buffer = []
    for i in range(target_horizon):
        t_global = i / (target_horizon - 1) if target_horizon > 1 else 0.0
        segment_length = 1.0 / (len(actions) - 1)
        segment_idx = min(int(t_global / segment_length), len(actions) - 2)
        t_local = (t_global - segment_idx * segment_length) / segment_length
        t_local = np.clip(t_local, 0.0, 1.0)

        start_action = actions[segment_idx]
        end_action = actions[segment_idx + 1]

        interp_pos = start_action[:3] + t_local * (end_action[:3] - start_action[:3])
        interp_joints = start_action[7:11] + t_local * (end_action[7:11] - start_action[7:11])
        interp_gripper = start_action[11:12] + t_local * (end_action[11:12] - start_action[11:12])

        start_quat = start_action[3:7]
        end_quat = end_action[3:7]
        if np.dot(start_quat, end_quat) < 0.0:
            end_quat = -end_quat
        start_rot = R.from_quat([start_quat[1], start_quat[2], start_quat[3], start_quat[0]])
        end_rot = R.from_quat([end_quat[1], end_quat[2], end_quat[3], end_quat[0]])
        slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
        interp_quat_xyzw = slerp(t_local).as_quat()
        interp_quat_wxyz = np.array(
            [interp_quat_xyzw[3], interp_quat_xyzw[0], interp_quat_xyzw[1], interp_quat_xyzw[2]]
        )

        interpolated_buffer.append(np.concatenate([interp_pos, interp_quat_wxyz, interp_joints, interp_gripper]))

    return interpolated_buffer
