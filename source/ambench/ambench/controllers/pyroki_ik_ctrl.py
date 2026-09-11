# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Pyroki IK Controller for joint-space control via inverse kinematics.

This controller uses Pyroki to solve inverse kinematics (IK) for desired end-effector poses,
returning joint angles that can be used for position control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyroki as pk
import torch
import yourdfpy

from ambench.controllers.pyroki_ctrl import solve_floating_base_ik

logger = logging.getLogger(__name__)


@dataclass
class PyrokiIKControllerConfig:
    """Configuration for Pyroki IK Controller.

    Attributes:
        urdf_path: Path to the robot URDF file.
        target_link_name: Name of the end-effector link to control. If None, uses last link.
        fix_base_position: Tuple of 3 bools (x, y, z) - True means fixed, False means free.
                          Default: (False, False, False) - all free.
        fix_base_orientation: Tuple of 3 bools (roll, pitch, yaw) - True means fixed, False means free.
                             Default: (True, True, False) - fix roll/pitch, free yaw.
        initial_base_position: Initial base position (x, y, z) in meters. Default: (0.0, 0.0, 0.0).
        initial_base_orientation: Initial base orientation as quaternion (w, x, y, z).
                                 Default: (1.0, 0.0, 0.0, 0.0) - identity.
        enable_collision: Whether to enable collision avoidance in IK solving.
        cost_weights: Optional dict to customize cost weights for floating base IK.
                     Available keys: "pose_pos", "pose_ori", "joint_limit", "smoothness_joints",
                     "smoothness_base_pos", "smoothness_base_ori", "self_collision", "base_pos_limit", "rest_arm".
                     Default: {"smoothness_base_pos": 1.0, "smoothness_base_ori": 1.0, "rest_arm": 1.0}.
        base_pos_limits: Optional dict to set base position limits. Keys: "max_x", "min_x", "max_y", "min_y", "max_z", "min_z".
                         Default: {"max_x": 1.0, "min_z": 0.0}.
        arm_joint_names: Optional list of joint names for the robotic arm (for rest cost). If None, uses first 3 joints.
        initial_joint_cfg: Optional actuated joint vector for Pyroki default/rest (highest priority).
        initial_joint_positions: Optional name -> angle map used to build initial_joint_cfg when
            initial_joint_cfg is None. Non-arm joints are reset to this default after each IK solve.
        safety_margin: Safety margin in meters for base position limits to avoid collisions with obstacles.
                       Default: 1.0.
    """

    urdf_path: str | None = None
    target_link_name: str | None = None
    fix_base_position: tuple[bool, bool, bool] = (False, False, False)  # (x, y, z)
    fix_base_orientation: tuple[bool, bool, bool] = (
        True,
        True,
        False,
    )  # (roll, pitch, yaw)
    initial_base_position: tuple[float, float, float] | None = None
    initial_base_orientation: tuple[float, float, float, float] | None = None
    enable_collision: bool = False
    cost_weights: dict[str, float] | None = field(
        default_factory=lambda: {
            "smoothness_base_pos": 1.0,  # too low -> unsmooth, too high -> manipulation mode
            "smoothness_base_ori": 0.1,
            "rest_arm": 0.1,
        }
    )
    base_pos_limits: dict[str, float] | None = None
    arm_joint_names: list[str] | None = None
    initial_joint_cfg: np.ndarray | None = None
    initial_joint_positions: dict[str, float] | None = None
    safety_margin: float = 1.0


class PyrokiIKController:
    """Pyroki-based Inverse Kinematics Controller.

    This controller solves IK using Pyroki to compute joint angles and base pose
    from desired end-effector poses for floating base robots.

    Example:
        >>> controller = PyrokiIKController(
        ...     num_envs=10,
        ...     device="cuda",
        ...     config=PyrokiIKControllerConfig(
        ...         urdf_path="path/to/robot.urdf",
        ...         target_link_name="ee_link",
        ...         fix_base_position=(False, False, False),  # All position DOFs free
        ...         fix_base_orientation=(True, True, False),  # Fix roll/pitch, free yaw
        ...         initial_base_position=(0.0, 0.0, 1.0)
        ...     )
        ... )
        >>> joint_angles, base_pos, base_quat = controller.compute(
        ...     target_pos=torch.tensor([[0.5, 0.0, 0.3]]),
        ...     target_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ... )
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        config: PyrokiIKControllerConfig | None = None,
    ):
        """Initialize the Pyroki IK Controller.

        Args:
            num_envs: Number of parallel environments.
            device: Device to run computations on ("cuda" or "cpu").
            config: Controller configuration. If None, uses defaults.
        """

        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.config = config if config is not None else PyrokiIKControllerConfig()

        urdf_file = Path(self.config.urdf_path).resolve()
        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_file}")

        def filename_handler(fname: str) -> str:
            """Resolves mesh file paths relative to the URDF file's directory."""
            return yourdfpy.filename_handler_magic(fname, dir=urdf_file.parent)

        try:
            urdf = yourdfpy.URDF.load(str(urdf_file), filename_handler=filename_handler)
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF from {urdf_file}: {e}")

        # Create Pyroki robot (temporary) to resolve actuated joint order.
        robot_temp = pk.Robot.from_urdf(urdf)
        joint_names = list(robot_temp.joints.actuated_names)
        if self.config.initial_joint_cfg is not None:
            initial_joint_cfg = np.array(self.config.initial_joint_cfg, dtype=np.float64)
        elif self.config.initial_joint_positions:
            initial_joint_cfg = self._joint_cfg_from_dict(joint_names, self.config.initial_joint_positions)
        else:
            initial_joint_cfg = np.zeros(robot_temp.joints.num_actuated_joints)

        # Create Pyroki robot
        self.robot = pk.Robot.from_urdf(urdf, default_joint_cfg=initial_joint_cfg)
        self._default_joint_cfg = initial_joint_cfg.copy()

        # Determine target link
        all_links = list(self.robot.links.names)
        if self.config.target_link_name:
            if self.config.target_link_name not in all_links:
                raise ValueError(
                    f"Target link '{self.config.target_link_name}' not found. Available links: {all_links}"
                )
            self.target_link = self.config.target_link_name
        else:
            # Use last link (typically end-effector)
            self.target_link = all_links[-1]

        # Initialize collision if enabled
        self.robot_coll = None
        if self.config.enable_collision:
            self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # Store current joint configuration for warm-starting IK
        self.current_joint_cfg = np.tile(initial_joint_cfg.reshape(1, -1), (num_envs, 1))  # [num_envs, num_joints]

        # Store current base pose for warm-starting floating base IK
        initial_base_pos = np.array(
            self.config.initial_base_position if self.config.initial_base_position is not None else [0.0, 0.0, 0.0]
        )
        initial_base_wxyz = np.array(
            self.config.initial_base_orientation
            if self.config.initial_base_orientation is not None
            else [1.0, 0.0, 0.0, 0.0]
        )
        self.current_base_pos = np.tile(initial_base_pos.reshape(1, -1), (num_envs, 1))  # [num_envs, 3]
        self.current_base_wxyz = np.tile(initial_base_wxyz.reshape(1, -1), (num_envs, 1))  # [num_envs, 4]

        # For logging IK failures
        self._ik_failure_count = 0

    @staticmethod
    def _joint_cfg_from_dict(joint_names: list[str], positions: dict[str, float]) -> np.ndarray:
        cfg = np.zeros(len(joint_names), dtype=np.float64)
        for idx, name in enumerate(joint_names):
            if name in positions:
                cfg[idx] = positions[name]
        return cfg

    def _apply_fixed_joint_positions(self, joint_cfg: np.ndarray, joint_names: list[str]) -> np.ndarray:
        """Overwrite joints locked in simulation so IK FK matches the sim model."""
        arm_joint_names = self.config.arm_joint_names
        if not arm_joint_names:
            return joint_cfg
        arm_indices = {joint_names.index(name) for name in arm_joint_names if name in joint_names}
        cfg = np.array(joint_cfg, dtype=np.float64, copy=True)
        for idx in range(len(joint_names)):
            if idx not in arm_indices:
                cfg[idx] = self._default_joint_cfg[idx]
        return cfg

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset controller state.

        Args:
            env_ids: Optional tensor of environment IDs to reset. If None, resets all environments.
        """
        default_cfg = self._default_joint_cfg
        if env_ids is None:
            self.current_joint_cfg = np.tile(default_cfg.reshape(1, -1), (self.num_envs, 1))
            initial_base_pos = np.array(
                self.config.initial_base_position if self.config.initial_base_position is not None else [0.0, 0.0, 0.0]
            )
            initial_base_wxyz = np.array(
                self.config.initial_base_orientation
                if self.config.initial_base_orientation is not None
                else [1.0, 0.0, 0.0, 0.0]
            )
            self.current_base_pos = np.tile(initial_base_pos.reshape(1, -1), (self.num_envs, 1))
            self.current_base_wxyz = np.tile(initial_base_wxyz.reshape(1, -1), (self.num_envs, 1))
        else:
            # Reset specific environments
            env_ids_np = env_ids.cpu().numpy()
            self.current_joint_cfg[env_ids_np] = default_cfg
            initial_base_pos = np.array(
                self.config.initial_base_position if self.config.initial_base_position is not None else [0.0, 0.0, 0.0]
            )
            initial_base_wxyz = np.array(
                self.config.initial_base_orientation
                if self.config.initial_base_orientation is not None
                else [1.0, 0.0, 0.0, 0.0]
            )
            self.current_base_pos[env_ids_np] = initial_base_pos
            self.current_base_wxyz[env_ids_np] = initial_base_wxyz

    def compute(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        prev_joint_cfg: torch.Tensor | None = None,
        prev_base_pos: torch.Tensor | None = None,
        prev_base_quat: torch.Tensor | None = None,
        base_pos_limits: dict[str, float] | list[dict[str, float]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute joint angles and base pose from desired end-effector pose using IK.

        Args:
            target_pos: Desired end-effector position in world frame, shape (num_envs, 3).
            target_quat: Desired end-effector orientation as quaternion [w, x, y, z] in world frame,
                        shape (num_envs, 4).
            prev_joint_cfg: Previous joint configuration for warm-starting IK, shape (num_envs, num_joints).
                           If None, uses internal state.
            prev_base_pos: Previous base position for warm-starting floating base IK, shape (num_envs, 3).
                          If None, uses internal state.
            prev_base_quat: Previous base orientation for warm-starting floating base IK, shape (num_envs, 4).
                           If None, uses internal state.
            base_pos_limits: Optional dict or list of dicts to override base position limits. Keys: "max_x", "min_x", "max_y", "min_y", "max_z", "min_z".
                           If a dict, uses same limits for all environments. If a list, uses per-environment limits.
                           If None, uses config.base_pos_limits.

        Returns:
            joint_angles: Joint angles in radians, shape (num_envs, num_joints).
            base_pos: Base position in world frame, shape (num_envs, 3).
            base_quat: Base orientation as quaternion [w, x, y, z], shape (num_envs, 4).
        """
        # Validate input shapes
        if target_pos.shape[0] != self.num_envs:
            raise ValueError(f"target_pos batch size {target_pos.shape[0]} does not match num_envs {self.num_envs}")
        if target_quat.shape[0] != self.num_envs:
            raise ValueError(f"target_quat batch size {target_quat.shape[0]} does not match num_envs {self.num_envs}")

        # Convert to numpy for pyroki
        target_pos_np = target_pos.detach().cpu().numpy()
        target_quat_np = target_quat.detach().cpu().numpy()

        # Use previous joint config for warm-start if provided
        if prev_joint_cfg is not None:
            prev_cfg_np = prev_joint_cfg.detach().cpu().numpy()
        else:
            prev_cfg_np = self.current_joint_cfg

        # Use previous base pose for warm-start if provided
        if prev_base_pos is not None:
            prev_base_pos_np = prev_base_pos.detach().cpu().numpy()
        else:
            prev_base_pos_np = self.current_base_pos

        if prev_base_quat is not None:
            prev_base_wxyz_np = prev_base_quat.detach().cpu().numpy()
        else:
            prev_base_wxyz_np = self.current_base_wxyz

        # Solve IK for each environment
        joint_angles_list = []
        base_pos_list = []
        base_quat_list = []

        # Determine base_pos_limits for each environment
        # If base_pos_limits is a list, use per-environment limits; otherwise use same for all
        if isinstance(base_pos_limits, list):
            env_base_pos_limits = base_pos_limits
        else:
            env_base_pos_limits = [base_pos_limits] * self.num_envs

        for env_idx in range(self.num_envs):
            try:
                # Use environment-specific base_pos_limits or fall back to config
                env_limits = (
                    env_base_pos_limits[env_idx]
                    if env_base_pos_limits[env_idx] is not None
                    else self.config.base_pos_limits
                )

                # Use floating base IK solver
                base_pos, base_wxyz, cfg = solve_floating_base_ik(
                    robot=self.robot,
                    target_link_name=self.target_link,
                    target_position=target_pos_np[env_idx],
                    target_wxyz=target_quat_np[env_idx],
                    fix_base_position=self.config.fix_base_position,
                    fix_base_orientation=self.config.fix_base_orientation,
                    prev_pos=prev_base_pos_np[env_idx],
                    prev_wxyz=prev_base_wxyz_np[env_idx],
                    prev_cfg=prev_cfg_np[env_idx],
                    robot_coll=(self.robot_coll if self.config.enable_collision else None),
                    cost_weights=self.config.cost_weights,
                    base_pos_limits=env_limits,
                    arm_joint_names=self.config.arm_joint_names,
                )
                cfg = self._apply_fixed_joint_positions(cfg, list(self.robot.joints.actuated_names))
                joint_angles_list.append(cfg)
                base_pos_list.append(base_pos)
                base_quat_list.append(base_wxyz)
                # Update internal state
                self.current_joint_cfg[env_idx] = cfg
                self.current_base_pos[env_idx] = base_pos
                self.current_base_wxyz[env_idx] = base_wxyz
            except Exception as exc:
                # If IK fails, use previous configuration
                # Log the first and every 100th failure without writing directly to stdout.
                should_log = self._ik_failure_count % 100 == 0
                if env_idx == 0 and should_log:
                    error_msg = str(exc) or f"{type(exc).__name__} (no message)"
                    logger.warning(
                        "IK solver failed for env %d (failure #%d): %s",
                        env_idx,
                        self._ik_failure_count,
                        error_msg,
                        exc_info=self._ik_failure_count == 0,
                    )
                    logger.debug(
                        "IK failure inputs: target_pos=%s target_quat=%s "
                        "prev_base_pos=%s prev_base_quat=%s prev_joint_cfg=%s",
                        target_pos_np[env_idx],
                        target_quat_np[env_idx],
                        prev_base_pos_np[env_idx],
                        prev_base_wxyz_np[env_idx],
                        prev_cfg_np[env_idx],
                    )

                self._ik_failure_count += 1
                joint_angles_list.append(prev_cfg_np[env_idx])
                base_pos_list.append(prev_base_pos_np[env_idx])
                base_quat_list.append(prev_base_wxyz_np[env_idx])

        # Convert back to torch tensors
        joint_angles = torch.tensor(np.array(joint_angles_list), device=self.device, dtype=torch.float32)
        base_pos = torch.tensor(np.array(base_pos_list), device=self.device, dtype=torch.float32)
        base_quat = torch.tensor(np.array(base_quat_list), device=self.device, dtype=torch.float32)
        return joint_angles, base_pos, base_quat

    @property
    def num_joints(self) -> int:
        """Get the number of actuated joints."""
        return self.robot.joints.num_actuated_joints

    @property
    def joint_names(self) -> list[str]:
        """Get the list of actuated joint names."""
        return list(self.robot.joints.actuated_names)


def ik_compute(
    ik_controller: PyrokiIKController,
    env,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    num_envs = ik_controller.num_envs
    device = env.unwrapped.device

    # Check which environments are in their first step (just reset)
    # episode_length_buf == 0 or == 1 indicates a reset environment
    episode_length_buf = env.unwrapped.episode_length_buf
    is_first_step = (episode_length_buf == 0) | (episode_length_buf == 1)

    # Get all actuated joint positions (IK controller expects all joints, not just arm joints)
    # Use the joint names from the IK controller to get the correct joint positions
    all_joint_names = ik_controller.joint_names
    all_joint_indices, _ = env.unwrapped.robot.find_joints(all_joint_names)
    all_joint_indices = torch.tensor(all_joint_indices, dtype=torch.long, device=device)

    # Get current robot state
    current_joint_cfg = env.unwrapped.robot.data.joint_pos[:, all_joint_indices]
    current_base_pos = env.unwrapped.robot.data.body_pos_w[:, env.unwrapped.base_link_idx]
    current_base_quat = env.unwrapped.robot.data.body_quat_w[:, env.unwrapped.base_link_idx]

    # Get previous state from IK controller
    prev_joint_cfg_from_controller = torch.tensor(ik_controller.current_joint_cfg, device=device)
    prev_base_pos_from_controller = torch.tensor(ik_controller.current_base_pos, device=device)
    prev_base_quat_from_controller = torch.tensor(ik_controller.current_base_wxyz, device=device)

    # For environments that just reset, use current pose; otherwise use previous pose from controller
    prev_joint_cfg = torch.where(
        is_first_step.unsqueeze(-1),
        current_joint_cfg,
        prev_joint_cfg_from_controller,
    )
    prev_base_pos = torch.where(
        is_first_step.unsqueeze(-1),
        current_base_pos,
        prev_base_pos_from_controller,
    )
    prev_base_quat = torch.where(
        is_first_step.unsqueeze(-1),
        current_base_quat,
        prev_base_quat_from_controller,
    )

    # Calculate base_pos_limits based on obstacle position to avoid collision for each environment
    safety_margin = ik_controller.config.safety_margin
    base_pos_limits_list = []

    for env_idx in range(num_envs):
        env_base_pos_limits = None

        # Check if wall exists (for peg-in-hole task)
        if hasattr(env.unwrapped, "wall") and env.unwrapped.wall is not None:
            wall_pos = env.unwrapped.wall.data.root_pos_w[env_idx].cpu().numpy()
            wall_size = env.unwrapped.cfg.wall.size  # (0.1, 1.0, 1.0)
            wall_thickness = wall_size[0]  # Thickness is the first dimension

            wall_front_face_x = wall_pos[0] - wall_thickness / 2.0
            max_x_limit = wall_front_face_x - safety_margin

            env_base_pos_limits = {
                "max_x": max_x_limit,
                "min_z": 0.0,
            }
        # Check if pulling_door exists (for cabinet pick and place task)
        elif hasattr(env.unwrapped, "pulling_door") and env.unwrapped.pulling_door is not None:
            # Check if pulling_door exists (for cabinet pick and place task)
            pulling_door_pos = env.unwrapped.pulling_door.data.root_pos_w[env_idx].cpu().numpy()

            cabinet_front_face_x = pulling_door_pos[0]
            max_x_limit = cabinet_front_face_x - (safety_margin + 0.15)

            env_base_pos_limits = {
                "max_x": max_x_limit,
                "min_z": 0.0,
            }
        # Check if door exists (for open door task)
        elif hasattr(env.unwrapped, "door") and env.unwrapped.door is not None:
            door = env.unwrapped.door
            try:
                door_body_indices, door_body_names = door.find_bodies(".*DoorBody.*|.*door.*")
                if len(door_body_indices) > 0:
                    door_body_idx = door_body_indices[0]
                else:
                    # Use first body if DoorBody not found
                    door_body_idx = 0
            except Exception:
                # Fallback to first body
                door_body_idx = 0

            door_body_pos = door.data.body_pos_w[env_idx, door_body_idx].cpu().numpy()

            door_thickness = 0.05
            door_front_face_x = door_body_pos[0] - door_thickness / 2.0
            max_x_limit = door_front_face_x - safety_margin

            env_base_pos_limits = {
                "max_x": max_x_limit,
                "min_z": 0.0,
            }
        elif hasattr(env.unwrapped, "window_object") and env.unwrapped.window_object is not None:
            window_pos = env.unwrapped.window_object.data.root_pos_w[env_idx].cpu().numpy()
            window_thickness = env.unwrapped.cfg.window_thickness

            window_front_face_x = window_pos[0] - window_thickness / 2.0
            max_x_limit = window_front_face_x - safety_margin

            env_base_pos_limits = {
                "max_x": max_x_limit,
                "min_z": 0.0,
            }
        # If neither wall nor pulling_door exists, use default limits
        else:
            env_base_pos_limits = {"min_z": 0.0}

        base_pos_limits_list.append(env_base_pos_limits)

    base_pos_limits = base_pos_limits_list

    # Script-level override: when env has ik_base_pos_max_x_override, force max_x to that value (world x).
    if hasattr(env.unwrapped, "ik_base_pos_max_x_override"):
        override = env.unwrapped.ik_base_pos_max_x_override
        if override is not None:
            for lim in base_pos_limits:
                if lim is not None and isinstance(lim, dict):
                    lim["max_x"] = override

    joint_angles, base_pos, base_quat = ik_controller.compute(
        target_pos=target_pos,
        target_quat=target_quat,
        prev_joint_cfg=prev_joint_cfg,
        prev_base_pos=prev_base_pos,
        prev_base_quat=prev_base_quat,
        base_pos_limits=base_pos_limits,
    )

    # Extract arm joint angles for the manipulator DOFs used by the environment.
    if ik_controller.config.arm_joint_names is not None:
        arm_joint_names = list(ik_controller.config.arm_joint_names)
    else:
        arm_joint_names = [name for name in ik_controller.joint_names if "arm_link" in name and "finger" not in name]
        arm_joint_names = sorted(arm_joint_names)[:4]

    arm_joint_indices_in_ik = [ik_controller.joint_names.index(name) for name in arm_joint_names]
    arm_joint_angles = joint_angles[:, arm_joint_indices_in_ik]
    num_arm_joints = arm_joint_angles.shape[1]

    # [x, y, z, qw, qx, qy, qz, arm joints..., gripper]
    actions = torch.zeros((base_pos.shape[0], 7 + num_arm_joints + 1), device=env.unwrapped.device)
    actions[:, 0:3] = base_pos
    actions[:, 3:7] = base_quat
    actions[:, 7 : 7 + num_arm_joints] = arm_joint_angles
    actions[:, 7 + num_arm_joints] = 0.0  # gripper open

    return actions
