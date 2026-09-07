# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Task-independent direct environment for composed robot/control profiles."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.usd import get_context
from pxr import Gf, UsdGeom

from ambench.controllers.control_pipeline import ControlPipeline
from ambench.controllers.controller_cfg import BaseController, ControllerOutput
from ambench.robots.robot_io import RobotCommand, RobotIO
from ambench.tasks.base.base_env_cfg import BaseEnvCfg

logger = logging.getLogger(__name__)


class BaseEnv(DirectRLEnv):
    """Shared simulation lifecycle around one robot profile and control pipeline."""

    cfg: BaseEnvCfg

    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.enable_aerodynamic_effects:
            cfg.sim.enable_scene_query_support = True
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.robot_io = RobotIO(
            robot=self.robot,
            spec=self.cfg.robot_profile.robot,
            num_envs=self.num_envs,
            device=self.device,
            dt=self.dt,
        )
        self._init_ids()
        pipeline_cfg = self.cfg.robot_profile.control
        self.control_pipeline: ControlPipeline = pipeline_cfg.class_type(self, pipeline_cfg)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        logger.info(
            "Robot/control profile: %s / %s / %s",
            self.cfg.robot_profile.robot.robot_id,
            pipeline_cfg.class_type.__name__,
            pipeline_cfg.controller.class_type.__name__,
        )

    @property
    def controller(self) -> BaseController:
        return self.control_pipeline.controller

    @property
    def control_output(self) -> ControllerOutput | None:
        return self.control_pipeline.last_output

    @property
    def ee_cmd_pos_w(self) -> torch.Tensor:
        return self.control_pipeline.ee_cmd_pos_w

    @property
    def ee_cmd_quat_w(self) -> torch.Tensor:
        return self.control_pipeline.ee_cmd_quat_w

    @property
    def is_first_step(self) -> torch.Tensor:
        return self.control_pipeline.is_first_step

    @property
    def base_link_idx(self) -> int:
        return self.robot_io.base_link_idx

    @property
    def ee_link_idx(self) -> int | None:
        return self.robot_io.ee_link_idx

    @property
    def arm_joint_ids(self) -> list[int]:
        return self.robot_io.arm_joint_ids

    @property
    def gripper_joint_ids(self) -> list[int]:
        return self.robot_io.gripper_joint_ids

    @property
    def motor_arm_joint_ids(self) -> list[int]:
        return self.robot_io.motor_arm_joint_ids

    @property
    def arm_targets(self) -> torch.Tensor:
        return self.robot_io.arm_targets

    @property
    def gripper_targets(self) -> torch.Tensor:
        return self.robot_io.gripper_targets

    def _init_ids(self) -> None:
        """Resolve task-owned body and joint IDs after RobotIO is ready."""

    def _get_success(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return final task success and optional binary subtask criteria."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), {}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check task success, record criteria, and apply the episode timeout."""
        terminated, criteria = self._get_success()
        if criteria:
            self.extras["success_criteria"] = {
                name: value.detach().clone().to(dtype=torch.bool) for name, value in criteria.items()
            }
        else:
            self.extras.pop("success_criteria", None)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _setup_scene(self) -> None:
        """Spawn task-independent robot, cameras, ground, clones, and lighting."""
        profile = self.cfg.robot_profile
        self.robot = Articulation(profile.robot.asset)
        self.ee_camera = Camera(profile.ee_camera) if profile.ee_camera is not None else None
        self.base_camera = Camera(profile.base_camera) if profile.base_camera is not None else None
        self.scene_camera = Camera(self.cfg.scene_camera_cfg) if self.cfg.scene_camera_cfg is not None else None

        if self.cfg.spawn_ground_plane:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(1.0, 1.0, 1.0)))

        self.scene.clone_environments(copy_from_source=False)
        self._apply_y_axis_env_origin_override()
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        if self.ee_camera is not None:
            self.scene.sensors["ee_camera"] = self.ee_camera
        if self.base_camera is not None:
            self.scene.sensors["base_camera"] = self.base_camera
        if self.scene_camera is not None:
            self.scene.sensors["scene_camera"] = self.scene_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _grasp_rigid_object_at_ee(
        self,
        rigid_object: RigidObject,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        grasp_width: float | None = None,
    ) -> None:
        """Place an object in the configured tool frame and close the gripper."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        gripper = self.cfg.robot_profile.robot.gripper
        width_converter = None if gripper is None else gripper.joint_position_from_object_width
        if grasp_width is not None and width_converter is not None:
            limits = self.robot.data.soft_joint_pos_limits[0, self.gripper_joint_ids, :]
            grasp_joint_pos = torch.full(
                (len(env_ids), len(self.gripper_joint_ids)),
                float(width_converter(grasp_width)),
                device=self.device,
            )
            grasp_joint_pos = torch.clamp(grasp_joint_pos, min=limits[:, 0], max=limits[:, 1])
            joint_pos = self.robot.data.joint_pos[env_ids].clone()
            joint_vel = self.robot.data.joint_vel[env_ids].clone()
            joint_pos[:, self.gripper_joint_ids] = grasp_joint_pos
            joint_vel[:, self.gripper_joint_ids] = 0.0
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        close_action = torch.full((len(env_ids),), -1.0, device=self.device)
        self.robot_io.gripper_targets[env_ids] = self.robot_io.compute_gripper_targets(close_action)

        if self.ee_link_idx is None:
            raise RuntimeError("Cannot place an object without an end-effector body.")
        ee_state = self.robot.data.body_link_state_w[:, self.ee_link_idx, :]
        root_state = rigid_object.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.robot_io.compute_tool_tip_pos_w(ee_state[env_ids])
        root_state[:, 3:7] = self.robot_io.link_to_command_quat(ee_state[env_ids, 3:7])
        root_state[:, 7:] = 0.0
        rigid_object.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _apply_y_axis_env_origin_override(self) -> None:
        """Relayout cloned environments along the Y axis before simulation starts."""
        if self.num_envs <= 1:
            return

        env_spacing = self.cfg.scene.env_spacing
        origins_y = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        y_offset = -(self.num_envs - 1) * env_spacing / 2.0
        for env_idx in range(self.num_envs):
            origins_y[env_idx, 1] = y_offset + env_idx * env_spacing
        self.scene._default_env_origins = origins_y

        stage = get_context().get_stage()
        for env_idx, env_path in enumerate(self.scene.env_prim_paths):
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(env_path))
            translate_op = next(
                (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
                None,
            )
            if translate_op is None:
                translate_op = xform.AddTranslateOp()
            position = origins_y[env_idx].detach().cpu().tolist()
            translate_op.Set(Gf.Vec3d(*position))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert one public action batch into a complete robot command."""
        self.actions = actions.clone()
        self._robot_command = self.control_pipeline.process(self.actions)

    def _apply_action(self) -> None:
        """Apply the command prepared by the selected control pipeline."""
        command: RobotCommand = self._robot_command
        self.robot_io.apply(command)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset simulator state, robot command buffers, and the control pipeline."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)  # type: ignore[arg-type]
        env_ids_tensor = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.tensor(env_ids, device=self.device, dtype=torch.long)
        )

        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor]
        joint_vel = self.robot.data.default_joint_vel[env_ids_tensor]
        default_root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]

        self.robot_io.reset(env_ids_tensor, joint_pos)
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_tensor)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)
        self.control_pipeline.reset(default_root_state, env_ids_tensor)

    def step(self, action: torch.Tensor):
        """Apply observation noise once to list or tensor policy observations."""
        original_obs_noise = self.cfg.observation_noise_model
        if original_obs_noise is not None:
            self.cfg.observation_noise_model = None

        try:
            obs_buf, reward_buf, terminated, truncated, extras = super().step(action)
        finally:
            self.cfg.observation_noise_model = original_obs_noise

        if self.cfg.enable_observation_noise and original_obs_noise is not None:
            policy_obs = obs_buf.get("policy")
            if isinstance(policy_obs, list) and policy_obs:
                obs_list = policy_obs
                keys = list(obs_list[0].keys())
                obs_tensors = [torch.stack([env_obs[key] for env_obs in obs_list]) for key in keys]
                obs_tensor_noisy = self._observation_noise_model(torch.cat(obs_tensors, dim=-1))
                obs_splits = torch.split(obs_tensor_noisy, [tensor.shape[-1] for tensor in obs_tensors], dim=-1)
                obs_buf["policy"] = [
                    {key: obs_splits[key_idx][env_idx] for key_idx, key in enumerate(keys)}
                    for env_idx in range(len(obs_list))
                ]
            elif isinstance(policy_obs, torch.Tensor):
                obs_buf["policy"] = self._observation_noise_model(policy_obs)

        return obs_buf, reward_buf, terminated, truncated, extras
