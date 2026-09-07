# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from omni.usd import get_context
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from ambench.robots.visuals.propeller_cfg import PropellerVizCfg


class PropAnimation:
    """Visual-only propeller animation helper driven by motor thrust."""

    def __init__(
        self,
        *,
        robot_prim_path: str,
        propeller_viz_cfg: PropellerVizCfg | None,
        spin_directions: torch.Tensor | None,
        reference_thrust: float | None = None,
        num_envs: int,
        device: str | torch.device,
    ):
        self._device = torch.device(device) if isinstance(device, str) else device
        self._num_envs = num_envs
        self._robot_prim_path = robot_prim_path

        self._cfg: PropellerVizCfg | None = None
        self._reference_thrust = 1.0
        self._prim_exprs: list[str] = []
        self._views: list[sim_utils.XformPrimView] = []
        self._angles: torch.Tensor | None = None
        self._zero_translations: torch.Tensor | None = None
        self._base_orientations: torch.Tensor | None = None
        self._axis: torch.Tensor | None = None
        self._spin_dirs: torch.Tensor | None = None
        self._last_angular_speed: torch.Tensor | None = None
        self._usd_orient_ops: list[list[Any]] = []

        if propeller_viz_cfg is None:
            return

        prop_cfg = propeller_viz_cfg
        if not prop_cfg.enabled or len(prop_cfg.relative_prim_paths) == 0:
            return
        if spin_directions is None:
            raise ValueError("Propeller visualization requires resolved rotor spin directions.")
        if spin_directions.numel() != len(prop_cfg.relative_prim_paths):
            raise ValueError(
                "Propeller visualization requires one prim path per rotor, "
                f"got {len(prop_cfg.relative_prim_paths)} paths and {spin_directions.numel()} rotors."
            )

        self._cfg = prop_cfg
        self._reference_thrust = max(reference_thrust or 1.0, 1e-6)
        self._angles = torch.zeros((num_envs, len(prop_cfg.relative_prim_paths)), device=self._device)
        self._zero_translations = torch.zeros((num_envs, 3), device=self._device)
        self._base_orientations = torch.zeros((num_envs, 4), device=self._device)
        self._base_orientations[:, 0] = 1.0
        self._axis = torch.tensor(prop_cfg.local_axis, device=self._device, dtype=torch.float32).unsqueeze(0)
        self._spin_dirs = spin_directions.to(device=self._device, dtype=torch.float32).unsqueeze(0)
        self._prim_exprs = [f"{robot_prim_path}/{relative_path}" for relative_path in prop_cfg.relative_prim_paths]

        self.refresh_views()

    @property
    def enabled(self) -> bool:
        return self._cfg is not None

    @property
    def last_angular_speed(self) -> torch.Tensor | None:
        return self._last_angular_speed

    def resolve_angular_speed(self, motor_thrusts: torch.Tensor) -> torch.Tensor | None:
        if self._cfg is None or self._spin_dirs is None:
            return None

        thrust_norm = torch.clamp(motor_thrusts, min=0.0) / self._reference_thrust
        angular_speed = torch.sqrt(torch.clamp(thrust_norm, max=1.0)) * self._cfg.max_angular_speed
        return angular_speed * self._spin_dirs

    def refresh_views(self) -> None:
        """Refresh propeller prim views after the stage is fully populated."""

        if self._cfg is None:
            return

        self._views = [
            sim_utils.XformPrimView(prim_expr, device=self._device, validate_xform_ops=False)
            for prim_expr in self._prim_exprs
        ]
        self._refresh_usd_orient_ops()

    def _refresh_usd_orient_ops(self) -> None:
        """Collect concrete USD orient ops for referenced visual prims that tensor views cannot resolve."""

        if self._cfg is None:
            return

        stage = get_context().get_stage()
        if stage is None:
            return

        self._usd_orient_ops = []
        for relative_path in self._cfg.relative_prim_paths:
            prop_ops = []
            for env_idx in range(self._num_envs):
                prim_path = self._concrete_prim_path(relative_path, env_idx)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                xformable = UsdGeom.Xformable(prim)
                orient_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        orient_op = op
                        break
                if orient_op is None:
                    orient_op = xformable.AddOrientOp()
                prop_ops.append(orient_op)
            self._usd_orient_ops.append(prop_ops)

    def _concrete_prim_path(self, relative_path: str, env_idx: int) -> str:
        return f"{self._robot_prim_path}/{relative_path}".replace("env_.*", f"env_{env_idx}")

    def _views_ready(self) -> bool:
        return len(self._views) > 0 and all(view.count == self._num_envs for view in self._views)

    def _usd_ops_ready(self) -> bool:
        return len(self._usd_orient_ops) == len(self._prim_exprs) and all(
            len(prop_ops) == self._num_envs for prop_ops in self._usd_orient_ops
        )

    def _set_usd_orientations(self, orientations: torch.Tensor) -> None:
        orientations_cpu = orientations.detach().cpu()
        for prop_idx, prop_ops in enumerate(self._usd_orient_ops):
            for env_idx, orient_op in enumerate(prop_ops):
                quat = orientations_cpu[env_idx, prop_idx]
                orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))

    def update(self, motor_thrusts: torch.Tensor, dt: float) -> None:
        """Update propeller visual transforms from the latest motor thrusts."""

        if self._cfg is None or self._angles is None or self._axis is None or self._spin_dirs is None:
            return

        if not self._views_ready() and not self._usd_ops_ready():
            self.refresh_views()
            if not self._views_ready() and not self._usd_ops_ready():
                return

        angular_speed = self.resolve_angular_speed(motor_thrusts)
        if angular_speed is None:
            return
        self._last_angular_speed = angular_speed
        self._angles = torch.remainder(self._angles + angular_speed * dt, 2.0 * torch.pi)

        repeated_axis = self._axis.repeat(self._num_envs, 1)
        orientations_by_prop = []
        for prop_idx in range(self._angles.shape[1]):
            delta_quat = math_utils.quat_from_angle_axis(self._angles[:, prop_idx], repeated_axis)
            orientations_by_prop.append(math_utils.quat_mul(self._base_orientations, delta_quat))

        if self._usd_ops_ready():
            self._set_usd_orientations(torch.stack(orientations_by_prop, dim=1))
        elif self._views_ready():
            for prop_idx, view in enumerate(self._views):
                view.set_local_poses(
                    translations=self._zero_translations,
                    orientations=orientations_by_prop[prop_idx],
                )

    def reset(self, env_ids: torch.Tensor | list[int] | tuple[int, ...]) -> None:
        """Reset animated propeller state for the selected environments."""

        if self._angles is None or self._base_orientations is None or self._zero_translations is None:
            return

        self._angles[env_ids] = 0.0
        identity = self._base_orientations[env_ids]
        zero_translation = self._zero_translations[env_ids]
        if self._usd_ops_ready():
            env_id_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            for prop_ops in self._usd_orient_ops:
                for env_idx in env_id_list:
                    prop_ops[env_idx].Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        elif self._views_ready():
            for view in self._views:
                view.set_local_poses(translations=zero_translation, orientations=identity, indices=env_ids)
