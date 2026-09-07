# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Floating-base inverse kinematics using Pyroki and JAX."""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_floating_base_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_position: onp.ndarray,
    target_wxyz: onp.ndarray,
    fix_base_position: tuple[bool, bool, bool],
    fix_base_orientation: tuple[bool, bool, bool],
    prev_pos: onp.ndarray,
    prev_wxyz: onp.ndarray,
    prev_cfg: onp.ndarray,
    robot_coll: pk.collision.RobotCollision | None = None,
    cost_weights: dict[str, float] | None = None,
    base_pos_limits: dict[str, float] | None = None,
    arm_joint_names: list[str] | None = None,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Solve inverse kinematics for a robot with a mobile base.

    Args:
        robot: PyRoKi Robot.
        target_link_name: Name of the target link.
        target_position: Target position with shape ``(3,)``.
        target_wxyz: Target orientation with shape ``(4,)`` in WXYZ order.
        fix_base_position: Whether to fix the base position (x, y, z).
        fix_base_orientation: Whether to fix the base orientation
            (roll, pitch, yaw).
        prev_pos, prev_wxyz, prev_cfg: Previous base position, orientation,
            and joint configuration, for smooth motion.
        robot_coll: Optional RobotCollision object for self-collision
            avoidance.
        cost_weights: Optional dict to customize cost weights.
            Available keys:
            - "pose_pos": weight for position matching (default: 5.0)
            - "pose_ori": weight for orientation matching (default: 1.0)
            - "joint_limit": weight for joint limits (default: 100.0)
            - "smoothness_joints": weight for joint smoothness cost
                (default: 0.1)
            - "smoothness_base_pos": weight for base translation smoothness
                (default: 0.1)
            - "smoothness_base_ori": weight for base rotation smoothness
                (default: 0.1)
            - "self_collision": weight for self-collision avoidance
                (default: 5.0)
            - "base_pos_limit": weight for base position limit violations
                (default: 100.0)
            - "rest_arm": weight for rest cost pulling arm joints toward the
                robot default (initial) joint configuration (default: 0.1)
        base_pos_limits: Optional dict to set base position limits.
            Available keys: "max_x", "min_x", "max_y", "min_y",
            "max_z", "min_z"
            Example: {"max_x": 2.0, "min_z": 0.0} constrains
                x <= 2.0 and z >= 0.0
        arm_joint_names: Optional list of joint names for the robotic arm.
            If not provided, defaults to first 3 actuated joints.
            Rest cost will only be applied to these joints.

    Returns:
        base_pos: onp.ndarray. Shape: (3,).
        base_wxyz: onp.ndarray. Shape: (4,).
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    if target_position.shape != (3,) or target_wxyz.shape != (4,):
        raise ValueError(
            f"Expected target_position (3,) and target_wxyz (4,), got {target_position.shape} and {target_wxyz.shape}."
        )
    if prev_pos.shape != (3,) or prev_wxyz.shape != (4,):
        raise ValueError(f"Expected prev_pos (3,) and prev_wxyz (4,), got {prev_pos.shape} and {prev_wxyz.shape}.")
    expected_cfg_shape = (robot.joints.num_actuated_joints,)
    if prev_cfg.shape != expected_cfg_shape:
        raise ValueError(f"Expected prev_cfg {expected_cfg_shape}, got {prev_cfg.shape}.")
    target_link_idx = robot.links.names.index(target_link_name)

    T_world_targets = jaxlie.SE3(jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1))
    # Determine arm joint indices
    if arm_joint_names is None:
        # Default to first 3 actuated joints (assuming they're the arm)
        num_arm_joints = min(3, robot.joints.num_actuated_joints)
        arm_joint_indices = list(range(num_arm_joints))
    else:
        arm_joint_indices = [robot.joints.actuated_names.index(name) for name in arm_joint_names]

    # Convert to array (always pass valid array, empty if no arm joints)
    if arm_joint_indices:
        arm_joint_indices_array = jnp.array(arm_joint_indices)
    else:
        arm_joint_indices_array = jnp.array([])

    base_pose, cfg = _solve_ik_jax(
        robot,
        T_world_targets,
        jnp.array(target_link_idx),
        jnp.array(fix_base_position + fix_base_orientation),
        jnp.array(prev_pos),
        jnp.array(prev_wxyz),
        jnp.array(prev_cfg),
        robot_coll,
        cost_weights,
        base_pos_limits,
        arm_joint_indices_array,
    )
    if cfg.shape != expected_cfg_shape:
        raise RuntimeError(f"IK solver returned joint configuration {cfg.shape}, expected {expected_cfg_shape}.")

    base_pos = base_pose.translation()
    base_wxyz = base_pose.rotation().wxyz
    if base_pos.shape != (3,) or base_wxyz.shape != (4,):
        raise RuntimeError(f"IK solver returned base pose shapes {base_pos.shape} and {base_wxyz.shape}.")

    return onp.array(base_pos), onp.array(base_wxyz), onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    T_world_target: jaxlie.SE3,
    target_joint_indices: jnp.ndarray,
    fix_base: jnp.ndarray,
    prev_pos: jnp.ndarray,
    prev_wxyz: jnp.ndarray,
    prev_cfg: jnp.ndarray,
    robot_coll: pk.collision.RobotCollision | None = None,
    cost_weights: dict[str, float] | None = None,
    base_pos_limits: dict[str, float] | None = None,
    arm_joint_indices: jnp.ndarray | None = None,
) -> tuple[jaxlie.SE3, jax.Array]:
    joint_var = robot.joint_var_cls(0)
    default_joint_cfg = jnp.array(joint_var.default_factory())

    def retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, removing updates on axes."""
        delta = delta * (1 - fix_base)
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(prev_wxyz),
            prev_pos,
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=retract_fn,
    ): ...

    base_var = ConstrainedSE3Var(0)

    # Default cost weights
    default_weights = {
        "pose_pos": 5.0,
        "pose_ori": 1.0,
        "joint_limit": 100.0,
        "smoothness_joints": 0.1,
        "smoothness_base_pos": 0.1,
        "smoothness_base_ori": 0.1,
        "self_collision": 5.0,
        "base_pos_limit": 100.0,
        "rest_arm": 0.1,
    }
    # Override with user-provided weights
    if cost_weights is not None:
        default_weights.update(cost_weights)
    weights = default_weights

    # Create previous base pose constant
    T_world_base_prev = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(prev_wxyz),
        prev_pos,
    )

    # Base position limit cost
    @jaxls.Cost.create_factory
    def base_position_limit_cost(
        vals: jaxls.VarValues,
        base_var: jaxls.Var[jaxlie.SE3],
        limits: dict[str, float],
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Penalizes base position limit violations.
        Limits can be: max_x, min_x, max_y, min_y, max_z, min_z
        """
        T_world_base = vals[base_var]
        base_pos = T_world_base.translation()
        residuals = []

        # Check each limit
        if "max_x" in limits:
            residuals.append(jnp.maximum(0.0, base_pos[0] - limits["max_x"]))
        if "min_x" in limits:
            residuals.append(jnp.maximum(0.0, limits["min_x"] - base_pos[0]))
        if "max_y" in limits:
            residuals.append(jnp.maximum(0.0, base_pos[1] - limits["max_y"]))
        if "min_y" in limits:
            residuals.append(jnp.maximum(0.0, limits["min_y"] - base_pos[1]))
        if "max_z" in limits:
            residuals.append(jnp.maximum(0.0, base_pos[2] - limits["max_z"]))
        if "min_z" in limits:
            residuals.append(jnp.maximum(0.0, limits["min_z"] - base_pos[2]))

        if len(residuals) == 0:
            return jnp.array([])
        residual = jnp.stack(residuals)
        return (residual * weight).flatten()

    # Rest cost for arm joints only
    @jaxls.Cost.create_factory
    def rest_cost_arm(
        vals: jaxls.VarValues,
        joint_var: jaxls.Var[jnp.ndarray],
        arm_joint_indices: jnp.ndarray,
        rest_pose: jnp.ndarray,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes rest cost only for specified arm joints.
        Other joints are ignored (zero residual).
        """
        cfg = vals[joint_var]
        # Create mask: 1.0 for arm joints, 0.0 for others
        mask = jnp.zeros(cfg.shape[0])
        mask = mask.at[arm_joint_indices].set(1.0)
        # Compute residual only for arm joints
        residual = (cfg - rest_pose) * mask
        return (residual * weight).flatten()

    factors = [
        pk.costs.pose_cost_with_base(
            robot,
            joint_var,
            base_var,
            T_world_target,
            target_joint_indices,
            pos_weight=jnp.array(weights["pose_pos"]),
            ori_weight=jnp.array(weights["pose_ori"]),
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            jnp.array(weights["joint_limit"]),
        ),
        pk.costs.smoothness_cost_constant(
            joint_var,
            prev_cfg_const=prev_cfg,
            weight=jnp.array([weights["smoothness_joints"]]),
        ),
        pk.costs.smoothness_cost_se3_const(
            base_var,
            past_pose_const=T_world_base_prev,
            pos_weight=jnp.array(weights["smoothness_base_pos"]),
            ori_weight=jnp.array(weights.get("smoothness_base_ori", 0.1)),
        ),
    ]

    # Pull arm joints toward the URDF default (initial) configuration, not prev_cfg.
    if arm_joint_indices.size > 0:
        rest_pose = default_joint_cfg
        factors.append(
            rest_cost_arm(
                joint_var,
                arm_joint_indices=arm_joint_indices,
                rest_pose=rest_pose,
                weight=jnp.array(weights["rest_arm"]),
            )
        )

    if base_pos_limits:
        factors.append(
            base_position_limit_cost(
                base_var,
                limits=base_pos_limits,
                weight=jnp.array(weights["base_pos_limit"]),
            )
        )

    if robot_coll is not None:
        factors.append(
            pk.costs.self_collision_cost(
                robot,
                robot_coll=robot_coll,
                joint_var=joint_var,
                margin=0.02,
                weight=jnp.array(weights["self_collision"]),
            )
        )

    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var, base_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([
                joint_var.with_value(prev_cfg),
                base_var,
            ]),
            verbose=False,
        )
    )
    return sol[base_var], sol[joint_var]
