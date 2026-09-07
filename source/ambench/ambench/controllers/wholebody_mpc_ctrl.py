# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import casadi as cs
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from isaaclab.assets import Articulation
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as R

from ambench.controllers.controller_cfg import (
    BaseController,
)
from ambench.controllers.utils.control_utils import get_aggregate_physical_parameters
from ambench.controllers.utils.mpc_utils import (
    quaternion_from_rotation_matrix,
    quaternion_from_rotation_matrix_ca,
    rotation_matrix_from_euler_ca,
    rpy_to_rotation_matrix,
)

DEGREE_TO_RADIAN = np.pi / 180.0


def quaternion_to_rpy_ca(quat):
    # quat: [w, x, y, z]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = cs.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = cs.asin(2 * (w * y - z * x))
    yaw = cs.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return cs.vertcat(roll, pitch, yaw)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class DroneMPC4DOF:
    def __init__(
        self,
        mass,
        T,
        N,
        Q,
        R,
        R_delta,
        pos_min,
        pos_max,
        joint_min,
        joint_max,
        default_arm_angle,
        moment_of_inertia,
        compile_dir="./acados_mpc_build",
        model_name="aerial_manipulator",
    ):
        self.mass = mass
        self.T = T
        self.N = N
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.R_delta = np.diag(R_delta)
        self.pos_min = np.array(pos_min)
        self.pos_max = np.array(pos_max)
        self.joint_min = np.array(joint_min) * DEGREE_TO_RADIAN
        self.joint_max = np.array(joint_max) * DEGREE_TO_RADIAN
        self.default_arm_angle = np.array(default_arm_angle) * DEGREE_TO_RADIAN
        self.arm_delay_gain = 12.0
        self.compile_dir = compile_dir
        self.model_name = model_name

        self.I_x = moment_of_inertia[0]
        self.I_y = moment_of_inertia[1]
        self.I_z = moment_of_inertia[2]

        # State vector: [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)] - 16 states
        self.x = cs.MX.sym("x", 16)
        self.p = self.x[0:3]
        self.v = self.x[3:6]
        self.base_euler = self.x[6:9]
        self.base_ang_vel = self.x[9:12]
        self.arm_angle = self.x[12:16]

        # Control vector: [acc(3), torques(3), arm_angle_ref(4)] - 10 controls
        self.u = cs.MX.sym("u", 10)
        self.u_prev = cs.MX.sym("u_prev", 10)

        # Parameters for reference trajectory
        self.ee_pos_ref = cs.MX.sym("ee_pos_ref", 3)
        self.ee_quat_ref = cs.MX.sym("ee_quat_ref", 4)
        self.arm_angle_ref = cs.MX.sym("arm_angle_ref", 4)
        self.param = cs.vertcat(self.ee_pos_ref, self.ee_quat_ref, self.arm_angle_ref, self.u_prev)

        self.x_dot = self.build_dynamics(self.x, self.u)
        self.model = self.build_acados_model()
        self.ocp_solver = self.build_acados_ocp_solver()

        # Post-solve EE tracking cost/gradients (not part of the OCP objective).
        self._build_tracking_helpers()

    def build_dynamics(self, x, u):
        v = x[3:6]
        base_ang_vel = x[9:12]
        arm_angle = x[12:16]

        force = u[0:3]
        torques = u[3:6]
        arm_angle_ref = u[6:10]

        dp = v
        dv = force / self.mass

        # Small-angle base: d(euler)/dt ~= omega_body.
        d_base_euler = base_ang_vel
        d_base_ang_vel = cs.vertcat(torques[0] / self.I_x, torques[1] / self.I_y, torques[2] / self.I_z)

        darm_angle = (arm_angle_ref - arm_angle) * self.arm_delay_gain
        return cs.vertcat(dp, dv, d_base_euler, d_base_ang_vel, darm_angle)

    def arm_forward_kinematics(self, p, arm_angle, base_euler):
        """CasADi FK for the arm chain. Returns [ee_pos(3), ee_quat_wxyz(4)]."""
        q1 = arm_angle[0]
        q2 = arm_angle[1]
        q3 = arm_angle[2]
        q4 = arm_angle[3]

        Rb = rotation_matrix_from_euler_ca(base_euler)

        c1, s1 = cs.cos(q1), cs.sin(q1)
        R1 = cs.vertcat(
            cs.horzcat(c1, 0, s1),
            cs.horzcat(0, 1, 0),
            cs.horzcat(-s1, 0, c1),
        )

        # joint2 URDF axis (0, -1, 0) -> Ry(-q2)
        c2, s2 = cs.cos(-q2), cs.sin(-q2)
        R2 = cs.vertcat(
            cs.horzcat(c2, 0, s2),
            cs.horzcat(0, 1, 0),
            cs.horzcat(-s2, 0, c2),
        )

        c3, s3 = cs.cos(q3), cs.sin(q3)
        R3 = cs.vertcat(
            cs.horzcat(c3, 0, s3),
            cs.horzcat(0, 1, 0),
            cs.horzcat(-s3, 0, c3),
        )

        c4, s4 = cs.cos(q4), cs.sin(q4)
        R4 = cs.vertcat(
            cs.horzcat(1, 0, 0),
            cs.horzcat(0, c4, -s4),
            cs.horzcat(0, s4, c4),
        )

        # URDF chain offsets (FA-Hexa arm).
        p_arm_base = cs.vertcat(0.088, 0.0, 0.0)
        p_j1 = p_arm_base + cs.vertcat(0.0, 0.0, 0.06475)
        p_j2_local = cs.vertcat(-0.3795, 0.0, 0.059)
        p_j2 = p_j1 + cs.mtimes(R1, p_j2_local)
        R12 = cs.mtimes(R1, R2)
        p_j3_local = cs.vertcat(0.4475, 0.0, 0.0)
        p_j3 = p_j2 + cs.mtimes(R12, p_j3_local)
        R123 = cs.mtimes(R12, R3)
        p_j4_local = cs.vertcat(0.071, 0.0, 0.0)
        p_j4 = p_j3 + cs.mtimes(R123, p_j4_local)
        R_end = cs.mtimes(R123, R4)
        p_ee_local = cs.vertcat(0.0, 0.0, 0.0)
        p_ee = p_j4 + cs.mtimes(R_end, p_ee_local)

        ee_pos_world = cs.mtimes(Rb, p_ee) + p
        ee_R_world = cs.mtimes(Rb, R_end)

        ee_quat_wxyz = quaternion_from_rotation_matrix_ca(ee_R_world)
        return cs.vertcat(ee_pos_world, ee_quat_wxyz)

    def build_cost_expr(self):
        base_euler_state = self.base_euler
        ee_state = self.arm_forward_kinematics(self.p, self.arm_angle, base_euler_state)
        ee_pos, ee_quat = ee_state[0:3], ee_state[3:7]
        ee_quat_ref = self.ee_quat_ref
        ee_euler_ref = quaternion_to_rpy_ca(ee_quat_ref)
        ee_euler = quaternion_to_rpy_ca(ee_quat)
        ee_roll_error = ee_euler[0] - ee_euler_ref[0]
        ee_pitch_error = ee_euler[1] - ee_euler_ref[1]
        ee_yaw_error = ee_euler[2] - ee_euler_ref[2]
        ee_pos_error = ee_pos - self.ee_pos_ref
        cost_expr_y = cs.vertcat(
            ee_pos_error,
            ee_roll_error,
            ee_pitch_error,
            ee_yaw_error,
            self.v,
            self.base_euler,
            self.base_ang_vel,
            self.arm_angle,
            self.u,
            self.u - self.u_prev,
        )
        cost_expr_y_e = cs.vertcat(
            ee_pos_error,
            ee_roll_error,
            ee_pitch_error,
            ee_yaw_error,
            self.v,
            self.base_euler,
            self.base_ang_vel,
            self.arm_angle,
        )
        return cost_expr_y, cost_expr_y_e

    def build_acados_model(self):
        x_dot = cs.MX.sym("x_dot", 16)
        f_expl = self.x_dot
        f_impl = x_dot - f_expl
        model = AcadosModel()

        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl

        cost_y_expr, cost_y_expr_e = self.build_cost_expr()
        self.cost_y_expr = cost_y_expr
        self.cost_y_expr_e = cost_y_expr_e

        model.cost_y_expr = cost_y_expr
        model.cost_y_expr_e = cost_y_expr_e

        model.x = self.x
        model.xdot = x_dot
        model.u = self.u
        model.p = self.param
        model.name = self.model_name
        return model

    def build_acados_ocp_solver(self):
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.N
        ocp.dims.np = self.param.size()[0]
        ocp.parameter_values = np.zeros(ocp.dims.np)
        ocp.solver_options.tf = self.T
        nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.model.cost_y_expr.size()[0]
        ny_e = self.model.cost_y_expr_e.size()[0]
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        W = block_diag(self.Q, self.R, self.R_delta)
        ocp.cost.W = W

        # Terminal cost
        Q_terminal = self.Q[:19, :19]
        ocp.cost.W_e = 1.0 * Q_terminal

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        euler_min = np.array([-0.01, -0.01, -np.pi], dtype=np.float64)
        euler_max = np.array([0.01, 0.01, np.pi], dtype=np.float64)
        x_min = np.concatenate([self.pos_min, euler_min, self.joint_min]).astype(np.float64)
        x_max = np.concatenate([self.pos_max, euler_max, self.joint_max]).astype(np.float64)

        idxbx = np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 15], dtype=np.int64)

        ocp.constraints.idxbx = idxbx
        ocp.constraints.lbx = x_min
        ocp.constraints.ubx = x_max
        ocp.dims.nbx = int(idxbx.size)

        idxbx_0 = np.arange(nx, dtype=np.int64)
        ocp.constraints.idxbx_0 = idxbx_0
        ocp.constraints.lbx_0 = np.zeros(nx, dtype=np.float64)
        ocp.constraints.ubx_0 = np.zeros(nx, dtype=np.float64)
        ocp.dims.nbx_0 = int(idxbx_0.size)

        ocp.constraints.idxbx_e = idxbx
        ocp.constraints.lbx_e = x_min
        ocp.constraints.ubx_e = x_max
        ocp.dims.nbx_e = int(idxbx.size)

        idxbu = np.array([6, 7, 8, 9], dtype=np.int64)
        ocp.constraints.idxbu = idxbu
        ocp.constraints.lbu = self.joint_min.astype(np.float64)
        ocp.constraints.ubu = self.joint_max.astype(np.float64)
        ocp.dims.nbu = int(idxbu.size)

        # Path constraints (EE height above base)
        base_euler_state = self.base_euler
        ee_state = self.arm_forward_kinematics(self.p, self.arm_angle, base_euler_state)
        ee_pos = ee_state[0:3]
        h_expr = cs.vertcat((ee_pos - self.p)[2])

        ocp.model.con_h_expr = h_expr

        ocp.constraints.lh = np.array([0.1])
        ocp.constraints.uh = np.array([2.5])

        ocp.constraints.x0 = np.zeros(nx)
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"

        ocp.solver_options.qp_solver_iter_max = 2000
        ocp.solver_options.nlp_solver_max_iter = 200
        ocp.solver_options.qp_solver_tol_stat = 1e-3
        ocp.solver_options.qp_solver_tol_eq = 1e-3
        ocp.solver_options.qp_solver_tol_ineq = 1e-3
        ocp.solver_options.qp_solver_tol_comp = 1e-3
        ocp.solver_options.qp_solver_warm_start = 0

        ocp.solver_options.compile_dir = self.compile_dir

        rebuild = True
        return AcadosOcpSolver(ocp, generate=rebuild, build=rebuild)

    def set_reference_sequence(self, ee_pos_refs, ee_quat_refs, arm_angle, u_prev):
        self._ee_pos_refs = ee_pos_refs.copy()
        self._ee_quat_refs = ee_quat_refs.copy()

        default_arm_angle = np.array([
            self.default_arm_angle[0],
            self.default_arm_angle[1],
            self.default_arm_angle[2],
            self.default_arm_angle[3],
        ])

        for i in range(self.N):
            # ee_quat_refs are wxyz; scipy Rotation expects xyzw.
            r = R.from_quat([ee_quat_refs[i][1], ee_quat_refs[i][2], ee_quat_refs[i][3], ee_quat_refs[i][0]])
            ee_euler_ref = r.as_euler("xyz")
            base_euler_ref = np.array([0, 0, ee_euler_ref[2]])

            u_ref = np.zeros(10)
            u_ref[6:10] = default_arm_angle

            # yref layout matches build_cost_expr (stage): EE errors, v, euler, omega, arm, u, du.
            cost_ref = np.concatenate([
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                base_euler_ref,
                np.zeros(3),
                default_arm_angle,
                u_ref,
                np.zeros(10),
            ])

            params = np.concatenate([ee_pos_refs[i], ee_quat_refs[i], default_arm_angle, u_prev])
            self.ocp_solver.set(i, "yref", cost_ref)
            self.ocp_solver.set(i, "p", params)

        r = R.from_quat([ee_quat_refs[-1][1], ee_quat_refs[-1][2], ee_quat_refs[-1][3], ee_quat_refs[-1][0]])
        ee_euler_ref_final = r.as_euler("xyz")
        base_euler_ref_final = np.array([0, 0, ee_euler_ref_final[2]])

        final_cost_ref = np.concatenate([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            base_euler_ref_final,
            np.zeros(3),
            default_arm_angle,
        ])
        params = np.concatenate([ee_pos_refs[-1], ee_quat_refs[-1], default_arm_angle, u_prev])
        self.ocp_solver.set(self.N, "yref", final_cost_ref)
        self.ocp_solver.set(self.N, "p", params)

    def run_optimization(self, x0):
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)
        self.ocp_solver.solve()
        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        return u_opt, x_opt, total_cost

    def run_optimization_with_tracking_cost(self, x0):
        """Run optimization and return both total cost and tracking-only cost"""
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)
        self.ocp_solver.solve()

        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])

        tracking_cost = self._extract_tracking_cost()

        return u_opt, x_opt, total_cost, tracking_cost

    def run_optimization_with_tracking_gradient(self, x0):
        """Run optimization and return total cost, tracking cost, and reference gradients."""
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)
        self.ocp_solver.solve()

        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])

        tracking_cost = self._extract_tracking_cost()
        tracking_gradient = self._extract_tracking_gradients()

        return u_opt, x_opt, total_cost, tracking_cost, tracking_gradient

    def _extract_tracking_cost(self):
        """Compute EE tracking cost using pre-defined cost functions."""
        dt = self.T / self.N
        trk_cost = 0.0

        for k in range(self.N):
            xk = self.ocp_solver.get(k, "x")
            uk = self.ocp_solver.get(k, "u")
            pk = self.ocp_solver.get(k, "p")

            cost_k = float(self._trk_stage_cost_func(xk, uk, pk))
            trk_cost += dt * cost_k

        xN = self.ocp_solver.get(self.N, "x")
        pN = self.ocp_solver.get(self.N, "p")
        cost_N = float(self._trk_term_cost_func(xN, pN))
        trk_cost += cost_N

        return trk_cost

    def get_tracking_cost(self):
        """Get tracking cost from last optimization (call after run_optimization)"""
        return self._extract_tracking_cost()

    def _build_tracking_helpers(self):
        """CasADi helpers for post-solve EE tracking cost and d(cost)/d(ee_pos_ref, ee_quat_ref)."""
        self.trk_idx = np.arange(6)

        pos_err_stage = self.cost_y_expr[0:3]
        pos_err_term = self.cost_y_expr_e[0:3]

        ori_err_stage = self.cost_y_expr[3:6]
        ori_err_term = self.cost_y_expr_e[3:6]

        trk_err_stage = cs.vertcat(pos_err_stage, ori_err_stage)
        trk_err_term = cs.vertcat(pos_err_term, ori_err_term)

        Q_trk = self.Q[np.ix_(self.trk_idx, self.trk_idx)]
        Qe_trk = 10.0 * Q_trk
        self._Q_trk = Q_trk
        self._Qe_trk = Qe_trk

        trk_stage_cost_expr = 0.5 * cs.mtimes([trk_err_stage.T, cs.DM(Q_trk), trk_err_stage])
        trk_term_cost_expr = 0.5 * cs.mtimes([trk_err_term.T, cs.DM(Qe_trk), trk_err_term])

        self._trk_stage_cost_func = cs.Function("stage_trk_cost", [self.x, self.u, self.param], [trk_stage_cost_expr])
        self._trk_term_cost_func = cs.Function("term_trk_cost", [self.x, self.param], [trk_term_cost_expr])

        trk_grad_pos_stage_expr = cs.gradient(trk_stage_cost_expr, self.ee_pos_ref)
        trk_grad_quat_stage_expr = cs.gradient(trk_stage_cost_expr, self.ee_quat_ref)
        trk_grad_combined_stage_expr = cs.vertcat(trk_grad_pos_stage_expr, trk_grad_quat_stage_expr)

        trk_grad_pos_term_expr = cs.gradient(trk_term_cost_expr, self.ee_pos_ref)
        trk_grad_quat_term_expr = cs.gradient(trk_term_cost_expr, self.ee_quat_ref)
        trk_grad_combined_term_expr = cs.vertcat(trk_grad_pos_term_expr, trk_grad_quat_term_expr)

        self._trk_grad_stage_func = cs.Function(
            "grad_stage_trk", [self.x, self.u, self.param], [trk_grad_combined_stage_expr]
        )
        self._trk_grad_term_func = cs.Function("grad_term_trk", [self.x, self.param], [trk_grad_combined_term_expr])

    def _extract_tracking_gradients(self):
        """Return dJ/d(ee_pos_ref, ee_quat_ref) for every *stage* knot.

        Output shape: (N, 7) - combined position and quaternion gradients per time step.
        Format: [pos_grad(3), quat_grad(4)] where quat_grad is w.r.t. [w, x, y, z].
        """
        dt = self.T / self.N
        stage_grads = []

        for k in range(self.N):
            xk = self.ocp_solver.get(k, "x")
            uk = self.ocp_solver.get(k, "u")
            pk = self.ocp_solver.get(k, "p")
            gk = self._trk_grad_stage_func(xk, uk, pk).full().squeeze()
            stage_grads.append(gk * dt)

        return np.stack(stage_grads, axis=0)

    def reset(self, reset_qp_solver_mem=1):
        """Reset the MPC solver to initial state (all zeros)"""
        self.ocp_solver.reset(reset_qp_solver_mem)

    def update_state_pos_bounds(self, pos_min_new=None, pos_max_new=None):
        """
        Dynamically update state bounds (lbx/ubx) for stages 1..N (keep stage0 fixed by x0).
        State x = [p(3), v(3), euler(3), ang_vel(3), arm(4)] => nx=16
        """
        euler_min = np.array([-0.01, -0.01, -np.pi], dtype=np.float64)
        euler_max = np.array([0.01, 0.01, np.pi], dtype=np.float64)

        if pos_min_new is not None:
            self.pos_min = np.array(pos_min_new, dtype=np.float64).copy()
        if pos_max_new is not None:
            self.pos_max = np.array(pos_max_new, dtype=np.float64).copy()

        x_min = np.concatenate([self.pos_min, euler_min, self.joint_min]).astype(np.float64)
        x_max = np.concatenate([self.pos_max, euler_max, self.joint_max]).astype(np.float64)

        for k in range(1, self.N + 1):
            self.ocp_solver.set(k, "lbx", x_min)
            self.ocp_solver.set(k, "ubx", x_max)


class ArmMPCPlanner:
    def __init__(
        self,
        mass,
        T,
        N,
        Q,
        R,
        R_delta,
        pos_min,
        pos_max,
        joint_min,
        joint_max,
        default_arm_angle,
        output_filter_gain,
        moment_of_inertia,
        compile_dir="./acados_mpc_build",
        model_name="fully_actuated_uav",
    ):

        self.mpc = DroneMPC4DOF(
            mass,
            T,
            N,
            Q,
            R,
            R_delta,
            pos_min,
            pos_max,
            joint_min,
            joint_max,
            default_arm_angle,
            moment_of_inertia,
            compile_dir,
            model_name,
        )
        self.x0 = np.zeros(16)
        self.output_filter_gain = np.array(output_filter_gain)
        self.u_cmd_last = None

    def optimize(self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev):
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost = self.mpc.run_optimization(self.x0)
        u_cmd = u_opt[0]
        if self.u_cmd_last is None:
            self.u_cmd_last = u_cmd
        else:
            u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
            self.u_cmd_last = u_cmd
        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        base_euler_opt = x_opt[:, 6:9]
        arm_angle_opt = x_opt[:, 12:16]

        force_opt = u_cmd[0:3]
        torque_cmd = u_cmd[3:6]
        arm_angle_cmd = u_cmd[6:10]

        force_opt = force_opt + self.mpc.mass * np.array([0, 0, 9.81])

        arm_angle_cmd = np.clip(arm_angle_cmd, self.mpc.joint_min, self.mpc.joint_max)

        return (
            force_opt,
            torque_cmd,
            p_opt[-1],
            v_opt[-1],
            arm_angle_opt[-1],
            arm_angle_cmd,
            base_euler_opt[-1],
            total_cost,
        )

    def optimize_with_tracking_cost(self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev):
        """Optimize and return total cost plus post-solve EE tracking cost."""
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost, tracking_cost = self.mpc.run_optimization_with_tracking_cost(self.x0)

        u_cmd = u_opt[0]
        if self.u_cmd_last is None:
            self.u_cmd_last = u_cmd
        else:
            u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
            self.u_cmd_last = u_cmd

        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        base_euler_opt = x_opt[:, 6:9]
        arm_angle_opt = x_opt[:, 12:16]
        force_opt = u_cmd[0:3]
        torque_cmd = u_cmd[3:6]
        arm_angle_cmd = u_cmd[6:10]

        force_opt = force_opt + self.mpc.mass * np.array([0, 0, 9.81])

        arm_angle_cmd = np.clip(arm_angle_cmd, self.mpc.joint_min, self.mpc.joint_max)
        arm_angle_delta = arm_angle_cmd - arm_angle
        arm_vel_limit = np.array([10.0, 10.0, 30.0, 30.0]) * DEGREE_TO_RADIAN
        arm_angle_delta = np.clip(arm_angle_delta, -arm_vel_limit, arm_vel_limit)
        arm_angle_cmd = arm_angle + arm_angle_delta

        predicted_ee_trajectory = self._convert_x_opt_to_ee_poses(x_opt)

        return (
            force_opt,
            torque_cmd,
            p_opt[-1],
            v_opt[-1],
            arm_angle_opt[-1],
            arm_angle_cmd,
            base_euler_opt[-1],
            total_cost,
            tracking_cost,
            predicted_ee_trajectory,
        )

    def optimize_with_tracking_gradient(
        self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev
    ):
        """Optimize and return forces, tracking gradient, and full state/control trajectories."""
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost, tracking_cost, tracking_gradient = self.mpc.run_optimization_with_tracking_gradient(
            self.x0
        )

        u_cmd = u_opt[0]
        if self.u_cmd_last is None:
            self.u_cmd_last = u_cmd
        else:
            u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
            self.u_cmd_last = u_cmd
        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        base_euler_opt = x_opt[:, 6:9]
        arm_angle_opt = x_opt[:, 12:16]

        force_opt = u_cmd[0:3]
        torque_cmd = u_cmd[3:6]
        arm_angle_cmd = u_cmd[6:10]

        force_opt = force_opt + self.mpc.mass * np.array([0, 0, 9.81])

        arm_angle_cmd = np.clip(arm_angle_cmd, self.mpc.joint_min, self.mpc.joint_max)

        return (
            force_opt,
            torque_cmd,
            p_opt[-1],
            v_opt[-1],
            arm_angle_opt[-1],
            arm_angle_cmd,
            base_euler_opt[-1],
            total_cost,
            tracking_gradient,
            x_opt,
            u_opt,
        )

    def forward_kinematics(self, base_pos, base_euler, arm_angle):
        """NumPy FK for the arm chain. Returns [ee_pos(3), ee_quat_wxyz(4)]."""
        q1, q2, q3, q4 = arm_angle[0], arm_angle[1], arm_angle[2], arm_angle[3]

        Rb = rpy_to_rotation_matrix(base_euler)

        c1, s1 = np.cos(q1), np.sin(q1)
        R1 = np.array([[c1, 0.0, s1], [0.0, 1.0, 0.0], [-s1, 0.0, c1]], dtype=float)

        c2, s2 = np.cos(-q2), np.sin(-q2)
        R2 = np.array([[c2, 0.0, s2], [0.0, 1.0, 0.0], [-s2, 0.0, c2]], dtype=float)

        c3, s3 = np.cos(q3), np.sin(q3)
        R3 = np.array([[c3, 0.0, s3], [0.0, 1.0, 0.0], [-s3, 0.0, c3]], dtype=float)

        c4, s4 = np.cos(q4), np.sin(q4)
        R4 = np.array([[1.0, 0.0, 0.0], [0.0, c4, -s4], [0.0, s4, c4]], dtype=float)

        p_arm_base = np.array([0.088, 0.0, 0.0], dtype=float)
        p_j1 = p_arm_base + np.array([0.0, 0.0, 0.06475], dtype=float)

        p_j2_local = np.array([-0.3795, 0.0, 0.059], dtype=float)
        p_j2 = p_j1 + (R1 @ p_j2_local)

        R12 = R1 @ R2

        p_j3_local = np.array([0.4475, 0.0, 0.0], dtype=float)
        p_j3 = p_j2 + (R12 @ p_j3_local)

        R123 = R12 @ R3

        p_j4_local = np.array([0.071, 0.0, 0.0], dtype=float)
        p_j4 = p_j3 + (R123 @ p_j4_local)

        R_end = R123 @ R4
        p_ee = p_j4

        ee_pos_world = (Rb @ p_ee) + np.asarray(base_pos, dtype=float)
        ee_R_world = Rb @ R_end

        ee_quat_wxyz = quaternion_from_rotation_matrix(ee_R_world)
        return np.concatenate([ee_pos_world, ee_quat_wxyz], axis=0)

    def _convert_x_opt_to_ee_poses(self, x_opt):
        """Convert MPC state trajectory to EE poses using forward kinematics.

        Args:
            x_opt: MPC state trajectory of shape (N+1, 16)
                   [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)]

        Returns:
            ee_poses: Array of shape (N+1, 7) containing [pos(3), quat_wxyz(4)]
        """
        ee_poses = []

        for i in range(x_opt.shape[0]):
            state = x_opt[i]
            base_pos = state[0:3]
            base_euler = state[6:9]
            arm_angle = state[12:16]

            ee_pose = self.forward_kinematics(base_pos, base_euler, arm_angle)
            ee_poses.append(ee_pose)

        return np.array(ee_poses)

    def reset(self, reset_qp_solver_mem=1):
        """Reset the MPC planner and clear internal state."""
        self.mpc.reset(reset_qp_solver_mem)
        self.x0 = np.zeros(16)
        self.u_cmd_last = None


class WholeBodyMPCController(BaseController):
    """
    Whole-body MPC controller for a hexacopter with a 4-DOF manipulator arm.
    """

    REQUIRES_ROBOT = True
    REQUIRES_SCENE = True

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device = "cuda",
        dt: float | None = None,
        mpc_params: dict[str, Any] | None = None,
        robot: Articulation | None = None,
        scene=None,
        robot_spec=None,
        **kwargs,
    ):
        """Initialize the whole-body MPC controller.

        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on ("cuda" or "cpu")
            dt: Time step in seconds
            mpc_params: MPC parameters dictionary containing T, N, Q, R, etc.
            robot: Articulation object. Required to compute physical parameters (mass, inertia) from simulator.
            scene: Scene object used to read gravity from the simulation.
            robot_spec: Robot morphology and multirotor specification.
            **kwargs: Additional arguments passed to BaseController (enable_saturation, enable_aerodynamic_effects, enable_wind_effect, etc.)
        """
        super().__init__(
            num_envs=num_envs,
            device=device,
            dt=dt,
            robot_spec=robot_spec,
            robot=robot,
            scene=scene,
            **kwargs,
        )

        self.T = mpc_params["T"]
        self.N = mpc_params["N"]
        self.Q = mpc_params["Q"]
        self.R = mpc_params["R"]
        self.R_delta = mpc_params["R_delta"]
        self.pos_min = mpc_params["pos_min"]
        self.pos_max = mpc_params["pos_max"]
        self.joint_min = mpc_params["joint_min"]
        self.joint_max = mpc_params["joint_max"]
        self.default_arm_angle = mpc_params["default_arm_angle"]
        self.output_filter_gain = mpc_params["output_filter_gain"]
        self.compile_dir = mpc_params.get("compile_dir", "./acados_mpc_build")
        self.model_name = mpc_params.get("model_name", "fa_hexa_mpc")

        mass, inertia_3x3, _ = get_aggregate_physical_parameters(robot, scene)
        inertia_diag = torch.diag(inertia_3x3).cpu().numpy()

        self.mass = float(mass)
        self.inertia_diag = inertia_diag

        self.planners = []
        for e in range(self.num_envs):
            planner = ArmMPCPlanner(
                mass=self.mass,
                T=self.T,
                N=self.N,
                Q=self.Q,
                R=self.R,
                R_delta=self.R_delta,
                pos_min=self.pos_min,
                pos_max=self.pos_max,
                joint_min=self.joint_min,
                joint_max=self.joint_max,
                default_arm_angle=self.default_arm_angle,
                output_filter_gain=self.output_filter_gain,
                moment_of_inertia=self.inertia_diag,
                compile_dir=self.compile_dir,
                model_name=self.model_name,
            )
            self.planners.append(planner)

        self.x_d = torch.zeros((self.num_envs, 19), device=self.device)
        self.x_d[:, 3] = 1.0
        self.x_d_prev = torch.zeros((self.num_envs, 19), device=self.device)
        self.x_d_prev[:, 3] = 1.0

        self.ee_x_d = self.x_d
        self.ee_x_d_prev = self.x_d_prev

        self.u_prev = np.zeros((self.num_envs, 10), dtype=np.float64)
        self.tracking_gradient = np.zeros((self.N, 7), dtype=np.float64)

        self.reference_from_dp = False
        self.reference_traj = np.zeros((self.N, 7), dtype=np.float32)

    def reset(self, default_root_state: torch.Tensor, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.ee_x_d[env_ids, 0:7] = default_root_state[env_ids, 0:7]
        self.ee_x_d_prev[env_ids, 0:7] = self.ee_x_d[env_ids, 0:7]
        self.ee_x_d[env_ids, 7:19] = 0.0
        self.ee_x_d_prev[env_ids, 7:19] = 0.0

        env_ids_np = env_ids.cpu().numpy().tolist()
        for e in env_ids_np:
            self.planners[e].reset()
            self.u_prev[e, :] = 0.0

    def compute_desired_states(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        is_first_step: torch.Tensor,
    ) -> None:

        target_quat = math_utils.normalize(target_quat)
        pos_d_prev = self.ee_x_d_prev[:, 0:3]
        quat_prev = self.ee_x_d_prev[:, 3:7]
        lin_vel_d_prev = self.ee_x_d_prev[:, 7:10]
        ang_vel_d_prev = self.ee_x_d_prev[:, 10:13]

        self.ee_x_d[:, 0:3] = target_pos
        self.ee_x_d[:, 3:7] = target_quat

        lin_vel_d = torch.zeros_like(target_pos)
        ang_vel_d = torch.zeros((self.num_envs, 3), device=self.device, dtype=target_pos.dtype)
        acc_d = torch.zeros_like(target_pos)
        ang_acc_d = torch.zeros((self.num_envs, 3), device=self.device, dtype=target_pos.dtype)

        not_first = ~is_first_step
        if not_first.any():
            lin_vel_d[not_first] = (target_pos[not_first] - pos_d_prev[not_first]) / self.dt

            quat_prev_inv = math_utils.quat_conjugate(quat_prev[not_first])
            quat_rel = math_utils.quat_mul(target_quat[not_first], quat_prev_inv)

            quat_rel_w = quat_rel[:, 0:1]
            quat_rel_xyz = quat_rel[:, 1:4]
            quat_rel_xyz_norm = torch.norm(quat_rel_xyz, dim=1, keepdim=True)

            small_angle_threshold = 1e-6
            angle = 2.0 * torch.atan2(quat_rel_xyz_norm, quat_rel_w + 1e-8)

            log_quat = torch.where(
                quat_rel_xyz_norm > small_angle_threshold,
                (angle / (quat_rel_xyz_norm + 1e-8)) * quat_rel_xyz,
                quat_rel_xyz,
            )

            ang_vel_computed = 2.0 * log_quat / self.dt
            ang_vel_d[not_first] = ang_vel_computed

            acc_d[not_first] = (lin_vel_d[not_first] - lin_vel_d_prev[not_first]) / self.dt
            ang_acc_d[not_first] = (ang_vel_d[not_first] - ang_vel_d_prev[not_first]) / self.dt

        self.ee_x_d[:, 7:10] = lin_vel_d.clamp(-1.0, 1.0)
        self.ee_x_d[:, 10:13] = ang_vel_d.clamp(-1.0, 1.0)
        self.ee_x_d[:, 13:16] = acc_d.clamp(-1.0, 1.0)
        self.ee_x_d[:, 16:19] = ang_acc_d.clamp(-1.0, 1.0)

        self.ee_x_d_prev[:, 0:3] = target_pos
        self.ee_x_d_prev[:, 3:7] = target_quat
        self.ee_x_d_prev[:, 7:10] = lin_vel_d
        self.ee_x_d_prev[:, 10:13] = ang_vel_d

    def compute_mpc(self, obs: torch.Tensor):
        base_pos = obs[:, 0:3]
        base_quat = obs[:, 3:7]
        base_lin_vel = obs[:, 7:10]
        base_ang_vel = obs[:, 10:13]
        arm_q = obs[:, 13:17]

        base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(base_quat)
        base_euler = torch.stack([base_roll, base_pitch, base_yaw], dim=1)

        force_out = torch.zeros((self.num_envs, 3), device=self.device)
        torque_out = torch.zeros((self.num_envs, 3), device=self.device)
        arm_cmd_out = torch.zeros((self.num_envs, 4), device=self.device)

        ee_x_np = self.ee_x_d.cpu().numpy()
        ee_pos_np = ee_x_np[:, 0:3]
        ee_quat_np = ee_x_np[:, 3:7]

        N = self.N

        for e in range(self.num_envs):
            if self.reference_from_dp:
                ee_pos_refs = self.reference_traj[:, 0:3]
                ee_quat_refs = self.reference_traj[:, 3:7]
            else:
                ee_pos_refs = np.tile(ee_pos_np[e], (N + 1, 1))
                ee_quat_refs = np.tile(ee_quat_np[e], (N + 1, 1))

            (
                force_cmd,
                torque_cmd,
                p_opt,
                v_opt,
                _,
                arm_angle_cmd,
                base_euler_opt,
                _,
                tracking_gradient,
                _,
                _,
            ) = self.planners[e].optimize_with_tracking_gradient(
                p=base_pos[e].cpu().numpy(),
                v=base_lin_vel[e].cpu().numpy(),
                arm_angle=arm_q[e].cpu().numpy(),
                base_euler=base_euler[e].cpu().numpy(),
                base_ang_vel=base_ang_vel[e].cpu().numpy(),
                ee_pos_refs=ee_pos_refs,
                ee_quat_refs=ee_quat_refs,
                u_prev=self.u_prev[e],
            )
            self.tracking_gradient = tracking_gradient

            self.x_d[e, 0:3] = torch.tensor(p_opt, device=self.device)
            base_roll_o, base_pitch_o, base_yaw_o = torch.tensor(base_euler_opt, device=self.device)
            self.x_d[e, 3:7] = math_utils.quat_from_euler_xyz(base_roll_o, base_pitch_o, base_yaw_o)

            self.u_prev[e, 0:3] = force_cmd
            self.u_prev[e, 3:6] = torque_cmd
            self.u_prev[e, 6:10] = arm_angle_cmd

            force_out[e] = torch.tensor(force_cmd, device=self.device)
            torque_out[e] = torch.tensor(torque_cmd, device=self.device)
            arm_cmd_out[e] = torch.tensor(arm_angle_cmd, device=self.device)

            # MPC wrench is world-frame; PID expects body frame.
            R_wb = math_utils.matrix_from_euler(base_euler[e].unsqueeze(0), "XYZ")[0].T
            force_out[e] = R_wb @ force_out[e]
            torque_out[e] = R_wb @ torque_out[e]

        return force_out, torque_out, arm_cmd_out

    def compute_desired_wrench(self, obs: torch.Tensor, *args, **kwargs):
        force_cmd, torque_cmd, _ = self.compute_mpc(obs)
        wrench_cmd = torch.zeros((self.num_envs, 6), device=self.device)
        wrench_cmd[:, 0:3] = force_cmd
        wrench_cmd[:, 3:6] = torque_cmd
        return wrench_cmd

    def update_pos_max_from_wall(self, env, safety_margin: float = 0.85):
        """Update MPC position bounds for all environments based on wall/door positions."""
        min_z_limit = 0.0
        max_x_limits = {}

        if hasattr(env.unwrapped, "wall") and env.unwrapped.wall is not None:
            wall_pos_all = env.unwrapped.wall.data.root_pos_w.cpu().numpy()
            wall_size = env.unwrapped.cfg.wall.size
            wall_thickness = float(wall_size[0])

            for e in range(self.num_envs):
                wall_pos = wall_pos_all[e]
                wall_front_face_x = float(wall_pos[0]) - wall_thickness / 2.0
                max_x_limits[e] = wall_front_face_x - float(safety_margin)

        elif hasattr(env.unwrapped, "pulling_door") and env.unwrapped.pulling_door is not None:
            door_pos_all = env.unwrapped.pulling_door.data.root_pos_w.cpu().numpy()

            for e in range(self.num_envs):
                door_pos = door_pos_all[e]
                cabinet_front_face_x = float(door_pos[0])
                max_x_limits[e] = cabinet_front_face_x - float(safety_margin)

        for e in range(self.num_envs):
            mpc = self.planners[e].mpc

            if e in max_x_limits:
                mpc.pos_max[0] = max_x_limits[e]
            mpc.pos_min[2] = min_z_limit

            mpc.update_state_pos_bounds(pos_min_new=mpc.pos_min, pos_max_new=mpc.pos_max)

        return {"max_x": max_x_limits if max_x_limits else None, "min_z": min_z_limit}
