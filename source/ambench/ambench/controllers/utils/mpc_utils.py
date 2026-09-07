# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation as R

############################################################################################################
# CasADi rotation matrix and quaternion conversion functions


def quaternion_from_rotation_matrix(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    qx, qy, qz, qw = quaternion
    quaternion = np.array([qw, qx, qy, qz])
    return quaternion


def rpy_to_rotation_matrix(rpy):
    ypr = rpy[::-1]
    r = R.from_euler("zyx", ypr, degrees=False)
    return r.as_matrix()


def quaternion_from_rotation_matrix_ca(R):
    """
    Convert a rotation matrix to a quaternion using CasADi symbolic operations.

    Parameters:
    R (casadi.SX or casadi.MX): A 3x3 rotation matrix.

    Returns:
    q (casadi.SX or casadi.MX): A 4-element quaternion vector [qw, qx, qy, qz].
    """
    # Compute the elements of the quaternion
    qw = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    qx = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 + R[0, 0] - R[1, 1] - R[2, 2]))
    qy = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 - R[0, 0] + R[1, 1] - R[2, 2]))
    qz = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 - R[0, 0] - R[1, 1] + R[2, 2]))

    # Adjust the signs of the quaternion components
    qx = ca.if_else((R[2, 1] - R[1, 2]) < 0, -qx, qx)
    qy = ca.if_else((R[0, 2] - R[2, 0]) < 0, -qy, qy)
    qz = ca.if_else((R[1, 0] - R[0, 1]) < 0, -qz, qz)

    # Assemble the quaternion vector
    q = ca.vertcat(qw, qx, qy, qz)

    return q


def rotation_matrix_from_euler_ca(euler):
    """
    Convert Euler angles to a rotation matrix using CasADi symbolic operations in yaw-pitch-roll order.

    Parameters:
    euler (casadi.SX or casadi.MX): A 3-element vector of Euler angles [yaw, pitch, roll].

    Returns:
    R (casadi.SX or casadi.MX): A 3x3 rotation matrix.

    Note:
    The rotations are applied in the yaw-pitch-roll order.
    """
    # Extract the Euler angles
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    # Compute the sine and cosine of the Euler angles
    c1 = ca.cos(yaw)  # cos(yaw)
    c2 = ca.cos(pitch)  # cos(pitch)
    c3 = ca.cos(roll)  # cos(roll)
    s1 = ca.sin(yaw)  # sin(yaw)
    s2 = ca.sin(pitch)  # sin(pitch)
    s3 = ca.sin(roll)  # sin(roll)

    # Construct the rotation matrix
    R = ca.MX.zeros(3, 3)  # Create a 3x3 matrix

    # First row
    R[0, 0] = c1 * c2
    R[0, 1] = c1 * s2 * s3 - s1 * c3
    R[0, 2] = c1 * s2 * c3 + s1 * s3

    # Second row
    R[1, 0] = s1 * c2
    R[1, 1] = s1 * s2 * s3 + c1 * c3
    R[1, 2] = s1 * s2 * c3 - c1 * s3

    # Third row
    R[2, 0] = -s2
    R[2, 1] = c2 * s3
    R[2, 2] = c2 * c3

    return R
