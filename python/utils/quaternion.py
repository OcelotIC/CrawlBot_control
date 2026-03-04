"""Quaternion utilities compatible with SPART convention.

Convention: q = [q1, q2, q3, q4] where q4 is the SCALAR part.
"""

import numpy as np


def quat_product(q1, q2):
    """Quaternion product q1 ⊗ q2 (SPART convention: scalar last).

    Formula:
        qv = q1_s*q2_v + q2_s*q1_v + q1_v × q2_v
        qs = q1_s*q2_s - q1_v · q2_v
    """
    q1_v = q1[:3]
    q1_s = q1[3]
    q2_v = q2[:3]
    q2_s = q2[3]

    qv = q1_s * q2_v + q2_s * q1_v + np.cross(q1_v, q2_v)
    qs = q1_s * q2_s - np.dot(q1_v, q2_v)
    return np.array([qv[0], qv[1], qv[2], qs])


def quat_integrate(q, omega, dt):
    """Integrate quaternion using exact Rodrigues formula.

    Parameters
    ----------
    q : (4,) array  – current quaternion (SPART convention)
    omega : (3,) array – angular velocity
    dt : float – time step

    Returns
    -------
    q_next : (4,) array – updated quaternion (normalized)
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-12:
        return q.copy()

    theta = omega_norm * dt
    axis = omega / omega_norm

    # Rotation quaternion (SPART: [qv; qs])
    q_rot = np.array([
        np.sin(theta / 2) * axis[0],
        np.sin(theta / 2) * axis[1],
        np.sin(theta / 2) * axis[2],
        np.cos(theta / 2),
    ])

    q_next = quat_product(q_rot, q)
    q_next /= np.linalg.norm(q_next)
    return q_next


def quat_error(q_des, q_cur):
    """Quaternion error (vector part of q_des ⊗ conj(q_cur)).

    Returns 2 * vector part as the orientation error (small-angle approx).
    """
    q_conj_cur = np.array([-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]])
    q_err = quat_product(q_des, q_conj_cur)
    return 2.0 * q_err[:3]


def matrix_quat_prod_left(q):
    """Left quaternion product matrix: q ⊗ p = [q]_L p."""
    return np.array([
        [ q[3], -q[2],  q[1], q[0]],
        [ q[2],  q[3], -q[0], q[1]],
        [-q[1],  q[0],  q[3], q[2]],
        [-q[0], -q[1], -q[2], q[3]],
    ])


def matrix_quat_prod_right(q):
    """Right quaternion product matrix: p ⊗ q = [q]_R p."""
    return np.array([
        [ q[3],  q[2], -q[1], q[0]],
        [-q[2],  q[3],  q[0], q[1]],
        [ q[1], -q[0],  q[3], q[2]],
        [-q[0], -q[1], -q[2], q[3]],
    ])


def quaternion_product(q1, q2):
    """Alias using matrix form: q1 ⊗ q2 = [q1]_L q2."""
    return matrix_quat_prod_left(q1) @ q2
