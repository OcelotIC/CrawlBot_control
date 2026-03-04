"""Attitude transformations: quaternions, DCM, Euler angles.

SPART convention: q = [q1, q2, q3, q4] where q4 is the SCALAR part.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion <-> DCM
# ---------------------------------------------------------------------------

def quat_dcm(q):
    """Direction Cosine Matrix from quaternion (SPART convention, scalar last).

    Parameters
    ----------
    q : (4,) array – [q1, q2, q3, q4] with q4 scalar.

    Returns
    -------
    DCM : (3, 3) array
    """
    q1, q2, q3, q4 = q
    return np.array([
        [1 - 2*(q2**2 + q3**2),    2*(q1*q2 + q3*q4),    2*(q1*q3 - q2*q4)],
        [2*(q2*q1 - q3*q4),    1 - 2*(q1**2 + q3**2),    2*(q2*q3 + q1*q4)],
        [2*(q3*q1 + q2*q4),    2*(q3*q2 - q1*q4),    1 - 2*(q1**2 + q2**2)],
    ])


def dcm_quat(DCM):
    """Quaternion from Direction Cosine Matrix (SPART convention, scalar last).

    Parameters
    ----------
    DCM : (3, 3) array

    Returns
    -------
    q : (4,) array – [q1, q2, q3, q4] with q4 scalar.
    """
    tr = np.trace(DCM)
    qs = np.array([
        0.25 * (1 + 2*DCM[0, 0] - tr),
        0.25 * (1 + 2*DCM[1, 1] - tr),
        0.25 * (1 + 2*DCM[2, 2] - tr),
        0.25 * (1 + tr),
    ])
    I = np.argmax(qs)
    ql = np.sqrt(qs[I])

    if I == 0:
        q = np.array([
            ql,
            (DCM[0, 1] + DCM[1, 0]) / (4*ql),
            (DCM[2, 0] + DCM[0, 2]) / (4*ql),
            (DCM[1, 2] - DCM[2, 1]) / (4*ql),
        ])
    elif I == 1:
        q = np.array([
            (DCM[0, 1] + DCM[1, 0]) / (4*ql),
            ql,
            (DCM[1, 2] + DCM[2, 1]) / (4*ql),
            (DCM[2, 0] - DCM[0, 2]) / (4*ql),
        ])
    elif I == 2:
        q = np.array([
            (DCM[2, 0] + DCM[0, 2]) / (4*ql),
            (DCM[1, 2] + DCM[2, 1]) / (4*ql),
            ql,
            (DCM[0, 1] - DCM[1, 0]) / (4*ql),
        ])
    else:  # I == 3
        q = np.array([
            (DCM[1, 2] - DCM[2, 1]) / (4*ql),
            (DCM[2, 0] - DCM[0, 2]) / (4*ql),
            (DCM[0, 1] - DCM[1, 0]) / (4*ql),
            ql,
        ])
    return q


# ---------------------------------------------------------------------------
# Euler axis-angle <-> DCM
# ---------------------------------------------------------------------------

def euler_dcm(e, alpha):
    """DCM from Euler axis *e* and angle *alpha* (via quaternion)."""
    q = np.array([
        e[0] * np.sin(alpha / 2),
        e[1] * np.sin(alpha / 2),
        e[2] * np.sin(alpha / 2),
        np.cos(alpha / 2),
    ])
    return quat_dcm(q)


def dcm_euler(DCM):
    """Rotation axis and angle from a DCM.

    Returns
    -------
    result : (4,) array – [e1, e2, e3, alpha]
    """
    alpha = np.real(np.arccos(np.clip((np.trace(DCM) - 1) / 2, -1, 1)))
    if np.abs(np.sin(alpha)) < 1e-12:
        e = np.array([1.0, 0.0, 0.0])
    else:
        e = (1 / (2 * np.sin(alpha))) * np.array([
            DCM[1, 2] - DCM[2, 1],
            DCM[2, 0] - DCM[0, 2],
            DCM[0, 1] - DCM[1, 0],
        ])
    return np.array([e[0], e[1], e[2], alpha])


# ---------------------------------------------------------------------------
# Euler angles (321 / ZYX sequence) <-> DCM
# ---------------------------------------------------------------------------

def angles321_dcm(angles):
    """DCM from Euler angles (321 / ZYX sequence).

    Parameters
    ----------
    angles : (3,) array – [phi (x-roll), theta (y-pitch), psi (z-yaw)]

    Returns
    -------
    DCM : (3, 3) array
    """
    phi, theta, psi = angles
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [ct*cy,                    ct*sy,                   -st],
        [sp*st*cy - cp*sy,         sp*st*sy + cp*cy,         sp*ct],
        [cp*st*cy + sp*sy,         cp*st*sy - sp*cy,         cp*ct],
    ])


def dcm_angles321(DCM):
    """Euler angles (321 / ZYX sequence) from DCM.

    Returns
    -------
    angles : (3,) array – [phi, theta, psi]
    """
    return np.array([
        np.arctan2(DCM[1, 2], DCM[2, 2]),
        np.arcsin(np.clip(-DCM[0, 2], -1, 1)),
        np.arctan2(DCM[0, 1], DCM[0, 0]),
    ])


# ---------------------------------------------------------------------------
# Euler angles (123 / XYZ sequence) <-> DCM
# ---------------------------------------------------------------------------

def angles123_dcm(angles):
    """DCM from Euler angles (123 / XYZ sequence).

    Parameters
    ----------
    angles : (3,) array – [phi, theta, psi]
    """
    phi, theta, psi = angles
    C1 = np.array([[1, 0, 0],
                    [0, np.cos(phi), np.sin(phi)],
                    [0, -np.sin(phi), np.cos(phi)]])
    C2 = np.array([[np.cos(theta), 0, -np.sin(theta)],
                    [0, 1, 0],
                    [np.sin(theta), 0, np.cos(theta)]])
    C3 = np.array([[np.cos(psi), np.sin(psi), 0],
                    [-np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])
    return C3 @ C2 @ C1


# ---------------------------------------------------------------------------
# Quaternion -> Euler 321
# ---------------------------------------------------------------------------

def quat_angles321(q):
    """Convert quaternion to Euler angles (321 sequence)."""
    DCM = quat_dcm(q)
    return dcm_angles321(DCM)
