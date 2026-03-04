"""Spatial math utilities: skew-symmetric matrices, SE(3) adjoint."""

import numpy as np


def skew_sym(x):
    """Compute the 3x3 skew-symmetric matrix of a 3-vector.

    [x]_× such that [x]_× @ y == cross(x, y).
    """
    x = np.asarray(x).ravel()
    return np.array([
        [0,    -x[2],  x[1]],
        [x[2],  0,    -x[0]],
        [-x[1], x[0],  0   ],
    ])


def adjoint_se3(g):
    """Compute the 6x6 adjoint matrix of a homogeneous transform g in SE(3).

    g = [R, p; 0 0 0 1]

    Ad_g = [ R,        [p]_× R ]
           [ 0,        R       ]

    Parameters
    ----------
    g : (4, 4) array
        Homogeneous transformation matrix.

    Returns
    -------
    Ad_g : (6, 6) array
    """
    R = g[:3, :3]
    p = g[:3, 3]
    p_skew = skew_sym(p)
    Ad_g = np.zeros((6, 6))
    Ad_g[:3, :3] = R
    Ad_g[:3, 3:] = p_skew @ R
    Ad_g[3:, 3:] = R
    return Ad_g
