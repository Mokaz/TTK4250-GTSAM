"""Small helpers used across the assignment. Nothing here is graded."""

from __future__ import annotations

import gtsam
import numpy as np


def rotmat2(thetas):
    """2x2 rotation matrix if a single theta, or Nx2x2 if an array of thetas."""
    thetas = np.asarray(thetas)

    c = np.cos(thetas)
    s = np.sin(thetas)

    return np.stack(
        [
            np.stack([c, -s], axis=-1),
            np.stack([s, c], axis=-1),
        ],
        axis=-2,
    )


def ssa(angle):
    """Smallest signed angle, wrapped to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cartesian2polar(x: float, y: float) -> tuple[float, float]:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T)/2 to clean up small asymmetries from numerics."""
    return (A + A.T) / 2


def make_psd(A: np.ndarray) -> np.ndarray:
    """Project a matrix onto the closest positive semi-definite matrix."""
    A_sym = symmetrize(A)
    eigval, eigvec = np.linalg.eigh(A_sym)
    eigval[eigval < 0] = 0
    return eigvec @ np.diag(eigval) @ eigvec.T


def pose2_to_array(p: gtsam.Pose2) -> np.ndarray:
    """Convert a gtsam.Pose2 to np.array([x, y, theta])."""
    return np.array([p.x(), p.y(), p.theta()])


def pose2_tangent_error(estimate: gtsam.Pose2, truth: gtsam.Pose2) -> np.ndarray:
    """Pose error expressed in the tangent space at ``estimate``.

    This is the error that the covariance returned by GTSAM describes, so it is
    the one that must be used for NEES. Naively subtracting [x, y, theta] is
    *not* the same thing and will give you a wrong (usually too large) NEES.
    """
    return gtsam.Pose2.Logmap(estimate.between(truth))


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function of a vector.

    Used by the tests; handy for checking your own analytic Jacobians.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(np.asarray(f(x), dtype=float))
    J = np.zeros((f0.size, x.size))

    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        f_plus = np.atleast_1d(np.asarray(f(x + dx), dtype=float))
        f_minus = np.atleast_1d(np.asarray(f(x - dx), dtype=float))
        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J
