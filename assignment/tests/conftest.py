"""Shared fixtures and helpers for the test suite."""

from __future__ import annotations

import gtsam  # noqa: F401 - imported so a missing GTSAM fails loudly and early
import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260914)


def numerical_jacobian_pose(f, pose, output_dim: int, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``f(pose)`` in GTSAM's tangent convention.

    The perturbation is applied with ``Pose2.retract``, which is the same
    retraction GTSAM's analytic Jacobians and marginal covariances use. A
    Jacobian taken with respect to raw ``[x, y, theta]`` instead would not match.
    """
    J = np.zeros((output_dim, 3))

    for i in range(3):
        dx = np.zeros(3)
        dx[i] = eps
        f_plus = np.atleast_1d(np.asarray(f(pose.retract(dx)), dtype=float))
        f_minus = np.atleast_1d(np.asarray(f(pose.retract(-dx)), dtype=float))
        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J


def numerical_jacobian_point(f, point, output_dim: int, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``f(point)`` for a 2D point."""
    point = np.asarray(point, dtype=float)
    J = np.zeros((output_dim, 2))

    for i in range(2):
        dx = np.zeros(2)
        dx[i] = eps
        f_plus = np.atleast_1d(np.asarray(f(point + dx), dtype=float))
        f_minus = np.atleast_1d(np.asarray(f(point - dx), dtype=float))
        J[:, i] = (f_plus - f_minus) / (2 * eps)

    return J
