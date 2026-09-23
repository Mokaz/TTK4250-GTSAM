"""Task 1 (a) relative_pose and (b) preintegrate."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest

from graphslam.preprocessing import Car, preintegrate, relative_pose

CAR = Car()


def _integrate_unicycle(velocity: float, yaw_rate: float, dt: float) -> np.ndarray:
    """Reference solution by numerical integration of the unicycle ODE.

    Independent of anything in the Lie group machinery, so it catches an Euler
    step masquerading as an exact integration.
    """
    from scipy.integrate import solve_ivp

    def dynamics(_t, state):
        _, _, theta = state
        return [velocity * np.cos(theta), velocity * np.sin(theta), yaw_rate]

    solution = solve_ivp(
        dynamics, (0.0, dt), [0.0, 0.0, 0.0], rtol=1e-12, atol=1e-12, dense_output=True
    )
    return solution.y[:, -1]


# ---------------------------------------------------------------------------
# (a) relative_pose
# ---------------------------------------------------------------------------


def test_relative_pose_straight_motion() -> None:
    pose = relative_pose(vel_encoder=2.0, steer=0.0, dt=0.5)

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()], [1.0, 0.0, 0.0], atol=1e-12
    )


def test_relative_pose_applies_the_encoder_offset_correction() -> None:
    """The encoder is off the centre line, so v_body != v_encoder when turning."""
    vel_encoder, steer, dt = 3.0, 0.20, 0.4

    velocity_body = vel_encoder / (1.0 - (CAR.H / CAR.L) * np.tan(steer))
    expected_yaw = (velocity_body / CAR.L) * np.tan(steer) * dt

    pose = relative_pose(vel_encoder, steer, dt)

    assert pose.theta() == pytest.approx(expected_yaw, rel=1e-9)


def test_relative_pose_follows_an_arc_not_a_straight_line() -> None:
    """Exact integration of the twist, i.e. Expmap -- not an Euler step."""
    vel_encoder, steer, dt = 4.0, 0.35, 1.0

    velocity_body = vel_encoder / (1.0 - (CAR.H / CAR.L) * np.tan(steer))
    yaw_rate = (velocity_body / CAR.L) * np.tan(steer)
    expected = _integrate_unicycle(velocity_body, yaw_rate, dt)

    pose = relative_pose(vel_encoder, steer, dt)

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()], expected, atol=1e-4
    )

    # An Euler step would put the whole displacement on the body x-axis.
    assert abs(pose.y()) > 1e-3, "the increment should have a lateral component"


def test_relative_pose_is_zero_for_zero_velocity() -> None:
    pose = relative_pose(vel_encoder=0.0, steer=0.3, dt=1.0)

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()], [0.0, 0.0, 0.0], atol=1e-12
    )


# ---------------------------------------------------------------------------
# (b) preintegrate
# ---------------------------------------------------------------------------


def test_preintegrate_of_nothing_is_the_identity() -> None:
    pose, covariance = preintegrate([], [])

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()], [0.0, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(covariance, np.zeros((3, 3)), atol=1e-12)


def test_preintegrate_compounds_the_mean() -> None:
    a = gtsam.Pose2(1.0, 0.0, np.pi / 2)
    b = gtsam.Pose2(2.0, 0.0, 0.0)

    pose, _ = preintegrate([a, b], [np.eye(3) * 1e-6] * 2)
    expected = a.compose(b)

    np.testing.assert_allclose(
        [pose.x(), pose.y(), pose.theta()],
        [expected.x(), expected.y(), expected.theta()],
        atol=1e-12,
    )


def test_preintegrate_of_a_single_increment_returns_its_covariance() -> None:
    """Composing onto the identity must leave the covariance untouched."""
    increment = gtsam.Pose2(1.5, -0.2, 0.3)
    covariance = np.diag([0.04, 0.01, 0.0009])

    _, compounded_cov = preintegrate([increment], [covariance])

    np.testing.assert_allclose(compounded_cov, covariance, atol=1e-12)


def test_preintegrate_transports_covariance_through_the_second_increment() -> None:
    """The first increment's covariance is rotated by the adjoint of the second."""
    first = gtsam.Pose2(1.0, 0.0, 0.4)
    second = gtsam.Pose2(2.0, 0.5, -0.7)

    covariance = np.diag([0.04, 0.01, 0.0009])
    _, compounded_cov = preintegrate([first, second], [covariance, np.zeros((3, 3))])

    adjoint = second.inverse().AdjointMap()
    expected = adjoint @ covariance @ adjoint.T

    np.testing.assert_allclose(compounded_cov, expected, atol=1e-10)


def test_preintegrate_does_not_simply_sum_covariances() -> None:
    """A rotation mixes the components: the result is not the naive sum."""
    increments = [gtsam.Pose2(1.0, 0.0, 0.5), gtsam.Pose2(1.0, 0.0, 0.5)]
    covariances = [np.diag([0.04, 0.01, 0.0009])] * 2

    _, compounded_cov = preintegrate(increments, covariances)

    naive_sum = covariances[0] + covariances[1]
    assert not np.allclose(compounded_cov, naive_sum, atol=1e-6)
    # ... and it must still be a valid covariance.
    np.testing.assert_allclose(compounded_cov, compounded_cov.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(compounded_cov) >= -1e-12)


def test_preintegrate_covariance_grows_with_more_increments() -> None:
    increment = gtsam.Pose2(1.0, 0.0, 0.1)
    covariance = np.diag([0.04, 0.01, 0.0009])

    _, short = preintegrate([increment] * 2, [covariance] * 2)
    _, long = preintegrate([increment] * 5, [covariance] * 5)

    assert np.trace(long) > np.trace(short)
