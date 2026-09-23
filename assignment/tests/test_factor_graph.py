"""Task 1 (c) graph construction, (d) measurement model, (e) covariance
reordering, (f) innovation covariance and (g1) inverse measurement."""

from __future__ import annotations

import gtsam
import numpy as np
import pytest
from gtsam.symbol_shorthand import L, X

from conftest import numerical_jacobian_point, numerical_jacobian_pose
from graphslam.config import SlamConfig
from graphslam.factor_graph import (
    add_landmark_factor,
    add_odometry_factor,
    bearing_range_noise_model,
    innovation_covariance,
    inverse_measurement,
    predict_measurement,
    predict_pose,
    reorder_joint_covariance,
)
from graphslam.utils import ssa


# ---------------------------------------------------------------------------
# (c1) predict_pose
# ---------------------------------------------------------------------------


def test_predict_pose_composes_in_the_body_frame() -> None:
    previous = gtsam.Pose2(1.0, 2.0, np.pi / 2)
    increment = gtsam.Pose2(3.0, 0.0, 0.0)

    predicted = predict_pose(previous, increment)

    # Heading is +90 deg, so a forward increment moves along +y in the world.
    np.testing.assert_allclose(
        [predicted.x(), predicted.y(), predicted.theta()],
        [1.0, 5.0, np.pi / 2],
        atol=1e-12,
    )


def test_predict_pose_is_not_a_component_wise_sum() -> None:
    previous = gtsam.Pose2(0.0, 0.0, 0.7)
    increment = gtsam.Pose2(1.0, 0.0, 0.2)

    predicted = predict_pose(previous, increment)

    assert predicted.x() != pytest.approx(1.0), "the increment must be rotated into the world frame"
    assert predicted.theta() == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# (c2) add_odometry_factor
# ---------------------------------------------------------------------------


def test_add_odometry_factor_adds_one_binary_factor() -> None:
    graph = gtsam.NonlinearFactorGraph()
    add_odometry_factor(graph, X(0), X(1), gtsam.Pose2(1.0, 0.0, 0.0), np.eye(3) * 0.01)

    assert graph.size() == 1
    assert list(graph.at(0).keys()) == [X(0), X(1)]


def test_add_odometry_factor_has_zero_error_at_the_exact_solution() -> None:
    graph = gtsam.NonlinearFactorGraph()
    increment = gtsam.Pose2(1.0, 0.5, 0.3)
    add_odometry_factor(graph, X(0), X(1), increment, np.eye(3) * 0.01)

    values = gtsam.Values()
    start = gtsam.Pose2(2.0, -1.0, 0.8)
    values.insert(X(0), start)
    values.insert(X(1), start.compose(increment))

    assert graph.error(values) == pytest.approx(0.0, abs=1e-12)


def test_add_odometry_factor_uses_the_full_covariance() -> None:
    """A correlated covariance must not be flattened to its diagonal.

    GTSAM's ``error`` is half the squared Mahalanobis distance of the residual,
    so a known perturbation pins down exactly which matrix was used.
    """
    covariance = np.array(
        [
            [0.04, 0.02, 0.00],
            [0.02, 0.03, 0.01],
            [0.00, 0.01, 0.02],
        ]
    )

    graph = gtsam.NonlinearFactorGraph()
    increment = gtsam.Pose2(1.0, 0.0, 0.0)
    add_odometry_factor(graph, X(0), X(1), increment, covariance)

    perturbation = np.array([0.05, -0.03, 0.02])

    values = gtsam.Values()
    values.insert(X(0), gtsam.Pose2(0.0, 0.0, 0.0))
    values.insert(X(1), increment.retract(perturbation))

    expected = 0.5 * perturbation @ np.linalg.solve(covariance, perturbation)

    assert graph.error(values) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# (c3) add_landmark_factor
# ---------------------------------------------------------------------------


def _config_with(**noise_overrides) -> SlamConfig:
    config = SlamConfig()
    for name, value in noise_overrides.items():
        setattr(config.noise, name, value)
    return config


def test_add_landmark_factor_has_zero_error_at_the_exact_solution() -> None:
    pose = gtsam.Pose2(1.0, 2.0, 0.4)
    landmark = np.array([6.0, 5.0])
    measurement, _, _ = predict_measurement(pose, landmark)

    graph = gtsam.NonlinearFactorGraph()
    add_landmark_factor(
        graph, X(0), L(0), measurement, bearing_range_noise_model(SlamConfig())
    )

    values = gtsam.Values()
    values.insert(X(0), pose)
    values.insert(L(0), gtsam.Point2(*landmark))

    assert graph.size() == 1
    assert list(graph.at(0).keys()) == [X(0), L(0)]
    assert graph.error(values) == pytest.approx(0.0, abs=1e-12)


def test_add_landmark_factor_treats_the_measurement_as_range_then_bearing() -> None:
    """The measurement array is [range, bearing]; the factor wants the reverse.

    Feeding the array straight through in the wrong order still produces a
    perfectly runnable system, which is why this is checked explicitly.
    """
    pose = gtsam.Pose2(0.0, 0.0, 0.0)

    graph = gtsam.NonlinearFactorGraph()
    # range 4, bearing 0.3 rad: if the two are swapped, the factor believes the
    # landmark is 0.3 m away at a bearing of 4 rad.
    add_landmark_factor(
        graph, X(0), L(0), np.array([4.0, 0.3]), bearing_range_noise_model(SlamConfig())
    )

    values = gtsam.Values()
    values.insert(X(0), pose)
    values.insert(L(0), gtsam.Point2(4.0 * np.cos(0.3), 4.0 * np.sin(0.3)))

    assert graph.error(values) == pytest.approx(0.0, abs=1e-9)


def test_landmark_noise_model_is_ordered_bearing_then_range() -> None:
    config = _config_with(sigma_range=0.5, sigma_bearing_deg=np.rad2deg(0.02))

    pose = gtsam.Pose2(0.0, 0.0, 0.0)
    landmark = np.array([10.0, 0.0])
    measurement, _, _ = predict_measurement(pose, landmark)

    graph = gtsam.NonlinearFactorGraph()
    add_landmark_factor(graph, X(0), L(0), measurement, bearing_range_noise_model(config))

    # Move the landmark 1 m further away: a pure range error of 1 m.
    values = gtsam.Values()
    values.insert(X(0), pose)
    values.insert(L(0), gtsam.Point2(11.0, 0.0))

    expected = 0.5 * (1.0 / config.noise.sigma_range) ** 2

    assert graph.error(values) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# (d) predict_measurement
# ---------------------------------------------------------------------------


def test_predict_measurement_values() -> None:
    pose = gtsam.Pose2(1.0, 2.0, 0.4)
    landmark = np.array([4.0, 6.0])

    z, _, _ = predict_measurement(pose, landmark)

    delta = landmark - np.array([pose.x(), pose.y()])
    expected_range = np.linalg.norm(delta)
    expected_bearing = ssa(np.arctan2(delta[1], delta[0]) - pose.theta())

    assert z[0] == pytest.approx(expected_range)
    assert ssa(z[1] - expected_bearing) == pytest.approx(0.0, abs=1e-12)


def test_predict_measurement_shapes() -> None:
    z, H_pose, H_landmark = predict_measurement(gtsam.Pose2(0.0, 0.0, 0.0), np.array([3.0, 1.0]))

    assert z.shape == (2,)
    assert H_pose.shape == (2, 3)
    assert H_landmark.shape == (2, 2)


@pytest.mark.parametrize(
    "pose_values, landmark",
    [
        ((0.0, 0.0, 0.0), (5.0, 0.0)),
        ((1.0, 2.0, 0.4), (4.0, 6.0)),
        ((-3.0, 1.5, -1.2), (2.0, -4.0)),
    ],
)
def test_predict_measurement_jacobians_match_numerical_differentiation(
    pose_values, landmark
) -> None:
    pose = gtsam.Pose2(*pose_values)
    landmark = np.array(landmark)

    _, H_pose, H_landmark = predict_measurement(pose, landmark)

    def measurement_of_pose(p):
        z, _, _ = predict_measurement(p, landmark)
        return z

    def measurement_of_landmark(m):
        z, _, _ = predict_measurement(pose, m)
        return z

    np.testing.assert_allclose(
        H_pose, numerical_jacobian_pose(measurement_of_pose, pose, 2), atol=1e-6
    )
    np.testing.assert_allclose(
        H_landmark, numerical_jacobian_point(measurement_of_landmark, landmark, 2), atol=1e-6
    )


# ---------------------------------------------------------------------------
# (g1) inverse_measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pose_values, landmark",
    [
        ((0.0, 0.0, 0.0), (5.0, 0.0)),
        ((1.0, 2.0, 0.4), (4.0, 6.0)),
        ((-3.0, 1.5, -1.2), (2.0, -4.0)),
    ],
)
def test_inverse_measurement_inverts_predict_measurement(pose_values, landmark) -> None:
    pose = gtsam.Pose2(*pose_values)
    landmark = np.array(landmark)

    z, _, _ = predict_measurement(pose, landmark)
    recovered = inverse_measurement(pose, z)

    assert np.asarray(recovered).shape == (2,)
    np.testing.assert_allclose(recovered, landmark, atol=1e-9)


# ---------------------------------------------------------------------------
# (e) reorder_joint_covariance
# ---------------------------------------------------------------------------


def _labelled_block_matrix(dims, order):
    """Block matrix whose (i, j) block is filled with the constant ``10*a + b``.

    ``order`` lists the block labels in the order they appear in the matrix, so
    the same logical covariance can be laid out in any block order and the
    blocks stay identifiable.
    """
    total = int(np.sum([dims[label] for label in order]))
    offsets = np.concatenate([[0], np.cumsum([dims[label] for label in order])]).astype(int)

    M = np.zeros((total, total))
    for i, label_i in enumerate(order):
        for j, label_j in enumerate(order):
            value = 10 * min(label_i, label_j) + max(label_i, label_j)
            M[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = value
    return M


def test_reorder_joint_covariance_moves_the_pose_block_to_the_front() -> None:
    """GTSAM sorts by key value, and every L key sorts before every X key."""
    keys = [X(7), L(3), L(1)]
    dims_by_label = {1: 3, 2: 2, 3: 2}  # label 1 is the pose block

    # What we asked for: pose, L(3), L(1).
    wanted = _labelled_block_matrix(dims_by_label, order=[1, 2, 3])

    # What GTSAM hands back: ascending key, i.e. L(1), L(3), X(7).
    as_returned = _labelled_block_matrix(dims_by_label, order=[3, 2, 1])

    reordered = reorder_joint_covariance(as_returned, keys, [3, 2, 2])

    np.testing.assert_allclose(reordered, wanted)
    # The pose block really is the leading 3x3.
    np.testing.assert_allclose(reordered[0:3, 0:3], np.full((3, 3), 11.0))


def test_reorder_joint_covariance_is_the_identity_when_already_sorted() -> None:
    keys = [L(1), L(2), L(5)]
    dims = [2, 2, 2]
    covariance = np.arange(36.0).reshape(6, 6)
    covariance = covariance + covariance.T

    np.testing.assert_allclose(reorder_joint_covariance(covariance, keys, dims), covariance)


def test_reorder_joint_covariance_preserves_symmetry_and_trace() -> None:
    keys = [X(4), L(9), L(2), L(11)]
    dims = [3, 2, 2, 2]

    rng = np.random.default_rng(0)
    A = rng.normal(size=(9, 9))
    covariance = A @ A.T

    reordered = reorder_joint_covariance(covariance, keys, dims)

    np.testing.assert_allclose(reordered, reordered.T, atol=1e-12)
    assert np.trace(reordered) == pytest.approx(np.trace(covariance))


def test_reorder_joint_covariance_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        reorder_joint_covariance(np.eye(5), [X(0), L(0)], [3, 3])


# ---------------------------------------------------------------------------
# (f) innovation_covariance
# ---------------------------------------------------------------------------


def test_innovation_covariance_single_landmark() -> None:
    H_pose = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.2]])
    H_landmark = np.array([[-1.0, 0.0], [0.0, -1.0]])

    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    P = A @ A.T
    R = np.diag([0.04, 0.0003])

    S = innovation_covariance([H_pose], [H_landmark], P, R)

    H = np.hstack((H_pose, H_landmark))
    np.testing.assert_allclose(S, H @ P @ H.T + R, atol=1e-12)


def test_innovation_covariance_shape_and_noise_term() -> None:
    n = 3
    rng = np.random.default_rng(2)

    H_poses = [rng.normal(size=(2, 3)) for _ in range(n)]
    H_landmarks = [rng.normal(size=(2, 2)) for _ in range(n)]
    P = np.zeros((3 + 2 * n, 3 + 2 * n))
    R = np.diag([0.04, 0.0003])

    S = innovation_covariance(H_poses, H_landmarks, P, R)

    assert S.shape == (2 * n, 2 * n)
    # With P = 0 the only thing left is the block diagonal measurement noise.
    np.testing.assert_allclose(S, np.kron(np.eye(n), R), atol=1e-12)


def test_innovation_covariance_is_block_diagonal_only_without_pose_uncertainty() -> None:
    """Shared pose uncertainty is what correlates the innovations.

    If the off-diagonal blocks of S vanish, the joint test in JCBB degenerates
    into a sequence of individual tests.
    """
    n = 2
    H_poses = [np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) for _ in range(n)]
    H_landmarks = [np.eye(2) for _ in range(n)]
    R = np.diag([0.04, 0.0003])

    no_pose_uncertainty = np.zeros((3 + 2 * n, 3 + 2 * n))
    no_pose_uncertainty[3:, 3:] = np.eye(2 * n) * 0.1

    with_pose_uncertainty = no_pose_uncertainty.copy()
    with_pose_uncertainty[0:3, 0:3] = np.eye(3) * 0.2

    S_independent = innovation_covariance(H_poses, H_landmarks, no_pose_uncertainty, R)
    S_correlated = innovation_covariance(H_poses, H_landmarks, with_pose_uncertainty, R)

    np.testing.assert_allclose(S_independent[0:2, 2:4], np.zeros((2, 2)), atol=1e-12)
    assert np.abs(S_correlated[0:2, 2:4]).max() > 1e-6


def test_innovation_covariance_handles_an_empty_local_map() -> None:
    S = innovation_covariance([], [], np.zeros((3, 3)), np.eye(2))
    assert S.shape == (0, 0)
