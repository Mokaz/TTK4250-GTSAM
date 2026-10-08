"""Task 1 (c) graph construction, (d) measurement model, (e) joint
covariance, (f) innovation covariance and (g1) inverse measurement."""

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
    LocalMap,
    assemble_joint_covariance,
    innovation_covariance,
    inverse_measurement,
    predict_measurement,
    predict_pose,
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


def _exact_measurement(pose: gtsam.Pose2, landmark: np.ndarray) -> np.ndarray:
    """[range, bearing] straight from GTSAM, so these tests do not lean on (d)."""
    point = gtsam.Point2(*landmark)
    return np.array([pose.range(point), pose.bearing(point).theta()])


def _config_with(**noise_overrides) -> SlamConfig:
    config = SlamConfig()
    for name, value in noise_overrides.items():
        setattr(config.noise, name, value)
    return config


def test_add_landmark_factor_has_zero_error_at_the_exact_solution() -> None:
    pose = gtsam.Pose2(1.0, 2.0, 0.4)
    landmark = np.array([6.0, 5.0])
    measurement = _exact_measurement(pose, landmark)

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
    measurement = _exact_measurement(pose, landmark)

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
# (e) assemble_joint_covariance
# ---------------------------------------------------------------------------


class _LabelledJointMarginal:
    """Stands in for a gtsam.JointMarginal: block (a, b) is filled with 10*a + b.

    Each key is labelled 1, 2, 3, ... so every block, and which way round it is,
    can be identified in the assembled matrix.
    """

    def __init__(self, labels: dict[int, int], dims: dict[int, int]) -> None:
        self.labels, self.dims = labels, dims

    def at(self, key_i: int, key_j: int) -> np.ndarray:
        value = 10 * self.labels[key_i] + self.labels[key_j]
        return np.full((self.dims[key_i], self.dims[key_j]), float(value))


def _labelled_marginal():
    labels = {X(7): 1, L(3): 2, L(1): 3}
    dims = {X(7): 3, L(3): 2, L(1): 2}
    return _LabelledJointMarginal(labels, dims)


def test_assemble_joint_covariance_puts_the_pose_block_first() -> None:
    P = assemble_joint_covariance(_labelled_marginal(), [X(7), L(3), L(1)])

    assert P.shape == (7, 7)
    np.testing.assert_allclose(P[0:3, 0:3], 11.0, err_msg="the pose block must come first")


def test_assemble_joint_covariance_follows_the_order_of_keys() -> None:
    """Block (i, j) is the covariance between keys[i] and keys[j], whatever the order."""
    P = assemble_joint_covariance(_labelled_marginal(), [X(7), L(1), L(3)])

    np.testing.assert_allclose(P[3:5, 3:5], 33.0)  # L(1) with itself
    np.testing.assert_allclose(P[5:7, 5:7], 22.0)  # L(3) with itself
    np.testing.assert_allclose(P[0:3, 3:5], 13.0)  # pose with L(1)
    np.testing.assert_allclose(P[3:5, 5:7], 32.0)  # L(1) with L(3)


def test_assemble_joint_covariance_keeps_the_landmark_cross_blocks() -> None:
    """The landmark-landmark blocks are what makes JCBB joint; they must not be zero."""
    P = assemble_joint_covariance(_labelled_marginal(), [X(7), L(3), L(1)])

    np.testing.assert_allclose(P[3:5, 5:7], 23.0, err_msg="missing landmark-landmark cross block")
    np.testing.assert_allclose(P[5:7, 3:5], 32.0, err_msg="missing landmark-landmark cross block")


def _small_isam2_problem() -> tuple[gtsam.ISAM2, gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Four poses and three landmarks, solved by iSAM2 with no lazy relinearization.

    Each landmark is seen from a different subset of poses, so the landmark
    blocks of the joint covariance all differ and a block in the wrong place
    cannot pass by coincidence.
    """
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.02]))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.1, 0.05]))
    measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.3]))  # bearing, range

    poses = [gtsam.Pose2(2.0 * k, 0.0, 0.1 * k) for k in range(4)]
    landmarks = {0: np.array([3.0, 4.0]), 1: np.array([5.0, -3.0]), 2: np.array([8.0, 2.0])}
    sightings = {0: [0, 1], 1: [1, 2, 3], 2: [3]}

    graph.add(gtsam.PriorFactorPose2(X(0), poses[0], prior_noise))
    for k, pose in enumerate(poses):
        values.insert(X(k), pose)
        if k > 0:
            graph.add(gtsam.BetweenFactorPose2(X(k - 1), X(k), poses[k - 1].between(pose), odometry_noise))

    for j, landmark in landmarks.items():
        values.insert(L(j), landmark)
        for k in sightings[j]:
            graph.add(
                gtsam.BearingRangeFactor2D(
                    X(k), L(j), poses[k].bearing(landmark), poses[k].range(landmark), measurement_noise
                )
            )

    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(0.0)
    params.relinearizeSkip = 1
    isam2 = gtsam.ISAM2(params)
    isam2.update(graph, values)

    return isam2, graph, isam2.calculateEstimate()


@pytest.mark.parametrize("route", ["bayes_tree", "marginals"])
def test_assemble_joint_covariance_matches_gtsam_block_by_block(route: str) -> None:
    """With a real gtsam.JointMarginal from either GTSAM route.

    Every block of the assembled P must match the block GTSAM reports for that
    pair of keys, with the landmarks deliberately asked for out of key order.
    """
    isam2, graph, estimate = _small_isam2_problem()
    keys = [X(3), L(2), L(0), L(1)]
    query = gtsam.KeyVector(keys)

    if route == "bayes_tree":
        joint_marginal = isam2.jointMarginalCovariance(query)
    else:
        joint_marginal = gtsam.Marginals(graph, estimate).jointMarginalCovariance(query)

    P = assemble_joint_covariance(joint_marginal, keys)

    reference = gtsam.Marginals(graph, estimate).jointMarginalCovariance(gtsam.KeyVector(sorted(keys)))
    dims = [3, 2, 2, 2]
    offsets = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    assert P.shape == (9, 9)
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            np.testing.assert_allclose(
                P[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]],
                reference.at(key_i, key_j),
                atol=1e-8,
                err_msg=(
                    f"{route}: block ({gtsam.Symbol(key_i).string()}, "
                    f"{gtsam.Symbol(key_j).string()}) does not match GTSAM. The pose "
                    "block must come first, then the landmarks in the order of keys."
                ),
            )


@pytest.mark.parametrize(("solver", "method"), [("isam2", "bayes_tree"), ("isam2", "marginals"), ("batch", "bayes_tree")])
def test_the_back_end_returns_P_with_the_pose_block_first(solver: str, method: str) -> None:
    """The given back-end, end to end: GTSAM's query plus your assembly.

    Guards against a covariance whose blocks are in a different order than the
    Jacobians of (f): a misordered P still gives a valid-looking S, just a wrong
    one, and nothing crashes.
    """
    from graphslam.slam import Backend

    _, graph, values = _small_isam2_problem()
    config = SlamConfig()
    config.backend.solver = solver
    config.backend.covariance_method = method
    backend = Backend(config)
    backend.update(graph, values)

    local_map = LocalMap(keys=[L(2), L(0), L(1)])
    P = backend.joint_covariance(X(3), local_map)

    np.testing.assert_allclose(P[0:3, 0:3], backend.marginal_covariance(X(3)), atol=1e-8)
    np.testing.assert_allclose(P[3:5, 3:5], backend.marginal_covariance(L(2)), atol=1e-8)
    np.testing.assert_allclose(P[7:9, 7:9], backend.marginal_covariance(L(1)), atol=1e-8)


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
