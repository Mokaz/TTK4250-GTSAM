"""Factor graph construction, measurement prediction and covariance recovery.

This is the heart of the assignment. Everything marked ``TODO`` below is graded
by the test suite.

Where EKF-SLAM had a ``predict`` and an ``update``, a factor graph SLAM system
has three jobs instead:

1. **Build the graph.** Every factor you add is literally one term of the
   negative log posterior (9.6) in the book. The back-end (iSAM2) then finds the
   MAP estimate by Gauss-Newton on the whole trajectory and map at once.
2. **Recover a covariance.** The EKF handed you ``P`` for free. A factor graph
   stores information, not covariance, so the joint marginal over the current
   pose and the nearby landmarks must be *recovered* from the Bayes tree before
   anything can be gated. This is Sec. 9.4 of the book, and it is the single
   biggest practical difference between the two approaches.
3. **Decide what is what.** Data association happens outside the back-end
   (Sec. 9.1, front-end vs. back-end), using the covariance from step 2.

A note on Jacobians and frames that will save you an afternoon: the covariance
GTSAM gives you for a ``Pose2`` lives in the *tangent space* at the current
estimate, not in raw ``[x, y, theta]`` coordinates. The measurement Jacobians
returned by ``Pose2.range`` and ``Pose2.bearing`` use the same convention, so as
long as you use GTSAM's Jacobians together with GTSAM's covariance you are
consistent. Mixing them with a hand-derived ``d/d[x,y,theta]`` Jacobian is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np

from graphslam.config import SlamConfig

# ---------------------------------------------------------------------------
# Noise models  (given)
# ---------------------------------------------------------------------------


def prior_noise_model(config: SlamConfig) -> gtsam.noiseModel.Base:
    """Noise model for the prior on X(0), ordered [x, y, yaw]."""
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array(
            [
                config.noise.sigma_init_pose_x,
                config.noise.sigma_init_pose_y,
                config.noise.sigma_init_pose_yaw_rad,
            ]
        )
    )


def bearing_range_noise_model(config: SlamConfig) -> gtsam.noiseModel.Base:
    """Noise model for landmark observations.

    Careful: ``BearingRangeFactor2D`` expects **bearing first, then range**,
    while ``config.noise.range_bearing_cov_matrix`` and every measurement array
    in this code base are ordered **[range, bearing]**. Getting this backwards
    produces a system that still runs, still looks plausible, and is quietly
    wrong -- so the test suite checks it.
    """
    base = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([config.noise.sigma_bearing_rad, config.noise.sigma_range])
    )

    if config.backend.robust_kernel == "huber":
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(config.backend.huber_k),
            base,
        )

    return base


# ---------------------------------------------------------------------------
# Task 1 (c): growing the graph with odometry
# ---------------------------------------------------------------------------


def predict_pose(previous_pose: gtsam.Pose2, relative_pose: gtsam.Pose2) -> gtsam.Pose2:
    """Compose a relative pose increment onto the previous pose.

    This is the inverse of the ``(-)`` operator in (9.5): the odometry factor
    penalises ``x_k (-) x_{k-1}`` against the measured increment, so the natural
    initial guess for ``x_k`` is ``x_{k-1}`` composed with that increment.

    Parameters
    ----------
    previous_pose : gtsam.Pose2
        The current estimate of the previous pose, x_{k-1}.
    relative_pose : gtsam.Pose2
        The measured body-frame increment, u_k.

    Returns
    -------
    gtsam.Pose2
        The predicted pose x_k, used as the initial guess handed to the solver.
    """
    # TODO(c1): compose the relative pose onto the previous pose.
    # BEGIN SOLUTION
    return previous_pose.compose(relative_pose)
    # END SOLUTION


def add_odometry_factor(
    graph: gtsam.NonlinearFactorGraph,
    key_previous: int,
    key_current: int,
    relative_pose: gtsam.Pose2,
    relative_pose_cov: np.ndarray,
) -> None:
    """Add the odometry term of (9.6) to the graph.

    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
        Graph of new factors, to be handed to the solver.
    key_previous, key_current : int
        Keys of x_{k-1} and x_k, i.e. ``X(k-1)`` and ``X(k)``.
    relative_pose : gtsam.Pose2
        Measured (preintegrated) increment.
    relative_pose_cov : np.ndarray, shape=(3, 3)
        Covariance of that increment, in the tangent space of ``relative_pose``.

    Notes
    -----
    Use ``gtsam.noiseModel.Gaussian.Covariance`` and not ``.Sigmas``: the
    preintegrated covariance from Task 1 (b) is *not* diagonal, and throwing
    away its off-diagonal terms is one of the easier ways to end up with an
    over-confident, inconsistent system.
    """
    # TODO(c2): add a BetweenFactorPose2 with a full-covariance Gaussian noise model.
    # BEGIN SOLUTION
    noise = gtsam.noiseModel.Gaussian.Covariance(relative_pose_cov)
    graph.add(
        gtsam.BetweenFactorPose2(
            key_previous,
            key_current,
            relative_pose,
            noise,
        )
    )
    # END SOLUTION


def add_landmark_factor(
    graph: gtsam.NonlinearFactorGraph,
    pose_key: int,
    landmark_key: int,
    measurement: np.ndarray,
    noise_model: gtsam.noiseModel.Base,
) -> None:
    """Add the landmark measurement term of (9.6) to the graph.

    Parameters
    ----------
    graph : gtsam.NonlinearFactorGraph
    pose_key, landmark_key : int
        Keys of the pose the measurement was taken from and of the landmark it
        was associated to.
    measurement : np.ndarray, shape=(2,)
        The measurement **ordered [range, bearing]**.
    noise_model : gtsam.noiseModel.Base
        From :func:`bearing_range_noise_model`, ordered [bearing, range].
    """
    # TODO(c3): add a BearingRangeFactor2D. Mind the argument order.
    # BEGIN SOLUTION
    measured_range, measured_bearing = float(measurement[0]), float(measurement[1])
    graph.add(
        gtsam.BearingRangeFactor2D(
            pose_key,
            landmark_key,
            gtsam.Rot2(measured_bearing),
            measured_range,
            noise_model,
        )
    )
    # END SOLUTION


# ---------------------------------------------------------------------------
# Task 1 (d): the measurement model and its Jacobians
# ---------------------------------------------------------------------------


def predict_measurement(
    pose: gtsam.Pose2,
    landmark: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the range-bearing measurement of one landmark, with Jacobians.

    This is ``h(x_k, m_j)`` from (9.6)/(9.7), plus the two blocks of the
    Jacobian ``J`` in (9.8) that belong to this pose and this landmark.

    Parameters
    ----------
    pose : gtsam.Pose2
        Current pose estimate x_k.
    landmark : np.ndarray, shape=(2,)
        Landmark position m_j in the world frame.

    Returns
    -------
    z : np.ndarray, shape=(2,)
        Predicted measurement, **ordered [range, bearing]**.
    H_pose : np.ndarray, shape=(2, 3)
        d z / d x_k, in the tangent space at ``pose``.
    H_landmark : np.ndarray, shape=(2, 2)
        d z / d m_j.

    Notes
    -----
    ``gtsam.Pose2.range`` and ``gtsam.Pose2.bearing`` both take two optional
    Jacobian arguments which they fill in for you. They must be preallocated
    with the right shape *and* in Fortran order, e.g.::

        H = np.zeros((1, 3), order="F")

    ``bearing`` returns a ``gtsam.Rot2``; take ``.theta()`` to get the angle.
    """
    # TODO(d): predict [range, bearing] and stack the two rows of each Jacobian.
    # BEGIN SOLUTION
    landmark = gtsam.Point2(float(landmark[0]), float(landmark[1]))

    H_range_pose = np.zeros((1, 3), order="F")
    H_range_landmark = np.zeros((1, 2), order="F")
    H_bearing_pose = np.zeros((1, 3), order="F")
    H_bearing_landmark = np.zeros((1, 2), order="F")

    predicted_range = pose.range(landmark, H_range_pose, H_range_landmark)
    predicted_bearing = pose.bearing(landmark, H_bearing_pose, H_bearing_landmark).theta()

    z = np.array([predicted_range, predicted_bearing])
    H_pose = np.vstack((H_range_pose, H_bearing_pose))
    H_landmark = np.vstack((H_range_landmark, H_bearing_landmark))

    return z, H_pose, H_landmark
    # END SOLUTION


def inverse_measurement(pose: gtsam.Pose2, measurement: np.ndarray) -> np.ndarray:
    """Back-project a range-bearing measurement to a world-frame position.

    This is ``h^{-1}(z, x)``: the graph analogue of ``add_landmarks`` in the
    EKF-SLAM assignment. Unlike the EKF version you do *not* have to propagate a
    covariance -- the graph works that out for itself once the landmark is
    connected by factors, which is one of the nicer consequences of (9.2).

    Parameters
    ----------
    pose : gtsam.Pose2
        Pose the measurement was taken from.
    measurement : np.ndarray, shape=(2,)
        Measurement **ordered [range, bearing]**.

    Returns
    -------
    np.ndarray, shape=(2,)
        Landmark position in the world frame.
    """
    # TODO(g1): rotate by the bearing, translate by the range, map to the world frame.
    # BEGIN SOLUTION
    measured_range, measured_bearing = float(measurement[0]), float(measurement[1])
    landmark_body = gtsam.Rot2(measured_bearing).rotate(gtsam.Point2(measured_range, 0.0))
    landmark_world = pose.transformFrom(landmark_body)
    return np.asarray(landmark_world, dtype=float).reshape(2)
    # END SOLUTION


# ---------------------------------------------------------------------------
# Task 1 (e): recovering a joint marginal covariance from the graph
# ---------------------------------------------------------------------------


def reorder_joint_covariance(
    covariance: np.ndarray,
    keys: list[int],
    dims: list[int],
) -> np.ndarray:
    """Permute a GTSAM joint marginal covariance into the order *you* asked for.

    GTSAM returns the blocks of a joint marginal ordered by **ascending key
    value**, not in the order the keys were requested. Because
    ``gtsam.symbol_shorthand.L`` encodes the character ``'l'`` (108) and ``X``
    encodes ``'x'`` (120), every landmark key sorts *before* every pose key --
    so a query for ``[X(k), L(3), L(7)]`` comes back ordered
    ``[L(3), L(7), X(k)]``. Stack a measurement Jacobian against that without
    noticing and you get an innovation covariance that is subtly, silently
    wrong.

    Parameters
    ----------
    covariance : np.ndarray, shape=(D, D)
        Joint covariance as returned by GTSAM, in ascending-key order.
    keys : list[int]
        The keys in the order you want them, e.g. ``[X(k), L(3), L(7)]``.
    dims : list[int]
        Dimension of each key in ``keys`` (3 for a Pose2, 2 for a Point2).

    Returns
    -------
    np.ndarray, shape=(D, D)
        The same covariance with its blocks reordered to match ``keys``.
    """
    # TODO(e): build the index permutation and apply it to both rows and columns.
    # BEGIN SOLUTION
    if len(keys) != len(dims):
        raise ValueError(f"keys and dims must have equal length, got {len(keys)} and {len(dims)}")

    total_dim = int(np.sum(dims))
    if covariance.shape != (total_dim, total_dim):
        raise ValueError(
            f"covariance has shape {covariance.shape}, expected ({total_dim}, {total_dim})"
        )

    # Position of each requested key once the keys are sorted ascending, which
    # is the order GTSAM used when it laid out the returned matrix.
    sorted_positions = np.argsort(np.asarray(keys, dtype=np.uint64), kind="stable")

    # Offset of each block inside the returned matrix.
    offsets = np.zeros(len(keys), dtype=int)
    offset = 0
    for position in sorted_positions:
        offsets[position] = offset
        offset += dims[position]

    permutation = np.concatenate(
        [np.arange(offsets[i], offsets[i] + dims[i]) for i in range(len(keys))]
    ).astype(int)

    return covariance[np.ix_(permutation, permutation)]
    # END SOLUTION


# ---------------------------------------------------------------------------
# Task 1 (f): the innovation covariance
# ---------------------------------------------------------------------------


def innovation_covariance(
    jacobians_pose: list[np.ndarray],
    jacobians_landmark: list[np.ndarray],
    joint_covariance: np.ndarray,
    measurement_cov: np.ndarray,
) -> np.ndarray:
    """Assemble ``S = H P H^T + R`` for the whole local map at once.

    ``P`` is the joint marginal over ``[x_k, m_{j1}, ..., m_{jn}]`` recovered in
    Task 1 (e), so it has shape ``(3 + 2n, 3 + 2n)`` and, crucially, it contains
    the **cross-covariances between landmarks**. Those cross terms are the whole
    reason JCBB can be joint rather than a sequence of independent gates: they
    are what makes a set of individually plausible associations collectively
    implausible.

    Parameters
    ----------
    jacobians_pose : list of np.ndarray, each shape=(2, 3)
        ``H_pose`` from :func:`predict_measurement`, one per local landmark, in
        the same order as the landmark blocks of ``joint_covariance``.
    jacobians_landmark : list of np.ndarray, each shape=(2, 2)
        ``H_landmark`` from :func:`predict_measurement`, same order.
    joint_covariance : np.ndarray, shape=(3 + 2n, 3 + 2n)
        Joint marginal ``P``, pose block first.
    measurement_cov : np.ndarray, shape=(2, 2)
        Single-measurement ``R``, ordered [range, bearing].

    Returns
    -------
    np.ndarray, shape=(2n, 2n)
        The innovation covariance ``S``.

    Notes
    -----
    ``H`` is dense in its three leftmost columns (every measurement depends on
    the pose) and block diagonal elsewhere (measurement ``i`` only depends on
    landmark ``i``) -- the same structure as the EKF-SLAM measurement Jacobian,
    for the same reason.
    """
    # TODO(f): build the stacked H, build the block-diagonal R, and form S.
    # BEGIN SOLUTION
    n = len(jacobians_pose)
    if len(jacobians_landmark) != n:
        raise ValueError(
            "jacobians_pose and jacobians_landmark must have equal length, "
            f"got {n} and {len(jacobians_landmark)}"
        )

    if n == 0:
        return np.zeros((0, 0))

    H = np.zeros((2 * n, 3 + 2 * n))
    for i in range(n):
        H[2 * i : 2 * i + 2, 0:3] = jacobians_pose[i]
        H[2 * i : 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = jacobians_landmark[i]

    R = np.kron(np.eye(n), measurement_cov)

    return H @ joint_covariance @ H.T + R
    # END SOLUTION


# ---------------------------------------------------------------------------
# Local map extraction  (given -- builds on your predict_measurement)
# ---------------------------------------------------------------------------


@dataclass
class LocalMap:
    """The landmarks currently in view, and everything needed to gate them."""

    keys: list[int] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    predicted_measurements: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    jacobians_pose: list[np.ndarray] = field(default_factory=list)
    jacobians_landmark: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)


def extract_local_map(
    pose: gtsam.Pose2,
    landmark_keys: list[int],
    landmark_positions: dict[int, np.ndarray],
    config: SlamConfig,
) -> LocalMap:
    """Select the landmarks that could plausibly have produced a measurement.

    Restricting association to a local window is what keeps the covariance
    recovery in Task 1 (e) affordable: its cost grows with the number of keys in
    the query, not with the size of the map.
    """
    local = LocalMap()
    positions: list[np.ndarray] = []
    predicted: list[np.ndarray] = []

    for key in landmark_keys:
        landmark = landmark_positions[key]
        z, H_pose, H_landmark = predict_measurement(pose, landmark)
        predicted_range, predicted_bearing = z

        in_range = predicted_range < config.sensor.range_local
        in_field_of_view = abs(predicted_bearing) < config.sensor.bearing_local_rad

        if in_range and in_field_of_view:
            local.keys.append(key)
            positions.append(np.asarray(landmark, dtype=float).reshape(2))
            predicted.append(z)
            local.jacobians_pose.append(H_pose)
            local.jacobians_landmark.append(H_landmark)

    local.positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    local.predicted_measurements = np.asarray(predicted, dtype=float).reshape(-1, 2)

    return local


# ---------------------------------------------------------------------------
# Raw covariance queries  (given -- these differ between GTSAM builds)
# ---------------------------------------------------------------------------


_COVARIANCE_METHOD_CACHE: str | None = None


def _query_bayes_tree(isam2: gtsam.ISAM2, keys: list[int]) -> np.ndarray:
    """Fastest path: ask the Bayes tree directly (Sec. 9.5)."""
    return isam2.jointMarginalCovariance(gtsam.KeyVector(keys)).fullMatrix()


def _query_marginals(isam2: gtsam.ISAM2, keys: list[int]) -> np.ndarray:
    """Portable path: rebuild a Marginals object over the whole graph."""
    marginals = gtsam.Marginals(isam2.getFactorsUnsafe(), isam2.calculateEstimate())
    return marginals.jointMarginalCovariance(gtsam.KeyVector(keys)).fullMatrix()


def _query_elimination(isam2: gtsam.ISAM2, keys: list[int]) -> np.ndarray:
    """Explicit path: linearize, marginalize, invert the Hessian (Sec. 9.4.1)."""
    linear_graph = isam2.getFactorsUnsafe().linearize(isam2.getLinearizationPoint())
    marginal_graph = linear_graph.marginal(gtsam.KeyVector(keys))
    hessian, _ = marginal_graph.hessian()
    return np.linalg.inv(hessian)


_COVARIANCE_QUERIES = {
    "bayes_tree": _query_bayes_tree,
    "marginals": _query_marginals,
    "elimination": _query_elimination,
}


def query_joint_covariance(
    isam2: gtsam.ISAM2,
    keys: list[int],
    method: str = "auto",
) -> np.ndarray:
    """Recover the joint marginal covariance over ``keys`` from the back-end.

    The blocks come back in **ascending key order** -- see
    :func:`reorder_joint_covariance`.

    ``method="auto"`` probes once for the fast Bayes-tree query (which needs a
    recent GTSAM) and falls back to the portable ``Marginals`` path otherwise.
    The three methods compute the same quantity by different routes; comparing
    their cost on Victoria Park is a worthwhile experiment in its own right.
    """
    global _COVARIANCE_METHOD_CACHE

    if method != "auto":
        return _COVARIANCE_QUERIES[method](isam2, keys)

    if _COVARIANCE_METHOD_CACHE is None:
        for candidate in ("bayes_tree", "marginals", "elimination"):
            try:
                result = _COVARIANCE_QUERIES[candidate](isam2, keys)
            except Exception:  # noqa: BLE001 - probing for API availability
                continue
            _COVARIANCE_METHOD_CACHE = candidate
            print(f"[covariance] using the '{candidate}' recovery method")
            return result
        raise RuntimeError(
            "No working joint-covariance recovery method found for this GTSAM build."
        )

    return _COVARIANCE_QUERIES[_COVARIANCE_METHOD_CACHE](isam2, keys)


def local_joint_covariance(
    isam2: gtsam.ISAM2,
    pose_key: int,
    local_map: LocalMap,
    config: SlamConfig,
) -> np.ndarray:
    """Joint marginal over ``[pose] + local landmarks``, in that order.

    Thin wrapper that ties :func:`query_joint_covariance` (given) together with
    :func:`reorder_joint_covariance` (yours).
    """
    keys = [pose_key] + list(local_map.keys)
    dims = [3] + [2] * len(local_map)

    covariance = query_joint_covariance(isam2, keys, config.backend.covariance_method)

    return reorder_joint_covariance(covariance, keys, dims)
