"""Data association: JCBB and a ground-truth associator.

Nothing in this file is graded -- it is handed to you the same way JCBB was
handed to you in the EKF-SLAM assignment. It is worth reading anyway, because
the quantity it consumes is the one you build in Task 1 (e) and (f): the
innovation covariance ``S`` of the *whole local map jointly*, cross-covariances
included. Individual compatibility only looks at the 2x2 diagonal blocks of
``S``; joint compatibility is what the off-diagonal blocks buy you.

The branch and bound search is described in Sec. 7.3.1 and 7.3.2 of the book.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

from graphslam.config import SlamConfig
from graphslam.utils import ssa

chi2isf_cached = lru_cache(maxsize=None)(chi2.isf)


# ---------------------------------------------------------------------------
# JCBB
# ---------------------------------------------------------------------------


def JCBB_association(
    z: np.ndarray,
    zbar: np.ndarray,
    S: np.ndarray,
    alpha_individual: float,
    alpha_joint: float,
) -> np.ndarray:
    """Joint compatibility branch and bound.

    Parameters
    ----------
    z : np.ndarray, shape=(M, 2)
        Measurements, [range, bearing].
    zbar : np.ndarray, shape=(L, 2)
        Predicted measurements for the local map, [range, bearing].
    S : np.ndarray, shape=(2L, 2L)
        Joint innovation covariance from Task 1 (f).
    alpha_individual, alpha_joint : float
        Confidence levels of the individual and joint compatibility tests.

    Returns
    -------
    np.ndarray, shape=(M,), dtype=int
        ``a[i] >= 0`` is an index into ``zbar``; ``a[i] == -1`` means the
        measurement was left unassociated (a new landmark, or clutter).
    """
    L = zbar.shape[0]
    M = z.shape[0]

    if M == 0:
        return np.array([], dtype=int)
    if L == 0:
        return np.full(M, -1, dtype=int)

    if S.shape != (2 * L, 2 * L):
        raise ValueError(f"S must have shape ({2 * L}, {2 * L}) in JCBB, got {S.shape}")

    a = np.full(M, -1, dtype=int)
    a_best = np.full(M, -1, dtype=int)

    # Rows are measurements, columns are predicted measurements.
    ic = individual_compatibility(z, zbar, S)
    g2 = chi2.isf(1 - alpha_individual, 2)

    # Associate the least ambiguous measurements first: it prunes harder.
    order = np.argsort(np.amin(ic, axis=1))
    z_ordered = z[order]
    ic_ordered = ic[order]

    a_best_ordered = _jcbb_recursive(z_ordered, zbar, S, alpha_joint, g2, 0, a, ic_ordered, a_best)
    a_best[order] = a_best_ordered

    return a_best


def _jcbb_recursive(z, zbar, S, alpha_joint, g2, j, a, ic, abest):
    M = z.shape[0]
    n = num_associations(a)

    if j >= M:  # end of recursion
        n_best = num_associations(abest)
        if n > n_best:
            return a
        if n == n_best and NIS(z, zbar, S, a) < NIS(z, zbar, S, abest):
            return a
        return abest

    # Candidates for measurement j, individually compatible, best first.
    usable = np.where(ic[j, :] < g2)[0]
    order = np.argsort(ic[j, ic[j, :] < g2])

    for i in usable[order]:
        a[j] = i
        if NIS(z, zbar, S, a) < chi2isf_cached(1 - alpha_joint, 2 * (n + 1)):
            # Take this landmark out of circulation for the sub-tree, then
            # restore it. The copy decouples the column we are blanking out.
            ici = ic[j:, i].copy()
            ic[j:, i] = np.inf

            abest = _jcbb_recursive(z, zbar, S, alpha_joint, g2, j + 1, a.copy(), ic, abest)

            ic[j:, i] = ici

    # Leaving measurement j unassociated, but only if we can still win.
    if n + (M - j - 2) >= num_associations(abest):
        a[j] = -1
        abest = _jcbb_recursive(z, zbar, S, alpha_joint, g2, j + 1, a, ic, abest)

    return abest


def individual_compatibility(z: np.ndarray, zbar: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every measurement to every prediction.

    Uses only the 2x2 diagonal blocks of ``S``: this is the *individual* test.
    """
    M = z.shape[0]
    L = zbar.shape[0]
    ic = np.zeros((M, L))

    for i in range(M):
        for j in range(L):
            dz = z[i] - zbar[j]
            dz[1] = ssa(dz[1])  # the bearing residual must be wrapped
            S_jj = S[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
            ic[i, j] = float(dz.T @ np.linalg.solve(S_jj, dz))

    return ic


def NIS(z: np.ndarray, zbar: np.ndarray, S: np.ndarray, a: np.ndarray) -> float:
    """Normalized innovation squared of a whole association hypothesis.

    This is (4.66) evaluated jointly over every associated measurement, using
    the full (non-diagonal) sub-block of ``S``.
    """
    is_associated = a >= 0
    if not np.any(is_associated):
        return np.inf

    associated_indices = a[is_associated].astype(int)

    innovation = z[is_associated] - zbar[associated_indices]
    innovation[:, 1] = ssa(innovation[:, 1])
    innovation = innovation.ravel()

    base = 2 * associated_indices
    indices = np.empty(2 * associated_indices.size, dtype=int)
    indices[0::2] = base
    indices[1::2] = base + 1

    S_associated = S[np.ix_(indices, indices)]

    factor, lower = cho_factor(S_associated, overwrite_a=False, check_finite=False)
    solved = cho_solve((factor, lower), innovation, check_finite=False)

    return float(innovation @ solved)


def num_associations(array: np.ndarray) -> int:
    return int(np.count_nonzero(array > -1))


# ---------------------------------------------------------------------------
# Ground-truth association  (simulated data only)
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthAssociator:
    """Associate using the true poses and landmarks of the simulated data set.

    This is a debugging instrument, not an algorithm. Running the same tuning
    twice, once with ``method: jcbb`` and once with ``method: gt``, splits your
    error into the part the front-end is responsible for and the part the
    back-end is responsible for (Sec. 9.1). If the two runs look the same, your
    association is fine and the problem is in the models or the noise; if the
    ``gt`` run is dramatically better, chase the associations.
    """

    landmarks_gt: np.ndarray
    poses_gt: np.ndarray
    gate: float = 2.0

    def __post_init__(self) -> None:
        self.landmarks_gt = np.asarray(self.landmarks_gt, dtype=float).reshape(-1, 2)
        self.poses_gt = np.asarray(self.poses_gt, dtype=float).reshape(-1, 3)
        # Map key -> index into landmarks_gt, or -1 for a spurious landmark.
        self._key_to_truth: dict[int, int] = {}

    def register_landmark(self, key: int, position: np.ndarray) -> None:
        """Record which true landmark a newly created map landmark corresponds to."""
        distances = np.linalg.norm(self.landmarks_gt - np.asarray(position), axis=1)
        nearest = int(np.argmin(distances))
        self._key_to_truth[key] = nearest if distances[nearest] < self.gate else -1

    def associate(
        self,
        step: int,
        measurements: np.ndarray,
        local_keys: list[int],
    ) -> np.ndarray:
        """Associate measurements at ``step`` against the local map."""
        num_measurements = measurements.shape[0]
        association = np.full(num_measurements, -1, dtype=int)

        if num_measurements == 0 or len(local_keys) == 0:
            return association

        truth_to_local = {
            self._key_to_truth.get(key, -1): index
            for index, key in enumerate(local_keys)
            if self._key_to_truth.get(key, -1) >= 0
        }

        x, y, psi = self.poses_gt[min(step, len(self.poses_gt) - 1)]

        for i, (measured_range, measured_bearing) in enumerate(measurements):
            # Back-project with the true pose.
            angle = psi + measured_bearing
            world = np.array(
                [x + measured_range * np.cos(angle), y + measured_range * np.sin(angle)]
            )

            distances = np.linalg.norm(self.landmarks_gt - world, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] < self.gate and nearest in truth_to_local:
                association[i] = truth_to_local[nearest]

        return association


def get_associator(config: SlamConfig, dataset=None):
    """Return a callable ``(measurements, local_map, S, step) -> association``."""
    if config.association.method == "jcbb":

        def associate(measurements, local_map, S, step):  # noqa: ARG001 - uniform signature
            return JCBB_association(
                measurements,
                local_map.predicted_measurements,
                S,
                config.association.alpha_individual,
                config.association.alpha_joint,
            )

        return associate

    if config.association.method == "gt":
        if dataset is None or not hasattr(dataset, "landmarks_gt"):
            raise ValueError(
                "association.method 'gt' requires a data set with ground truth "
                "(the simulated data set)."
            )

        associator = GroundTruthAssociator(
            landmarks_gt=dataset.landmarks_gt,
            poses_gt=dataset.poses_gt,
            gate=config.association.gt_gate,
        )

        def associate(measurements, local_map, S, step):  # noqa: ARG001 - uniform signature
            return associator.associate(step, measurements, local_map.keys)

        associate.associator = associator  # so slam.py can register new landmarks
        return associate

    raise ValueError(f"Unknown association method: {config.association.method}")
