"""Data association: JCBB.

Nothing in this file is graded. It is worth reading anyway, because
the quantity it consumes is the one you build in Task 1 (e) and (f): the
innovation covariance ``S`` of the *whole local map jointly*, cross-covariances
included. Individual compatibility only looks at the 2x2 diagonal blocks of
``S``; joint compatibility is what the off-diagonal blocks buy you.

The branch and bound search is described in Sec. 7.3.1 and 7.3.2 of the book.
"""

from __future__ import annotations

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

    # Associate the most confident measurements first (smallest best-match
    # distance): good hypotheses are found early, which prunes harder.
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
    # Skipping j leaves at most M - j - 1 further pairings, so this explores the
    # branch only if it could get strictly MORE pairings than the best so far:
    # the bound from the original JCBB paper (Neira & Tardos, 2001). Sec. 7.3.2
    # of the book says "at least as many", i.e. ``n + (M - j - 1) >= ...``,
    # which also explores branches that could only tie and lets the NIS
    # tie-break above choose between them. On Victoria Park both give the same
    # associations, and the book's version costs about 50% more search time.
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


def get_associator(config: SlamConfig):
    """Return a callable ``(measurements, local_map, S) -> association``."""
    if config.association.method != "jcbb":
        raise ValueError(f"Unknown association method: {config.association.method}")

    def associate(measurements, local_map, S):
        return JCBB_association(
            measurements,
            local_map.predicted_measurements,
            S,
            config.association.alpha_individual,
            config.association.alpha_joint,
        )

    return associate
