"""How good is the map? Ground-truth checks for the simulated data set.

Nothing here is graded, and nothing here feeds back into SLAM: it only looks at
a finished map. It answers the question the ground truth makes possible --
how many of the true landmarks the run saw ended up in the map exactly once,
and how many map landmarks are duplicates or have no true counterpart at all.

Duplicates are the signature of failed data association: a measurement of a
landmark that is already in the map was not matched to it, and became a new
landmark instead. Comparing these numbers between tuning sets tells you how much
of the error is the front-end's, and how much is left for the models and the
noise (Sec. 9.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Map landmarks are matched to true landmarks one-to-one, and only within this
# distance. It is deliberately generous: the whole map drifts with the
# trajectory, so even a good landmark can sit a metre or more from the truth.
MATCH_DISTANCE_M = 3.0


@dataclass(frozen=True)
class MapQuality:
    """Map landmarks compared with the true landmarks.

    Attributes:
        observed: true landmarks the run measured at least once (or, if that is
            not known, all true landmarks).
        found: true landmarks with a map landmark matched to them.
        duplicates: extra map landmarks near a true landmark that already has one.
        spurious: map landmarks with no true landmark within ``MATCH_DISTANCE_M``.
        rms_error_m: RMS distance between matched map and true landmarks [m].
    """

    observed: int
    found: int
    duplicates: int
    spurious: int
    rms_error_m: float

    @property
    def missed(self) -> int:
        return max(self.observed - self.found, 0)

    def __str__(self) -> str:
        return (
            f"{self.found} of {self.observed} observed true landmarks mapped, "
            f"{self.missed} missed, {self.duplicates} duplicates, {self.spurious} spurious, "
            f"RMS position error {self.rms_error_m:.2f} m"
        )


def map_quality(
    landmarks: np.ndarray,
    landmarks_gt: np.ndarray,
    observed: np.ndarray | None = None,
    match_distance: float = MATCH_DISTANCE_M,
) -> MapQuality:
    """Compare a map with the true landmarks.

    Parameters
    ----------
    landmarks : np.ndarray, shape=(n, 2)
        Estimated landmark positions, e.g. ``snapshot["landmarks"]``.
    landmarks_gt : np.ndarray, shape=(m, 2)
        True landmark positions.
    observed : np.ndarray of int, optional
        Indices into ``landmarks_gt`` of the landmarks the run actually measured.
        Only these can be missed. Defaults to all of them.
    match_distance : float
        Largest distance [m] at which a map landmark counts as a true one.
    """
    landmarks = np.asarray(landmarks, dtype=float).reshape(-1, 2)
    landmarks_gt = np.asarray(landmarks_gt, dtype=float).reshape(-1, 2)
    num_observed = len(landmarks_gt) if observed is None else len(observed)

    if len(landmarks) == 0:
        return MapQuality(num_observed, 0, 0, 0, float("nan"))

    distance = np.linalg.norm(landmarks[:, None, :] - landmarks_gt[None, :, :], axis=2)
    too_far = distance >= match_distance

    # One map landmark per true landmark, closest pairs first.
    rows, cols = linear_sum_assignment(np.where(too_far, 1e9, distance))
    matched = ~too_far[rows, cols]
    rows, cols = rows[matched], cols[matched]

    unmatched = np.setdiff1d(np.arange(len(landmarks)), rows)
    duplicates = int(np.sum(~too_far[unmatched].all(axis=1)))
    spurious = len(unmatched) - duplicates
    rms = float(np.sqrt(np.mean(distance[rows, cols] ** 2))) if len(rows) else float("nan")

    return MapQuality(num_observed, len(rows), duplicates, spurious, rms)
