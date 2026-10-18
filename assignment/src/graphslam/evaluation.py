from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from graphslam.data_association import NIS
from graphslam.utils import pose2_tangent_errors, rotmat2, ssa

# Matching radius for map and true landmarks; generous, since the whole map drifts.
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


# ---------------------------------------------------------------------------
# Accuracy and consistency: the numbers the figures show
# ---------------------------------------------------------------------------


# Confidence level of every NEES/NIS bound and interval.
CONFIDENCE = 0.95


def chi2_bounds(dof, confidence: float = CONFIDENCE):
    """Two-sided ``confidence`` bounds of a chi-squared variable with ``dof`` degrees of freedom.

    ``dof`` may be an array, giving one pair of bounds per entry.
    """
    tail = (1.0 - confidence) / 2.0
    return chi2.ppf(tail, dof), chi2.ppf(1.0 - tail, dof)


def average_interval(total_dof: int, count: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Interval (4.67) for the average of ``count`` NEES or NIS values.

    ``total_dof`` is the sum of their degrees of freedom: N d for N values of
    dimension d. The landmark NIS is averaged per degree of freedom, so there
    ``count`` is the total number of degrees of freedom as well.
    """
    lower, upper = chi2_bounds(total_dof, confidence)
    return float(lower / count), float(upper / count)


def pose_errors(poses: np.ndarray, poses_gt: np.ndarray) -> np.ndarray:
    """World-frame [x, y, yaw] errors of aligned pose arrays, yaw wrapped."""
    n = min(len(poses), len(poses_gt))
    error = np.asarray(poses[:n], dtype=float) - np.asarray(poses_gt[:n], dtype=float)
    error[:, 2] = ssa(error[:, 2])
    return error


def pose_nees(poses: np.ndarray, poses_gt: np.ndarray, poses_covs: np.ndarray) -> np.ndarray:
    """NEES of every pose (3 degrees of freedom each).

    The error is taken in the tangent space at the estimate, the frame GTSAM's
    pose covariance lives in.
    """
    n = len(poses)
    error = pose2_tangent_errors(poses, poses_gt[:n])
    return np.einsum("ij,ijk,ik->i", error, np.linalg.inv(poses_covs), error)


def landmark_nis(diagnostics: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint landmark NIS, its degrees of freedom and the scan step, per scan.

    Scans without an associated measurement are left out.
    """
    num_scans = len(diagnostics)
    nis = np.empty(num_scans, dtype=float)
    dofs = np.empty(num_scans, dtype=int)
    steps = np.empty(num_scans, dtype=int)

    for i, step_diag in enumerate(diagnostics):
        assoc = step_diag.get("association")
        nis[i] = NIS(
            step_diag.get("measurements"),
            step_diag.get("predicted_measurements"),
            step_diag.get("innovation_covariance"),
            assoc,
        )
        dofs[i] = 2 * np.sum(assoc >= 0)
        steps[i] = step_diag.get("scan_step")[0]

    valid = dofs > 0
    return nis[valid], dofs[valid], steps[valid]


def nearest_indices(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """Return index of the nearest reference time for each query time."""
    insertion_indices = np.searchsorted(reference_times, query_times)
    right_indices = np.clip(insertion_indices, 0, len(reference_times) - 1)
    left_indices = np.clip(insertion_indices - 1, 0, len(reference_times) - 1)

    left_dt = np.abs(query_times - reference_times[left_indices])
    right_dt = np.abs(query_times - reference_times[right_indices])

    return np.where(left_dt <= right_dt, left_indices, right_indices)


def gnss_nis(
    gnss: np.ndarray,
    poses: np.ndarray,
    poses_covs: np.ndarray,
    poses_times: np.ndarray,
    sigma_gnss: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NIS of each GNSS sample against the pose nearest in time (2 degrees of freedom each).

    GNSS is treated as a measurement of the position with covariance
    ``sigma_gnss**2 * I``. Returns the NIS, the index of the pose each sample was
    compared with, and the position difference (estimate minus GNSS) [m].
    """
    pose_xy = poses[:, :2]
    covs_xy = poses_covs[:, :2, :2]
    covs_xy = rotmat2(poses[:, 2]) @ covs_xy @ rotmat2(poses[:, 2]).transpose((0, 2, 1))  # Rotate to world frame

    if len(poses_times) != len(pose_xy):
        raise ValueError("pose_times must have the same length as poses.")
    if len(poses_times) == 0:
        raise ValueError("At least one pose is required.")
    if len(gnss) == 0:
        raise ValueError("At least one GNSS measurement is required.")

    # Ignore GNSS samples outside the run, e.g. after a --steps cut-off.
    in_run = (gnss[:, 0] >= poses_times[0]) & (gnss[:, 0] <= poses_times[-1])
    gnss = gnss[in_run]
    if len(gnss) == 0:
        raise ValueError("No GNSS measurement falls inside the time span of the run.")

    nearest = nearest_indices(gnss[:, 0], poses_times)
    innovation_covs_xy = covs_xy[nearest] + np.eye(2) * sigma_gnss**2
    innovation_xy = pose_xy[nearest] - gnss[:, 1:3]
    nis = np.einsum("ij,ijk,ik->i", innovation_xy, np.linalg.inv(innovation_covs_xy), innovation_xy)
    return nis, nearest, innovation_xy


@dataclass(frozen=True)
class RunSummary:
    """The headline numbers of a run; ``None`` where the dataset has no reference.

    Expected values: pose ANEES 3, GPS ANIS 2, landmark ANIS 1 per degree of
    freedom. Each average comes with its 95% interval.
    """

    rmse_x_m: float | None = None
    rmse_y_m: float | None = None
    rmse_yaw_deg: float | None = None
    pose_anees: float | None = None
    pose_anees_interval: tuple[float, float] | None = None
    landmark_anis_per_dof: float | None = None
    landmark_anis_interval: tuple[float, float] | None = None
    gnss_anis: float | None = None
    gnss_anis_interval: tuple[float, float] | None = None
    sigma_gnss_m: float | None = None
    gnss_rms_distance_m: float | None = None
    map_quality: MapQuality | None = None

    def __str__(self) -> str:
        parts = []
        if self.rmse_x_m is not None:
            parts.append(
                f"RMSE x {self.rmse_x_m:.2f} m, y {self.rmse_y_m:.2f} m, yaw {self.rmse_yaw_deg:.2f} deg"
            )
        if self.pose_anees is not None:
            parts.append(f"pose ANEES {self.pose_anees:.2f} {_expected(3, self.pose_anees_interval)}")
        if self.landmark_anis_per_dof is not None:
            parts.append(
                f"landmark ANIS {self.landmark_anis_per_dof:.2f} per DOF "
                f"{_expected(1, self.landmark_anis_interval)}"
            )
        if self.gnss_rms_distance_m is not None:
            parts.append(f"RMS distance to GPS {self.gnss_rms_distance_m:.2f} m")
        if self.gnss_anis is not None:
            parts.append(
                f"GPS ANIS {self.gnss_anis:.2f} {_expected(2, self.gnss_anis_interval)}, "
                f"GPS sigma {self.sigma_gnss_m:g} m"
            )

        lines = []
        if self.map_quality is not None:
            lines.append(f"Map quality: {self.map_quality}")
        lines.append("Summary: " + (" | ".join(parts) if parts else "no reference to compare with"))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        """For metadata.json: plain numbers, with NaN written as null."""

        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [clean(item) for item in value]
            if isinstance(value, float) and math.isnan(value):
                return None
            return value

        return clean(asdict(self))


def _expected(value: float, interval: tuple[float, float] | None) -> str:
    """``(expected 3, 95% interval [2.85, 3.16])``."""
    if interval is None:
        return f"(expected {value:g})"
    return f"(expected {value:g}, {CONFIDENCE:.0%} interval [{interval[0]:.2f}, {interval[1]:.2f}])"
