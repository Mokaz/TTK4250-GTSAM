"""JCBB and the ground-truth associator. Given code -- these are regression
tests, not graded work, but they document what the front-end promises."""

from __future__ import annotations

import numpy as np
import pytest

from graphslam.data_association import (
    JCBB_association,
    NIS,
    GroundTruthAssociator,
    individual_compatibility,
    num_associations,
)


def test_jcbb_matches_a_compatible_measurement() -> None:
    associations = JCBB_association(
        np.array([[5.0, 0.2]]),
        np.array([[5.0, 0.2]]),
        np.diag([0.1, 0.01]),
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    np.testing.assert_array_equal(associations, [0])


def test_jcbb_leaves_measurements_unassociated_without_landmarks() -> None:
    associations = JCBB_association(
        np.array([[2.0, 0.0], [3.0, 0.1]]),
        np.empty((0, 2)),
        np.empty((0, 0)),
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    np.testing.assert_array_equal(associations, [-1, -1])


def test_jcbb_rejects_a_measurement_far_outside_the_gate() -> None:
    associations = JCBB_association(
        np.array([[50.0, 2.0]]),
        np.array([[5.0, 0.2]]),
        np.diag([0.1, 0.01]),
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    np.testing.assert_array_equal(associations, [-1])


def test_jcbb_is_one_to_one() -> None:
    """Two measurements may not claim the same landmark."""
    measurements = np.array([[5.0, 0.20], [5.0, 0.21]])
    predictions = np.array([[5.0, 0.20]])
    S = np.diag([1.0, 1.0])

    associations = JCBB_association(
        measurements, predictions, S, alpha_individual=0.99, alpha_joint=0.99
    )

    assert num_associations(associations) <= 1


def test_jcbb_handles_no_measurements() -> None:
    associations = JCBB_association(
        np.empty((0, 2)),
        np.array([[5.0, 0.2]]),
        np.diag([0.1, 0.01]),
        alpha_individual=0.99,
        alpha_joint=0.99,
    )

    assert associations.shape == (0,)


def test_individual_compatibility_wraps_the_bearing_residual() -> None:
    """A bearing just below +pi and one just above -pi are neighbours."""
    measurements = np.array([[5.0, np.pi - 0.01]])
    predictions = np.array([[5.0, -np.pi + 0.01]])
    S = np.diag([0.1, 0.01])

    ic = individual_compatibility(measurements, predictions, S)

    assert ic[0, 0] < 1.0, "the residual should be 0.02 rad, not ~2*pi"


def test_nis_of_an_empty_hypothesis_is_infinite() -> None:
    assert np.isinf(
        NIS(
            np.array([[5.0, 0.2]]),
            np.array([[5.0, 0.2]]),
            np.diag([0.1, 0.01]),
            np.array([-1]),
        )
    )


def test_nis_is_zero_for_a_perfect_association() -> None:
    value = NIS(
        np.array([[5.0, 0.2]]),
        np.array([[5.0, 0.2]]),
        np.diag([0.1, 0.01]),
        np.array([0]),
    )

    assert value == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Ground-truth associator
# ---------------------------------------------------------------------------


def test_ground_truth_associator_matches_through_the_true_pose() -> None:
    landmarks_gt = np.array([[10.0, 0.0], [0.0, 10.0]])
    poses_gt = np.array([[0.0, 0.0, 0.0]])

    associator = GroundTruthAssociator(landmarks_gt, poses_gt, gate=1.0)
    associator.register_landmark(key=101, position=np.array([10.05, 0.0]))
    associator.register_landmark(key=102, position=np.array([0.0, 9.95]))

    # Measurement of the first landmark: range 10, bearing 0.
    association = associator.associate(
        step=0, measurements=np.array([[10.0, 0.0]]), local_keys=[102, 101]
    )

    np.testing.assert_array_equal(association, [1])


def test_ground_truth_associator_ignores_landmarks_not_in_the_local_map() -> None:
    associator = GroundTruthAssociator(
        np.array([[10.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), gate=1.0
    )
    associator.register_landmark(key=101, position=np.array([10.0, 0.0]))

    association = associator.associate(
        step=0, measurements=np.array([[10.0, 0.0]]), local_keys=[]
    )

    np.testing.assert_array_equal(association, [-1])


def test_ground_truth_associator_rejects_a_spurious_landmark() -> None:
    """A map landmark with no true counterpart must never be associated to."""
    associator = GroundTruthAssociator(
        np.array([[10.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), gate=1.0
    )
    associator.register_landmark(key=101, position=np.array([500.0, 500.0]))

    association = associator.associate(
        step=0, measurements=np.array([[10.0, 0.0]]), local_keys=[101]
    )

    np.testing.assert_array_equal(association, [-1])
