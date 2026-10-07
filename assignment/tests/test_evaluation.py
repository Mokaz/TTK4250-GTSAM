"""The map-quality diagnostic. Given code -- these document what its numbers mean."""

from __future__ import annotations

import numpy as np
import pytest

from graphslam.evaluation import map_quality

TRUTH = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])


def test_a_perfect_map_has_everything_found_and_nothing_extra() -> None:
    quality = map_quality(TRUTH, TRUTH)

    assert (quality.found, quality.missed, quality.duplicates, quality.spurious) == (4, 0, 0, 0)
    assert quality.rms_error_m == pytest.approx(0.0)


def test_a_second_landmark_near_a_true_one_is_a_duplicate() -> None:
    landmarks = np.vstack([TRUTH, [[10.4, 0.3]]])

    quality = map_quality(landmarks, TRUTH)

    assert (quality.found, quality.duplicates, quality.spurious) == (4, 1, 0)


def test_a_landmark_far_from_every_true_one_is_spurious() -> None:
    landmarks = np.vstack([TRUTH, [[50.0, 50.0]]])

    quality = map_quality(landmarks, TRUTH)

    assert (quality.found, quality.duplicates, quality.spurious) == (4, 0, 1)


def test_a_drifted_map_still_matches_one_to_one() -> None:
    """The whole map shifted by 1.5 m: every landmark is still the right one."""
    quality = map_quality(TRUTH + [1.2, -0.9], TRUTH)

    assert (quality.found, quality.duplicates, quality.spurious) == (4, 0, 0)
    assert quality.rms_error_m == pytest.approx(1.5)


def test_only_observed_landmarks_can_be_missed() -> None:
    quality = map_quality(TRUTH[:2], TRUTH, observed=np.array([0, 1, 2]))

    assert quality.observed == 3
    assert quality.missed == 1


def test_an_empty_map_misses_everything() -> None:
    quality = map_quality(np.zeros((0, 2)), TRUTH)

    assert (quality.found, quality.missed) == (0, 4)
