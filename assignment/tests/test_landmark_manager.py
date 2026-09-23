"""Task 1 (g2) the M-of-N confirmation rule, plus the manager around it."""

from __future__ import annotations

import numpy as np
import pytest

from graphslam.landmark_manager import (
    SupportingObservation,
    TentativeLandmark,
    TentativeLandmarkManager,
)


def _landmark_seen_at(steps: list[int]) -> TentativeLandmark:
    return TentativeLandmark(
        position=np.array([1.0, 2.0]),
        supporting_observations=[
            SupportingObservation(step=step, measurement=np.array([4.0, 0.1]))
            for step in steps
        ],
    )


# ---------------------------------------------------------------------------
# (g2) is_confirmed
# ---------------------------------------------------------------------------


def test_confirm_on_first_sighting_when_m_and_n_are_one() -> None:
    """M = N = 1 is the right setting for clutter-free simulated data."""
    assert _landmark_seen_at([0]).is_confirmed(current_step=0, M=1, N=1)


def test_not_confirmed_before_enough_hits() -> None:
    assert not _landmark_seen_at([5]).is_confirmed(current_step=5, M=2, N=3)


def test_confirmed_once_enough_hits_are_inside_the_window() -> None:
    assert _landmark_seen_at([4, 5]).is_confirmed(current_step=5, M=2, N=3)


def test_hits_outside_the_window_do_not_count() -> None:
    """Seen at 0 and 5, window is [3, 5]: only one hit counts, so M=2 fails."""
    assert not _landmark_seen_at([0, 5]).is_confirmed(current_step=5, M=2, N=3)


def test_the_window_includes_both_end_points() -> None:
    """With N = 3 and current step 5 the window is [3, 5] inclusive."""
    assert _landmark_seen_at([3, 5]).is_confirmed(current_step=5, M=2, N=3)
    assert not _landmark_seen_at([2, 5]).is_confirmed(current_step=5, M=2, N=3)


def test_more_hits_than_required_still_confirms() -> None:
    assert _landmark_seen_at([3, 4, 5]).is_confirmed(current_step=5, M=2, N=3)


# ---------------------------------------------------------------------------
# The manager around it
# ---------------------------------------------------------------------------


def test_tentative_landmark_is_promoted_after_the_required_hits() -> None:
    manager = TentativeLandmarkManager(M=2, N=3, gate=0.5)

    confirmed = manager.add_tentative_landmarks(
        current_step=0,
        unassociated_measurements=np.array([[4.0, 0.1]]),
        new_tentative_landmarks=np.array([[1.0, 2.0]]),
    )
    assert confirmed == []
    assert len(manager) == 1

    confirmed = manager.add_tentative_landmarks(
        current_step=1,
        unassociated_measurements=np.array([[4.1, 0.1]]),
        new_tentative_landmarks=np.array([[1.1, 2.0]]),
    )

    assert len(confirmed) == 1
    assert len(manager) == 0
    np.testing.assert_allclose(confirmed[0].position, [1.05, 2.0])


def test_a_confirmed_landmark_carries_all_of_its_observations() -> None:
    """They become retroactive factors, so they all have to survive."""
    manager = TentativeLandmarkManager(M=3, N=5, gate=0.5)

    for step in range(3):
        confirmed = manager.add_tentative_landmarks(
            current_step=step,
            unassociated_measurements=np.array([[4.0, 0.1]]),
            new_tentative_landmarks=np.array([[1.0, 2.0]]),
        )

    assert len(confirmed) == 1
    assert [obs.step for obs in confirmed[0].supporting_observations] == [0, 1, 2]


def test_measurements_outside_the_gate_spawn_separate_landmarks() -> None:
    manager = TentativeLandmarkManager(M=5, N=5, gate=0.5)

    manager.add_tentative_landmarks(
        current_step=0,
        unassociated_measurements=np.array([[4.0, 0.1]]),
        new_tentative_landmarks=np.array([[1.0, 2.0]]),
    )
    manager.add_tentative_landmarks(
        current_step=1,
        unassociated_measurements=np.array([[9.0, 0.1]]),
        new_tentative_landmarks=np.array([[8.0, 2.0]]),
    )

    assert len(manager) == 2


def test_competing_measurements_are_matched_one_to_one() -> None:
    """Two measurements, two tentatives: each landmark gets its best match."""
    manager = TentativeLandmarkManager(M=5, N=5, gate=1.0)

    manager.add_tentative_landmarks(
        current_step=0,
        unassociated_measurements=np.array([[4.0, 0.1], [4.0, 0.5]]),
        new_tentative_landmarks=np.array([[0.0, 0.0], [3.0, 0.0]]),
    )
    assert len(manager) == 2

    # Both new measurements sit near the first tentative; only one may claim it.
    manager.add_tentative_landmarks(
        current_step=1,
        unassociated_measurements=np.array([[4.0, 0.1], [4.0, 0.5]]),
        new_tentative_landmarks=np.array([[0.1, 0.0], [3.1, 0.0]]),
    )

    assert len(manager) == 2
    assert all(lm.hit_count == 2 for lm in manager.tentative_landmarks)


def test_stale_tentative_landmarks_are_pruned() -> None:
    manager = TentativeLandmarkManager(M=3, N=3, gate=0.5)

    manager.add_tentative_landmarks(
        current_step=0,
        unassociated_measurements=np.array([[4.0, 0.1]]),
        new_tentative_landmarks=np.array([[1.0, 2.0]]),
    )
    assert len(manager) == 1

    manager.add_tentative_landmarks(
        current_step=10,
        unassociated_measurements=np.empty((0, 2)),
        new_tentative_landmarks=np.empty((0, 2)),
    )
    assert len(manager) == 0


def test_manager_rejects_an_invalid_window() -> None:
    with pytest.raises(ValueError):
        TentativeLandmarkManager(M=4, N=2, gate=0.5)
