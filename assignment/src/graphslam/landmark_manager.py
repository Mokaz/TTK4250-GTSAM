"""Landmark birth: deciding when a repeated detection becomes a map landmark.

A measurement that JCBB could not associate is either a landmark you have not
seen before, or clutter. Putting every one of them straight into the graph
fills the map with spurious landmarks that then attract wrong associations;
never putting them in means the map never grows. The classic compromise is the
same ``M`` of ``N`` rule used for track initiation in target tracking: hold the
detection as *tentative*, and promote it only once it has been seen in at least
``M`` distinct time steps inside a sliding window of ``N``.

Only :meth:`TentativeLandmark.is_confirmed` is graded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from graphslam.config import SlamConfig

# Sentinel cost for out-of-gate pairs in the assignment problem: large enough
# that the solver only picks such a pair when forced, after which it is
# rejected rather than used.
_GATED_OUT_COST = 1e9


@dataclass
class SupportingObservation:
    """One observation supporting a tentative landmark."""

    step: int
    measurement: np.ndarray  # [range, bearing]


@dataclass
class TentativeLandmark:
    """A candidate landmark that is not in the graph yet."""

    position: np.ndarray
    supporting_observations: list[SupportingObservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(2)

    @property
    def hit_count(self) -> int:
        return len(self.supporting_observations)

    @property
    def last_seen_step(self) -> int:
        return self.supporting_observations[-1].step

    def steps_since_seen(self, current_step: int) -> int:
        return current_step - self.last_seen_step

    def update(
        self,
        new_position: np.ndarray,
        step: int,
        measurement: np.ndarray,
    ) -> None:
        """Record a new supporting observation.

        The position is kept as a running average. This is deliberately crude:
        once the landmark is promoted, the graph re-estimates its position
        properly from all of the supporting observations at once, so the only
        job of this average is to be good enough to keep gating sensible.

        At most one observation is recorded per time step, because the
        association in :meth:`TentativeLandmarkManager._associate` is one-to-one.
        """
        new_position = np.asarray(new_position, dtype=float).reshape(2)
        measurement = np.asarray(measurement, dtype=float).reshape(2)

        self.supporting_observations.append(
            SupportingObservation(step=step, measurement=measurement)
        )

        alpha = 1.0 / self.hit_count
        self.position = (1.0 - alpha) * self.position + alpha * new_position

    def is_confirmed(self, current_step: int, M: int, N: int) -> bool:
        """Has this landmark been seen in at least ``M`` of the last ``N`` steps?

        The window is ``[current_step - N + 1, current_step]``, inclusive at both
        ends, so ``M = N = 1`` means "confirm on the first sighting" -- which is
        the right setting for the simulated data set, where there is no clutter.

        Parameters
        ----------
        current_step : int
            The time step that has just been processed.
        M : int
            Number of distinct time steps the landmark must have been seen in.
        N : int
            Length of the sliding window, in time steps.

        Returns
        -------
        bool
            Whether the landmark should be promoted into the graph.

        Notes
        -----
        Count *time steps*, not observations. They are the same thing here
        because association is one-to-one, but a front-end that allowed two
        measurements of one landmark in a single scan would confirm on a single
        scan's worth of evidence if you counted observations -- exactly the
        clutter burst the M-of-N rule exists to reject.
        """
        # TODO(g2): count supporting observations inside the window and compare to M.
        # BEGIN SOLUTION
        window_start = current_step - N + 1
        hits_in_window = sum(
            observation.step >= window_start
            for observation in self.supporting_observations
        )
        return hits_in_window >= M
        # END SOLUTION


def get_tentative_landmark_manager(config: SlamConfig) -> TentativeLandmarkManager:
    """Build a manager from the config."""
    return TentativeLandmarkManager(
        M=config.tentative.M,
        N=config.tentative.N,
        gate=config.tentative.gate,
    )


class TentativeLandmarkManager:
    """Holds tentative landmarks and promotes them when they are confirmed.

    One time step looks like this:

    1. Unassociated measurements are back-projected to world positions.
    2. Each is matched to an existing tentative landmark, or spawns a new one.
    3. Tentative landmarks that satisfy the M-of-N rule are handed back for
       promotion into the graph.
    4. Tentative landmarks that can no longer reach ``M`` in time are pruned.
    """

    def __init__(self, M: int, N: int, gate: float) -> None:
        if M <= 0:
            raise ValueError(f"M must be > 0, got {M}")
        if N <= 0:
            raise ValueError(f"N must be > 0, got {N}")
        if M > N:
            raise ValueError(f"M must be <= N, got M={M} and N={N}")
        if gate <= 0:
            raise ValueError(f"gate must be > 0, got {gate}")

        self.M = M
        self.N = N
        self.association_gate = gate
        self.tentative_landmarks: list[TentativeLandmark] = []

    def add_tentative_landmarks(
        self,
        current_step: int,
        unassociated_measurements: np.ndarray,
        new_tentative_landmarks: np.ndarray,
    ) -> list[TentativeLandmark]:
        """Process one time step's worth of unassociated measurements.

        Parameters
        ----------
        current_step : int
        unassociated_measurements : np.ndarray, shape=(M, 2)
            The measurements JCBB could not associate, as [range, bearing].
        new_tentative_landmarks : np.ndarray, shape=(M, 2)
            The same measurements back-projected to world-frame positions.

        Returns
        -------
        list[TentativeLandmark]
            Landmarks ready to be promoted into the graph.
        """
        num_measurements = unassociated_measurements.shape[0]
        if new_tentative_landmarks.shape[0] != num_measurements:
            raise ValueError(
                f"Expected new_tentative_landmarks to have shape ({num_measurements}, 2), "
                f"got {new_tentative_landmarks.shape}"
            )

        matches = self._associate(new_tentative_landmarks)

        for i in range(num_measurements):
            world_position = new_tentative_landmarks[i]
            measurement = unassociated_measurements[i]
            match_index = matches[i]

            if match_index < 0:
                self._spawn_tentative(
                    step=current_step,
                    position=world_position,
                    measurement=measurement,
                )
            else:
                self.tentative_landmarks[match_index].update(
                    step=current_step,
                    new_position=world_position,
                    measurement=measurement,
                )

        confirmed = self._extract_confirmed(current_step)
        self.prune_unconfirmable(current_step)

        return confirmed

    def _associate(self, measurement_positions: np.ndarray) -> np.ndarray:
        """Match this step's measurements to existing tentative landmarks.

        Solved as a one-to-one assignment minimising total Euclidean distance
        (Hungarian algorithm), subject to the gate. Greedy per-measurement
        matching would be order dependent: the first measurement to claim a
        landmark wins even when a later one fits it better.
        """
        num_measurements = measurement_positions.shape[0]
        matches = np.full(num_measurements, -1, dtype=int)

        landmarks = self.tentative_landmarks
        if num_measurements == 0 or len(landmarks) == 0:
            return matches

        landmark_positions = np.array([lm.position for lm in landmarks])
        distances = np.linalg.norm(
            measurement_positions[:, None, :] - landmark_positions[None, :, :], axis=2
        )

        cost = np.where(distances < self.association_gate, distances, _GATED_OUT_COST)

        row_indices, column_indices = linear_sum_assignment(cost)
        for row, column in zip(row_indices, column_indices):
            if distances[row, column] < self.association_gate:
                matches[row] = column

        return matches

    def _spawn_tentative(
        self,
        step: int,
        position: np.ndarray,
        measurement: np.ndarray,
    ) -> None:
        self.tentative_landmarks.append(
            TentativeLandmark(
                position=position,
                supporting_observations=[SupportingObservation(step, measurement)],
            )
        )

    def _extract_confirmed(self, current_step: int) -> list[TentativeLandmark]:
        """Remove and return the tentative landmarks that are now confirmed."""
        confirmed: list[TentativeLandmark] = []
        remaining: list[TentativeLandmark] = []

        for landmark in self.tentative_landmarks:
            if landmark.is_confirmed(current_step, self.M, self.N):
                confirmed.append(landmark)
            else:
                remaining.append(landmark)

        self.tentative_landmarks = remaining
        return confirmed

    def prune_unconfirmable(self, current_step: int) -> None:
        """Drop tentative landmarks that can no longer reach ``M`` within ``N``."""
        self.tentative_landmarks = [
            lm for lm in self.tentative_landmarks
            if lm.steps_since_seen(current_step) <= self.N
        ]

    def reset(self) -> None:
        self.tentative_landmarks.clear()

    def __len__(self) -> int:
        return len(self.tentative_landmarks)
