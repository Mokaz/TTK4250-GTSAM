"""The SLAM loop. Given -- you do not need to change anything in this file.

One time step, in the order it happens below:

1. Preintegrate the odometry since the last scan, add the pose and the odometry
   factor to the graph, and solve (Task 1 a, b, c).
2. Work out which landmarks are in view and what they should look like from
   here (Task 1 d).
3. Recover the joint marginal covariance over the pose and those landmarks, and
   turn it into an innovation covariance (Task 1 e, f).
4. Associate the measurements, add a factor for every association, and hand the
   rest to the landmark manager (Task 1 g).
5. Solve again, and log.

Compare that with the EKF-SLAM loop: steps 1 and 5 are the back-end doing what
``predict`` and ``update`` used to do, and step 3 is the work the EKF got for
free because it carried ``P`` around explicitly.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
from tqdm import tqdm

from graphslam import factor_graph as fg
from graphslam.config import SlamConfig
from graphslam.data_association import get_associator
from graphslam.landmark_manager import get_tentative_landmark_manager
from graphslam.logger import AssociationDiagnostics, SlamLogger, StepDiagnostics
from graphslam.utils import pose2_to_array


@dataclass(slots=True)
class SlamStepInput:
    """One step's input, whatever the data source."""

    relative_pose: gtsam.Pose2 | None
    relative_pose_cov: np.ndarray | None
    measurements: np.ndarray
    scan_time: float


class SlamDataset(Protocol):
    name: str

    @property
    def initial_pose(self) -> np.ndarray: ...

    @property
    def max_steps(self) -> int: ...

    def iterate_slam(
        self, config: SlamConfig, num_steps: int | None = None
    ) -> Iterator[SlamStepInput]: ...


# ---------------------------------------------------------------------------
# Back-end
# ---------------------------------------------------------------------------


class Backend:
    """Thin wrapper over iSAM2 or a batch Levenberg-Marquardt re-solve.

    The batch solver re-optimizes the entire graph from scratch at every step.
    It is the honest implementation of (9.13) and it is what iSAM2 is an
    incremental approximation of, so running both on the simulated data set and
    comparing the estimates -- and the runtimes -- is the most direct way to see
    what Sec. 9.3.3 and 9.5 are actually about. Do not try it on Victoria Park
    unless you have an afternoon to spare.
    """

    def __init__(self, config: SlamConfig) -> None:
        self.mode = config.backend.solver
        self.config = config

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()

        if self.mode == "isam2":
            params = gtsam.ISAM2Params()
            params.setRelinearizeThreshold(config.backend.relinearize_threshold)
            params.relinearizeSkip = config.backend.relinearize_skip
            self.isam2 = gtsam.ISAM2(params)
            self._estimate = gtsam.Values()
        else:
            self.isam2 = None
            self._estimate = gtsam.Values()

    def update(self, new_factors: gtsam.NonlinearFactorGraph, new_values: gtsam.Values) -> None:
        if self.mode == "isam2":
            self.isam2.update(new_factors, new_values)
            self._estimate = self.isam2.calculateEstimate()
        else:
            self.graph.push_back(new_factors)
            self._estimate.insert(new_values)
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self._estimate)
            self._estimate = optimizer.optimize()

    def pose(self, key: int) -> gtsam.Pose2:
        return self._estimate.atPose2(key)

    def point(self, key: int) -> np.ndarray:
        return np.asarray(self._estimate.atPoint2(key), dtype=float).reshape(2)

    def marginal_covariance(self, key: int) -> np.ndarray:
        if self.mode == "isam2":
            return self.isam2.marginalCovariance(key)
        return gtsam.Marginals(self.graph, self._estimate).marginalCovariance(key)

    def joint_covariance(self, pose_key: int, local_map: fg.LocalMap) -> np.ndarray:
        """Joint marginal over [pose] + local landmarks, in that order."""
        if self.mode == "isam2":
            return fg.local_joint_covariance(self.isam2, pose_key, local_map, self.config)

        keys = [pose_key] + list(local_map.keys)
        dims = [3] + [2] * len(local_map)
        marginals = gtsam.Marginals(self.graph, self._estimate)
        covariance = marginals.jointMarginalCovariance(gtsam.KeyVector(keys)).fullMatrix()
        return fg.reorder_joint_covariance(covariance, keys, dims)


@dataclass
class SlamState:
    """Keys of everything that is currently in the graph."""

    backend: Backend
    new_factors: gtsam.NonlinearFactorGraph
    new_values: gtsam.Values
    pose_keys: list[int]
    landmark_keys: list[int]

    def update_and_clear(self) -> None:
        self.backend.update(self.new_factors, self.new_values)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def get_poses(self) -> np.ndarray:
        return np.array([pose2_to_array(self.backend.pose(k)) for k in self.pose_keys])

    def get_poses_covariance(self) -> np.ndarray:
        return np.stack([self.backend.marginal_covariance(k) for k in self.pose_keys], axis=0)

    def get_landmarks(self) -> np.ndarray:
        return np.array([self.backend.point(k) for k in self.landmark_keys])

    def get_landmarks_covariance(self) -> np.ndarray:
        covariances = [self.backend.marginal_covariance(k) for k in self.landmark_keys]
        return np.stack(covariances, axis=0) if covariances else np.array([])

    def landmark_positions(self) -> dict[int, np.ndarray]:
        return {key: self.backend.point(key) for key in self.landmark_keys}

    def get_snapshot(self) -> dict:
        return {
            "poses": self.get_poses(),
            "poses_covariance": self.get_poses_covariance(),
            "landmarks": self.get_landmarks(),
            "landmarks_covariance": self.get_landmarks_covariance(),
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_slam(
    config: SlamConfig,
    dataset: SlamDataset,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> None:
    logger = SlamLogger(output_dir, config.logging)

    num_steps = dataset.max_steps if num_steps is None else min(dataset.max_steps, num_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    slam = SlamState(
        backend=Backend(config),
        new_factors=gtsam.NonlinearFactorGraph(),
        new_values=gtsam.Values(),
        pose_keys=[],
        landmark_keys=[],
    )

    manager = get_tentative_landmark_manager(config)
    associate = get_associator(config, dataset)
    bearing_range_noise = fg.bearing_range_noise_model(config)

    diagnostics_steps: list[StepDiagnostics] = []

    # ======= Prior on X(0) =======
    prior_mean = gtsam.Pose2(*dataset.initial_pose)
    slam.new_values.insert(X(0), prior_mean)
    slam.pose_keys.append(X(0))
    slam.new_factors.add(
        gtsam.PriorFactorPose2(X(0), prior_mean, fg.prior_noise_model(config))
    )
    slam.update_and_clear()

    t_run_start = time.perf_counter()

    for k, meas in tqdm(
        enumerate(dataset.iterate_slam(config, num_steps)), total=num_steps, desc="SLAM"
    ):
        diagnostics = StepDiagnostics()
        t_step = time.perf_counter()

        pose_key = X(k)

        # ======= 1. Odometry: grow the graph and solve =======
        if meas.relative_pose is not None:
            key_previous = X(k - 1)
            slam.pose_keys.append(pose_key)

            predicted_pose = fg.predict_pose(slam.backend.pose(key_previous), meas.relative_pose)
            slam.new_values.insert(pose_key, predicted_pose)

            fg.add_odometry_factor(
                slam.new_factors,
                key_previous,
                pose_key,
                meas.relative_pose,
                meas.relative_pose_cov,
            )

            t0 = time.perf_counter()
            slam.update_and_clear()
            diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        pose = slam.backend.pose(pose_key)

        # ======= 2. What should we be seeing from here? =======
        t0 = time.perf_counter()
        local_map = fg.extract_local_map(
            pose, slam.landmark_keys, slam.landmark_positions(), config
        )
        diagnostics.duration_local_landmark_extraction = time.perf_counter() - t0

        # ======= 3. Covariance recovery and innovation covariance =======
        if len(local_map) > 0:
            t0 = time.perf_counter()
            P = slam.backend.joint_covariance(pose_key, local_map)
            diagnostics.duration_covariance_extraction = time.perf_counter() - t0

            S = fg.innovation_covariance(
                local_map.jacobians_pose,
                local_map.jacobians_landmark,
                P,
                config.noise.range_bearing_cov_matrix,
            )
        else:
            P = np.zeros((3, 3))
            S = np.zeros((0, 0))

        # ======= 4. Associate =======
        measurements = meas.measurements

        t0 = time.perf_counter()
        association = associate(measurements, local_map, S, k)
        diagnostics.duration_association = time.perf_counter() - t0

        is_associated = association >= 0
        for measurement, local_index in zip(
            measurements[is_associated], association[is_associated]
        ):
            fg.add_landmark_factor(
                slam.new_factors,
                pose_key,
                local_map.keys[local_index],
                measurement,
                bearing_range_noise,
            )

        # ======= 5. Landmark birth =======
        unassociated = measurements[~is_associated]
        tentative_positions = np.asarray(
            [fg.inverse_measurement(pose, z) for z in unassociated], dtype=float
        ).reshape(-1, 2)

        confirmed = manager.add_tentative_landmarks(
            current_step=k,
            unassociated_measurements=unassociated,
            new_tentative_landmarks=tentative_positions,
        )

        for landmark in confirmed:
            landmark_key = L(len(slam.landmark_keys))
            slam.landmark_keys.append(landmark_key)
            slam.new_values.insert(landmark_key, gtsam.Point2(*landmark.position))

            if hasattr(associate, "associator"):
                associate.associator.register_landmark(landmark_key, landmark.position)

            # Every observation that supported this landmark becomes a factor,
            # retroactively connecting it to the poses it was seen from. This is
            # something a filter simply cannot do: the graph keeps the past
            # around, so evidence can be added to it after the fact.
            for observation in landmark.supporting_observations:
                fg.add_landmark_factor(
                    slam.new_factors,
                    X(observation.step),
                    landmark_key,
                    observation.measurement,
                    bearing_range_noise,
                )

        t0 = time.perf_counter()
        slam.update_and_clear()
        diagnostics.add_time("duration_optimization", time.perf_counter() - t0)

        # ======= Logging =======
        diagnostics.duration_step = time.perf_counter() - t_step
        diagnostics.scan_step = k
        diagnostics.scan_time = meas.scan_time
        diagnostics.num_landmarks = len(slam.landmark_keys)
        diagnostics.num_local_landmarks = len(local_map)
        diagnostics.num_associated_measurement = int(np.sum(is_associated))
        diagnostics.num_unassociated_measurement = int(np.sum(~is_associated))
        diagnostics_steps.append(diagnostics)

        if logger.should_save_association_diagnostics(k):
            logger.save_association_diagnostics(
                AssociationDiagnostics(
                    scan_step=k,
                    scan_time=meas.scan_time,
                    pose_index=k,
                    pose=pose2_to_array(pose),
                    measurements=measurements,
                    predicted_measurements=local_map.predicted_measurements,
                    association=association,
                    local_landmarks=local_map.positions,
                    local_landmark_keys=np.asarray(local_map.keys, dtype=np.int64),
                    prior_joint_covariance=P,
                    innovation_covariance=S,
                )
            )

        if logger.should_save_snapshot(k):
            logger.save_snapshot(k, slam.get_snapshot())

    total_time = time.perf_counter() - t_run_start

    if diagnostics_steps:
        final_diagnostics = diagnostics_steps[-1]
    else:
        final_diagnostics = StepDiagnostics(scan_step=0, num_landmarks=len(slam.landmark_keys))

    logger.save_snapshot(final_diagnostics.scan_step, slam.get_snapshot(), final=True)
    logger.save_steps_diagnostics(diagnostics_steps)
    logger.save_metadata(final_diagnostics, total_time, dataset=dataset.name)

    if save_plots or show_plots:
        from graphslam.plotter import SlamRunPlotter

        plotter = SlamRunPlotter.from_run(output_dir)
        plotter.plot_all(save=save_plots, show=show_plots)
