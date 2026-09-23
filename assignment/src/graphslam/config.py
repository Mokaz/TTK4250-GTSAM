"""Typed configuration for a SLAM run. Nothing here is graded.

Every value marked ``# TODO tune`` in the shipped YAML files under ``configs/``
is something you are expected to experiment with in Task 2 and Task 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


@dataclass
class NoiseConfig:
    """Uncertainty parameters.

    The relative pose (odometry) model is

        Delta_T_k = f(u_k) * exp(gamma_k),   gamma_k ~ N(0, Q dt)

    and each increment is preintegrated over a lidar interval before it becomes
    a single ``BetweenFactorPose2``.

    The landmark measurement model is

        z_kj = h(x_k, m_j) + eta_kj,         eta_kj ~ N(0, R)

    with ``h`` returning [range, bearing]. Note the ordering: GTSAM's
    ``BearingRangeFactor2D`` wants *bearing before range*, this config and every
    measurement array in the code use *range before bearing*.

    Attributes:
        sigma_velocity: std. dev. of the wheel-encoder velocity [m/s].
        sigma_steer_deg: std. dev. of the steering angle [deg].
        sigma_odom_x: std. dev. of additive odometry noise in relative x [m].
        sigma_odom_y: std. dev. of additive odometry noise in relative y [m].
        sigma_odom_yaw_deg: std. dev. of additive odometry noise in yaw [deg].
        sigma_range: std. dev. of landmark range noise [m].
        sigma_bearing_deg: std. dev. of landmark bearing noise [deg].
        sigma_init_pose_x: std. dev. of the initial pose prior in x [m].
        sigma_init_pose_y: std. dev. of the initial pose prior in y [m].
        sigma_init_pose_yaw_deg: std. dev. of the initial pose prior in yaw [deg].
    """

    sigma_velocity: float = 0.1
    sigma_steer_deg: float = 0.5

    sigma_odom_x: float = 0.1
    sigma_odom_y: float = 0.1
    sigma_odom_yaw_deg: float = 0.1

    sigma_range: float = 0.2
    sigma_bearing_deg: float = 1.0

    sigma_init_pose_x: float = 0.05
    sigma_init_pose_y: float = 0.05
    sigma_init_pose_yaw_deg: float = 0.5

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"Noise parameter {field_name} must be > 0, got {value}")

    @property
    def sigma_steer_rad(self) -> float:
        return np.deg2rad(self.sigma_steer_deg)

    @property
    def sigma_odom_yaw_rad(self) -> float:
        return np.deg2rad(self.sigma_odom_yaw_deg)

    @property
    def sigma_bearing_rad(self) -> float:
        return np.deg2rad(self.sigma_bearing_deg)

    @property
    def sigma_init_pose_yaw_rad(self) -> float:
        return np.deg2rad(self.sigma_init_pose_yaw_deg)

    @property
    def control_input_cov_matrix(self) -> np.ndarray:
        """Covariance of the raw control input [velocity, steering]."""
        return np.diag([self.sigma_velocity**2, self.sigma_steer_rad**2])

    @property
    def odom_cov_matrix(self) -> np.ndarray:
        """Covariance of the relative pose increment, ordered [x, y, yaw]."""
        return np.diag(
            [
                self.sigma_odom_x**2,
                self.sigma_odom_y**2,
                self.sigma_odom_yaw_rad**2,
            ]
        )

    @property
    def range_bearing_cov_matrix(self) -> np.ndarray:
        """Landmark measurement covariance R, ordered [range, bearing]."""
        return np.diag([self.sigma_range**2, self.sigma_bearing_rad**2])

    @property
    def init_pose_cov_matrix(self) -> np.ndarray:
        """Initial pose prior covariance, ordered [x, y, yaw]."""
        return np.diag(
            [
                self.sigma_init_pose_x**2,
                self.sigma_init_pose_y**2,
                self.sigma_init_pose_yaw_rad**2,
            ]
        )


@dataclass
class TentativeLandmarkManagerConfig:
    """Landmark birth (track initiation) parameters.

    Attributes:
        M: number of distinct time steps a tentative landmark must be seen in.
        N: length of the sliding confirmation window, in time steps.
        gate: Euclidean gate [m] for matching an unassociated measurement to an
            existing tentative landmark.
    """

    M: int = 3
    N: int = 4
    gate: float = 0.1

    def __post_init__(self) -> None:
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.M > self.N:
            raise ValueError(f"M must be <= N, got M={self.M} and N={self.N}")
        if self.gate <= 0:
            raise ValueError(f"gate must be positive, got {self.gate}")


@dataclass
class SensorConfig:
    """Sensor and local-map parameters.

    ``range_local`` and ``bearing_local_deg`` define which landmarks in the map
    are considered candidates for association at the current pose. They control
    the size of the joint covariance that has to be recovered from the graph,
    and are therefore the dominant runtime knob.
    """

    range_local: float = 50.0
    bearing_local_deg: float = 125.0

    def __post_init__(self) -> None:
        if self.range_local <= 0:
            raise ValueError(f"range_local must be positive, got {self.range_local}")
        if not (0 < self.bearing_local_deg <= 180):
            raise ValueError(
                f"bearing_local_deg must be in (0, 180], got {self.bearing_local_deg}"
            )

    @property
    def bearing_local_rad(self) -> float:
        return np.deg2rad(self.bearing_local_deg)


@dataclass
class AssociationConfig:
    """Data association parameters.

    Attributes:
        method: ``"jcbb"`` for joint compatibility branch and bound, or ``"gt"``
            for ground-truth association (simulated data only). Running with
            ``"gt"`` is the fastest way to tell a front-end problem from a
            back-end problem.
        alpha_individual: confidence level of the individual compatibility test.
        alpha_joint: confidence level of the joint compatibility test.
        gt_gate: nearest-landmark gate [m] used by the ``"gt"`` associator.
    """

    method: str = "jcbb"
    alpha_individual: float = 0.999
    alpha_joint: float = 0.9999
    gt_gate: float = 2.0

    def __post_init__(self) -> None:
        method_options = ["jcbb", "gt"]
        if self.method not in method_options:
            raise ValueError(
                f"Invalid association method {self.method}, must be one of {method_options}"
            )
        if not (0 < self.alpha_individual < 1):
            raise ValueError(
                f"alpha_individual must be in (0, 1), got {self.alpha_individual}"
            )
        if not (0 < self.alpha_joint < 1):
            raise ValueError(f"alpha_joint must be in (0, 1), got {self.alpha_joint}")
        if self.gt_gate <= 0:
            raise ValueError(f"gt_gate must be positive, got {self.gt_gate}")


@dataclass
class BackendConfig:
    """Factor graph back-end parameters.

    Attributes:
        solver: ``"isam2"`` for incremental smoothing, ``"batch"`` to re-solve
            the whole graph with Levenberg-Marquardt at every step. ``"batch"``
            is only usable on the simulated data set -- which is exactly the
            point of having it (see the book, Sec. 9.3.3 and 9.5).
        relinearize_threshold: iSAM2 relinearization threshold.
        relinearize_skip: relinearize only every N-th update.
        covariance_method: how the joint marginal covariance is recovered.
            ``"auto"`` picks the fastest method your GTSAM build supports.
        robust_kernel: ``"none"`` or ``"huber"`` on the landmark measurement
            factors.
        huber_k: Huber parameter, in units of the whitened residual.
    """

    solver: str = "isam2"
    relinearize_threshold: float = 0.1
    relinearize_skip: int = 10
    covariance_method: str = "auto"
    robust_kernel: str = "none"
    huber_k: float = 1.345

    def __post_init__(self) -> None:
        solver_options = ["isam2", "batch"]
        if self.solver not in solver_options:
            raise ValueError(f"solver must be one of {solver_options}, got {self.solver}")

        covariance_options = ["auto", "bayes_tree", "marginals", "elimination"]
        if self.covariance_method not in covariance_options:
            raise ValueError(
                f"covariance_method must be one of {covariance_options}, "
                f"got {self.covariance_method}"
            )

        kernel_options = ["none", "huber"]
        if self.robust_kernel not in kernel_options:
            raise ValueError(
                f"robust_kernel must be one of {kernel_options}, got {self.robust_kernel}"
            )

        if self.relinearize_threshold <= 0:
            raise ValueError(
                f"relinearize_threshold must be positive, got {self.relinearize_threshold}"
            )
        if self.relinearize_skip < 1:
            raise ValueError(f"relinearize_skip must be >= 1, got {self.relinearize_skip}")
        if self.huber_k <= 0:
            raise ValueError(f"huber_k must be positive, got {self.huber_k}")


@dataclass
class LoggingConfig:
    """Diagnostic logging. Larger strides mean smaller run directories."""

    log_association_diagnostics: bool = True
    association_stride: int = 200
    association_steps: list[int] = field(default_factory=list)

    log_snapshot: bool = True
    snapshot_stride: int = 200
    snapshot_steps: list[int] = field(default_factory=list)

    log_error: bool = False

    def __post_init__(self) -> None:
        if self.association_stride <= 0:
            raise ValueError(
                f"association_stride must be positive, got {self.association_stride}"
            )
        if any(step < 0 for step in self.association_steps):
            raise ValueError("association_steps must contain non-negative scan steps")
        if self.snapshot_stride <= 0:
            raise ValueError(f"snapshot_stride must be positive, got {self.snapshot_stride}")
        if any(step < 0 for step in self.snapshot_steps):
            raise ValueError("snapshot_steps must contain non-negative scan steps")


@dataclass
class SlamConfig:
    """Top-level configuration for a SLAM run."""

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    tentative: TentativeLandmarkManagerConfig = field(
        default_factory=TentativeLandmarkManagerConfig
    )
    association: AssociationConfig = field(default_factory=AssociationConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, filename: str | Path) -> SlamConfig:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        config = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.load(path))
        )
        if not isinstance(config, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(config).__name__}")

        print(f"Loaded configuration from {path}")
        return config

    def save(self, filename: str | Path) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.structured(self), path)
        print(f"Configuration saved to {path}")
