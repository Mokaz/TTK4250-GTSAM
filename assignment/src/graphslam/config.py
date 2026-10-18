"""Typed configuration for a SLAM run. Values marked ``# TODO tune`` in
``configs/`` are yours to tune in Tasks 2 and 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import OmegaConf


@dataclass
class NoiseConfig:
    """Noise parameters. Measurements are ordered [range, bearing].

    Attributes:
        sigma_odom_x, sigma_odom_y: odometry noise in relative x and y [m].
        sigma_odom_yaw_deg: odometry noise in yaw [deg].
        sigma_range: landmark range noise [m].
        sigma_bearing_deg: landmark bearing noise [deg].
        sigma_init_pose_x, sigma_init_pose_y: initial pose prior in x and y [m].
        sigma_init_pose_yaw_deg: initial pose prior in yaw [deg].
        sigma_gnss: assumed GPS accuracy [m]. Only used for the GPS NIS; the GPS
            never enters the graph.
    """

    sigma_odom_x: float = 0.1
    sigma_odom_y: float = 0.1
    sigma_odom_yaw_deg: float = 0.1

    sigma_range: float = 0.2
    sigma_bearing_deg: float = 1.0

    sigma_init_pose_x: float = 0.05
    sigma_init_pose_y: float = 0.05
    sigma_init_pose_yaw_deg: float = 0.5

    sigma_gnss: float = 1.0

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if value <= 0:
                raise ValueError(f"Noise parameter {field_name} must be > 0, got {value}")

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
    """Sensor limits and local map. Not tuning values.

    Attributes:
        max_range: how far trees are extracted from a Victoria Park scan [m].
        range_local, bearing_local_deg: the local map offered to data
            association: the sensor's reach plus a margin.
    """

    max_range: float = 50.0
    range_local: float = 50.0
    bearing_local_deg: float = 125.0

    def __post_init__(self) -> None:
        if self.max_range <= 0:
            raise ValueError(f"max_range must be positive, got {self.max_range}")
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
        method: ``"jcbb"`` (the only one).
        alpha_individual: confidence level of the individual compatibility test.
        alpha_joint: confidence level of the joint compatibility test.
        inject_wrong_at_step: Task 4: the scan step at which one association is
            forced wrong. ``None`` leaves association alone.
    """

    method: str = "jcbb"
    alpha_individual: float = 0.999
    alpha_joint: float = 0.9999
    inject_wrong_at_step: Optional[int] = None

    def __post_init__(self) -> None:
        method_options = ["jcbb"]
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
        if self.inject_wrong_at_step is not None and self.inject_wrong_at_step < 0:
            raise ValueError(
                f"inject_wrong_at_step must be a scan step (>= 0) or null, got {self.inject_wrong_at_step}"
            )


@dataclass
class BackendConfig:
    """Back-end parameters.

    Attributes:
        solver: ``"isam2"``, or ``"batch"`` to re-solve the whole graph every
            step (slow; simulated data only).
        relinearize_threshold: iSAM2 relinearization threshold.
        relinearize_skip: relinearize only every N-th update.
        covariance_method: ``"bayes_tree"`` (default) or ``"marginals"``; the same
            result, the latter is slower.
        robust_kernel: ``"none"`` or ``"huber"``, on the landmark factors.
        huber_k: Huber threshold, in measurement noise standard deviations.
    """

    solver: str = "isam2"
    relinearize_threshold: float = 0.1
    relinearize_skip: int = 10
    covariance_method: str = "bayes_tree"
    robust_kernel: str = "none"
    huber_k: float = 1.345

    def __post_init__(self) -> None:
        solver_options = ["isam2", "batch"]
        if self.solver not in solver_options:
            raise ValueError(f"solver must be one of {solver_options}, got {self.solver}")

        covariance_options = ["bayes_tree", "marginals"]
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

    @property
    def loss_label(self) -> str:
        """The loss on the landmark factors, for titles and the run summary."""
        if self.robust_kernel == "huber":
            return f"Huber, k = {self.huber_k:g}"
        return "least squares"


@dataclass
class VictoriaParkConfig:
    """Settings for the Victoria Park dataset only.

    Attributes:
        initial_heading_deg: heading of the first pose in the GPS frame [deg].
            SLAM cannot observe it; it only affects the comparison with the GPS.
    """

    initial_heading_deg: float = 36.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_heading_deg):
            raise ValueError(f"initial_heading_deg must be finite, got {self.initial_heading_deg}")


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
    victoria_park: VictoriaParkConfig = field(default_factory=VictoriaParkConfig)
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
