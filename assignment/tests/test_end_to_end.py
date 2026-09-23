"""A short run of the whole pipeline on the simulated data set.

Passing every unit test and still producing a broken system is entirely
possible -- the pieces can each be right and be wired together wrongly. This
catches that. It is also the fastest way to check that a change you made while
tuning has not broken something.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphslam.config import SlamConfig
from graphslam.logger import SlamLogger
from graphslam.slam import run_slam

from graphslam.paths import simulated_data_file

ROOT = Path(__file__).resolve().parents[1]
STEPS = 80

try:
    SIMULATED: Path | None = simulated_data_file()
except FileNotFoundError:
    SIMULATED = None

pytestmark = pytest.mark.skipif(
    SIMULATED is None or not SIMULATED.exists(),
    reason="simulated data set not found",
)


def _run(tmp_path: Path, **overrides) -> tuple[dict, np.ndarray]:
    from graphslam.loaders.simulated import SimulatedDataLoader

    config = SlamConfig.load(ROOT / "configs" / "sim_default.yaml")
    config.logging.log_association_diagnostics = False
    config.logging.log_snapshot = False
    for section, values in overrides.items():
        for name, value in values.items():
            setattr(getattr(config, section), name, value)

    dataset = SimulatedDataLoader(SIMULATED)

    run_slam(
        config=config,
        dataset=dataset,
        output_dir=tmp_path,
        num_steps=STEPS,
        show_plots=False,
        save_plots=False,
    )

    snapshot = SlamLogger.load_snapshot(tmp_path / "snapshots" / "snap_final.npz")
    return snapshot, dataset.poses_gt


def _position_rmse(poses: np.ndarray, poses_gt: np.ndarray) -> float:
    n = len(poses)
    error = poses[:, :2] - poses_gt[:n, :2]
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1))))


def test_pipeline_runs_and_tracks_the_trajectory(tmp_path: Path) -> None:
    snapshot, poses_gt = _run(tmp_path)

    poses = snapshot["poses"]
    landmarks = snapshot["landmarks"]

    assert len(poses) == STEPS, "one pose per scan step"
    assert len(landmarks) > 10, "the map should have grown"
    assert np.isfinite(poses).all()
    assert np.isfinite(landmarks).all()

    assert _position_rmse(poses, poses_gt) < 2.0


def test_ground_truth_association_is_at_least_as_good(tmp_path: Path) -> None:
    """Perfect association must not make the estimate worse.

    If it does, something is wrong upstream of the front-end: the models, the
    noise, or the way the graph is being built.
    """
    jcbb_snapshot, poses_gt = _run(tmp_path / "jcbb")
    gt_snapshot, _ = _run(tmp_path / "gt", association={"method": "gt"})

    jcbb_rmse = _position_rmse(jcbb_snapshot["poses"], poses_gt)
    gt_rmse = _position_rmse(gt_snapshot["poses"], poses_gt)

    assert gt_rmse <= jcbb_rmse + 0.5


def test_pose_covariances_are_valid(tmp_path: Path) -> None:
    snapshot, _ = _run(tmp_path)

    for covariance in snapshot["poses_covariance"]:
        assert covariance.shape == (3, 3)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(covariance) > -1e-10)


def test_uncertainty_grows_away_from_the_prior(tmp_path: Path) -> None:
    """The prior anchors X(0); later poses must be less certain than the first."""
    snapshot, _ = _run(tmp_path)

    covariances = snapshot["poses_covariance"]
    assert np.trace(covariances[-1]) > np.trace(covariances[0])
