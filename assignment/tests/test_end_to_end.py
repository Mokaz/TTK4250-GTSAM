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


def test_the_baseline_map_has_no_duplicate_or_spurious_landmarks(tmp_path: Path) -> None:
    """Every true landmark seen should end up in the map exactly once.

    With the shipped tuning, association does not fail on this data set. A
    duplicate or a spurious landmark here means a measurement was not matched to
    the landmark it came from -- usually because the innovation covariance from
    Task 1 (e) or (f), or the measurement model from (d), is wrong.
    """
    from graphslam.evaluation import map_quality
    from graphslam.loaders.simulated import SimulatedDataLoader

    snapshot, _ = _run(tmp_path)
    dataset = SimulatedDataLoader(SIMULATED)
    quality = map_quality(
        snapshot["landmarks"],
        dataset.landmarks_gt,
        observed=dataset.observed_landmarks(len(snapshot["poses"])),
    )

    assert quality.duplicates == 0, f"duplicate landmarks in the map: {quality}"
    assert quality.spurious == 0, f"spurious landmarks in the map: {quality}"
    assert quality.missed <= 2, f"true landmarks seen but never mapped: {quality}"


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


def _interrupted_dataset(after_steps: int):
    """The simulated set, with Ctrl+C pressed while step ``after_steps`` runs."""
    from graphslam.loaders.simulated import SimulatedDataLoader

    class Interrupted(SimulatedDataLoader):
        def iterate_slam(self, config, max_steps=None):
            for k, step in enumerate(super().iterate_slam(config, max_steps)):
                if k == after_steps:
                    raise KeyboardInterrupt
                yield step

    return Interrupted(SIMULATED)


def test_an_interrupted_run_is_saved_and_can_be_plotted(tmp_path: Path) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    from graphslam.plotter import SlamRunPlotter

    config = SlamConfig.load(ROOT / "configs" / "sim_default.yaml")
    run_slam(config, _interrupted_dataset(30), tmp_path, num_steps=STEPS, save_plots=False)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    steps = SlamLogger.load_steps(tmp_path)
    snapshot = SlamLogger.load_snapshot(tmp_path / "snapshots" / "snap_final.npz")

    assert metadata.get("aborted") is True, "an interrupted run should be marked as aborted"
    assert len(steps["scan_step"]) == 30
    assert len(snapshot["poses"]) == 30, "one pose per completed step"

    plotter = SlamRunPlotter.from_run(tmp_path)
    plotter.plot_all(save=True, show=False)
    assert (tmp_path / "figures" / "final_snapshot.pdf").exists()


def test_an_interrupted_run_is_discarded_without_save_on_abort(tmp_path: Path) -> None:
    config = SlamConfig.load(ROOT / "configs" / "sim_default.yaml")

    with pytest.raises(KeyboardInterrupt):
        run_slam(
            config, _interrupted_dataset(30), tmp_path, num_steps=STEPS,
            save_plots=False, save_on_abort=False,
        )

    assert not (tmp_path / "metadata.json").exists()
