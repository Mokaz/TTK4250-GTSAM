"""The plotting stack gets imported and exercised at least once.

``test_end_to_end`` runs with ``save_plots=False``, so without this file nothing
in the suite ever imports ``graphslam.plotter`` or the functions behind it. A
rename in ``data_association`` that was not propagated to
``plotting/plotting_funcs.py`` therefore passed every test and only showed up as
an ``ImportError`` after a 2000-step Victoria Park run had finished. These tests
close that gap: the first one catches import-level breakage in well under a
second, the rest actually draw the figures.

Everything here runs on the Agg backend, so no window is opened.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from graphslam.config import SlamConfig
from graphslam.logger import SlamLogger
from graphslam.paths import simulated_data_file
from graphslam.slam import run_slam

ROOT = Path(__file__).resolve().parents[1]
STEPS = 60

# Written by plot_all on the simulated data set. position_nis is absent because
# the simulated set has no GNSS; it is covered separately below.
EXPECTED_FIGURES = [
    "final_snapshot.pdf",
    "pose_diagnostics.pdf",
    "pose_nees.pdf",
    "landmark_nis.pdf",
    "timing_cumulative.pdf",
    "timing_per_step.pdf",
    "landmarks_and_timing.pdf",
    "timing_vs_landmarks.pdf",
    "landmark_growth.pdf",
]

try:
    SIMULATED: Path | None = simulated_data_file()
except FileNotFoundError:
    SIMULATED = None

needs_data = pytest.mark.skipif(
    SIMULATED is None or not SIMULATED.exists(),
    reason="simulated data set not found",
)


def test_plotting_modules_import() -> None:
    """Cheap guard against a rename that was not carried through.

    No data set, no run, no figure -- just the imports every plotting entry
    point performs. If this fails, nothing below is worth looking at.
    """
    import graphslam.plotter  # noqa: F401
    import graphslam.plotting.plotting_funcs  # noqa: F401
    import graphslam.plotting.thesis_style  # noqa: F401
    import graphslam.plotting.utils  # noqa: F401


@pytest.fixture(scope="module")
def plotted_run(tmp_path_factory) -> Path:
    """A short simulated run with plotting switched on, shared by the tests."""
    if SIMULATED is None or not SIMULATED.exists():
        pytest.skip("simulated data set not found")

    from graphslam.loaders.simulated import SimulatedDataLoader

    output_dir = tmp_path_factory.mktemp("plotted_run")

    # Deliberately the stock configuration: log_association_diagnostics is on
    # (plot_landmark_nis needs it) and log_snapshot is off, since snap_final.npz
    # is written regardless and that is what every plot here reads.
    config = SlamConfig.load(ROOT / "configs" / "sim_default.yaml")

    run_slam(
        config=config,
        dataset=SimulatedDataLoader(SIMULATED),
        output_dir=output_dir,
        num_steps=STEPS,
        show_plots=False,
        save_plots=True,
    )

    return output_dir


@needs_data
def test_plot_all_writes_every_figure(plotted_run: Path) -> None:
    """Each plotting function runs to completion and leaves a non-empty file."""
    figure_dir = plotted_run / "figures"
    assert figure_dir.is_dir(), "run_slam(save_plots=True) should create figures/"

    for name in EXPECTED_FIGURES:
        path = figure_dir / name
        assert path.exists(), f"{name} was not written"
        assert path.stat().st_size > 0, f"{name} is empty"


@needs_data
def test_position_nis_plot_runs(plotted_run: Path) -> None:
    """The one plot the simulated set cannot reach on its own.

    ``plot_position_nis`` is skipped above because only Victoria Park carries
    GNSS. Feeding it a synthetic track keeps it on a tested path instead of
    being discovered broken halfway through a long real run.
    """
    from graphslam.plotting.plotting_funcs import plot_position_nis

    snapshot = SlamLogger.load_snapshot(plotted_run / "snapshots" / "snap_final.npz")
    steps = SlamLogger.load_steps(plotted_run)

    poses = snapshot["poses"]
    times = steps["scan_time"][: len(poses)]

    # [time, x, y], one sample every fifth pose, offset so the NIS is non-zero.
    gnss = np.column_stack(
        [times[::5], poses[::5, 0] + 0.3, poses[::5, 1] - 0.2]
    )

    fig, ax = plot_position_nis(
        gnss=gnss,
        poses=poses,
        poses_covs=snapshot["poses_covariance"],
        poses_times=times,
    )

    assert ax.collections, "the NIS scatter should have been drawn"
    plt.close(fig)


@needs_data
def test_plotter_replots_a_finished_run(plotted_run: Path, tmp_path: Path) -> None:
    """``plot_run <dir>`` must work on a run directory after the fact.

    This is the path students use when they want a different figure format, or
    want to compare two runs without re-running either.
    """
    from graphslam.plotter import SlamRunPlotter

    plotter = SlamRunPlotter.from_run(plotted_run)
    assert plotter.snapshots, "snapshots should have been logged"

    fig, _ = plotter.plot_final_snapshot()
    out = tmp_path / "replot.png"
    fig.savefig(out)
    plt.close(fig)

    assert out.stat().st_size > 0
