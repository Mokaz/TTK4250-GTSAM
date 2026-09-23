# TTK4250 Group Assignment 2 — Factor graph SLAM

Landmark SLAM on a factor graph, solved with GTSAM's iSAM2 back-end and a JCBB
front-end, on the same two data sets as before: a simulated one with ground
truth, and the Victoria Park lidar data set.

> **This directory is the reference solution**, not the handout. Generate the
> student version with `python tools/make_handout.py --out ../handout`.

## Setup

Python 3.11 or newer.

```sh
uv sync            # or: pip install -e ".[dev]"
```

GTSAM ships as a binary wheel for most platforms. If none is available for
yours, see `docs/INSTALL.md`.

## Running

From the root of this directory:

```sh
uv run run_sim                      # simulated data
uv run run_real                     # Victoria Park

uv run run_sim --config configs/sim_default.yaml --steps 1000 --output-dir runs/sim/example
uv run plot_run runs/sim/example    # re-plot a finished run
```

Every run writes its resolved configuration, per-step diagnostics and state
snapshots into its output directory, so a run can be re-plotted and compared
later without re-running it.

## What you implement

Ten functions, all marked `TODO` in the source and graded by `pytest`:

| | Function | File |
|---|---|---|
| a | `relative_pose` | `src/graphslam/preprocessing.py` |
| b | `preintegrate` | `src/graphslam/preprocessing.py` |
| c1 | `predict_pose` | `src/graphslam/factor_graph.py` |
| c2 | `add_odometry_factor` | `src/graphslam/factor_graph.py` |
| c3 | `add_landmark_factor` | `src/graphslam/factor_graph.py` |
| d | `predict_measurement` | `src/graphslam/factor_graph.py` |
| e | `reorder_joint_covariance` | `src/graphslam/factor_graph.py` |
| f | `innovation_covariance` | `src/graphslam/factor_graph.py` |
| g1 | `inverse_measurement` | `src/graphslam/factor_graph.py` |
| g2 | `TentativeLandmark.is_confirmed` | `src/graphslam/landmark_manager.py` |

```sh
uv run pytest            # everything
uv run pytest -k jacobian -x
```

Nothing else needs changing. `slam.py`, `data_association.py`, the loaders, the
logger and the plotting are given.

## Layout

```text
configs/          run configurations; everything marked "TODO tune" is yours
data/             simulated and Victoria Park data sets
src/graphslam/
  preprocessing.py     front-end: odometry and lidar          (a, b)
  factor_graph.py      factors, measurement model, covariance (c-f, g1)
  landmark_manager.py  landmark birth, M-of-N                 (g2)
  data_association.py  JCBB and a ground-truth associator     given
  slam.py              the main loop                          given
  loaders/             dataset adapters                       given
  plotting/, plotter.py, logger.py                            given
tests/            the graded test suite
tools/            handout generation and create_handin.py
```

## Two things worth knowing before you start

**Ordering.** Measurements are `[range, bearing]` everywhere in this code base.
GTSAM's `BearingRangeFactor2D` wants bearing first. Getting this backwards
produces a system that runs and looks plausible.

**Frames.** The covariance GTSAM reports for a `Pose2` lives in the tangent
space at the current estimate, and so do the Jacobians from `Pose2.range` and
`Pose2.bearing`. Use them together and you are consistent; mix either with a
hand-derived `d/d[x, y, theta]` and you are not. This also applies when you
compute NEES — see `graphslam.utils.pose2_tangent_error`.

## Debugging

Set `association: method: gt` in `configs/sim_default.yaml` to run with perfect
data association on the simulated set. Comparing that against a `jcbb` run with
the same tuning splits your error into a front-end part and a back-end part,
which is usually the fastest way to work out what is actually wrong.
