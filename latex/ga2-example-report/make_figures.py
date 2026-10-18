"""Composite figures for the example report, drawn with the handout's own plotting functions.

Run from the folder holding runs/; writes into the folder given as the first argument.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend

from graphslam.evaluation import landmark_nis, pose_errors
from graphslam.logger import SlamLogger
from graphslam.plotter import SlamRunPlotter
from graphslam.plotting.thesis_style import apply_thesis_style

apply_thesis_style()
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
which = set(sys.argv[2:]) or {"sim", "real", "task4"}
_cache: dict[str, SlamRunPlotter] = {}


def run(path: str) -> SlamRunPlotter:
    if path not in _cache:
        _cache[path] = SlamRunPlotter.from_run(path)
    return _cache[path]


def prefix(ax, label: str) -> None:
    ax.set_title(f"{label}: {ax.get_title()}", fontsize=8)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def snapshot(ax, path: str, label: str, legend: bool) -> None:
    p = run(path)
    quality = p.map_quality()
    title = f"{label}: {p.snapshots[-1]['landmarks'].shape[0]} landmarks"
    if quality is not None:
        title = (f"{label}: {quality.found}/{quality.observed} mapped, "
                 f"{quality.duplicates} dupl., {quality.spurious} spurious")
    p.plot_final_snapshot(ax=ax, title=title, show_legend=legend)
    ax.set_title(title, fontsize=8)
    if ax.get_legend() is not None:
        for t in ax.get_legend().get_texts():
            t.set_fontsize(6)


def consistency_grid(top: str, bottom: str, runs: list[tuple[str, str]], name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.6))
    for col, (path, label) in enumerate(runs):
        getattr(run(path), top)(ax=axes[0, col])
        getattr(run(path), bottom)(ax=axes[1, col])
        prefix(axes[0, col], label)
        prefix(axes[1, col], label)
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.tight_layout()
    save(fig, name)


if "sim" in which:
    consistency_grid(
        "plot_pose_nees", "plot_landmark_nis",
        [("runs/sim/baseline", "S0"), ("runs/sim/sim_true", "S1")], "sim_consistency",
    )

    # Errors of S1 and S3 side by side.
    fig, grid = plt.subplots(3, 2, figsize=(9, 4.8), sharex=True)
    for col, (path, label) in enumerate([("runs/sim/sim_true", "S1"), ("runs/sim/sim_strict_birth", "S3")]):
        axes = grid[:, col]
        run(path).plot_pose_diagnostics(axes=axes)
        axes[0].set_title(f"{label}: pose errors with covariance envelopes", fontsize=8)
        for ax in axes:
            for leg in [c for c in ax.get_children() if isinstance(c, Legend)]:
                if any("sigma" in t.get_text() for t in leg.get_texts()):
                    leg.remove()  # the shaded 1/2/3 sigma bands are explained in the caption
                    continue
                for t in leg.get_texts():
                    t.set_fontsize(6)
        for ax in axes[:2]:
            ax.set_xlabel("")
    fig.tight_layout()
    save(fig, "sim_strict_birth")

if "real" in which:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (path, label) in zip(axes, [
        ("runs/real/baseline", "V0"),
        ("runs/real/real_tight_R_loose_gates", "V1"),
        ("runs/real/real_low_odom", "V3"),
    ]):
        snapshot(ax, path, label, legend=False)
    fig.tight_layout()
    save(fig, "real_maps")

    consistency_grid(
        "plot_landmark_nis", "plot_position_nis",
        [("runs/real/baseline", "V0"), ("runs/real/real_tight_R_loose_gates", "V1")], "real_consistency",
    )

    fig, ax = plt.subplots(figsize=(9, 2.4))
    for label, path, color, ls in [
        ("V0 baseline", "runs/real/baseline", "0.3", "-"),
        ("V1 tight R, loose gates", "runs/real/real_tight_R_loose_gates", "C0", "-"),
        ("V2 tight R", "runs/real/real_tight_R", "C1", "--"),
        ("V3 low odometry noise", "runs/real/real_low_odom", "C3", "-."),
    ]:
        steps = SlamLogger.load_steps(path)
        ax.plot(steps["scan_step"], steps["num_landmarks"], color=color, ls=ls, lw=1.2, label=label)
    ax.set_xlabel("Scan step")
    ax.set_ylabel("Landmarks")
    ax.set_title("Map size over the full run", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "real_growth")

if "task4" in which:
    # Stock figures only: the map, and wrong_association for both runs.
    fig = plt.figure(figsize=(9, 3.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1.6, 1])
    snapshot(fig.add_subplot(gs[:, 0]), "runs/sim/task4", "Least squares", legend=False)
    for col, (path, label) in enumerate([("runs/sim/task4", "Least squares"), ("runs/sim/task4_huber", "Huber")], start=1):
        axes = np.array([fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])])
        run(path).plot_wrong_association(axes=axes)
        axes[0].set_title(f"{label}: landmark NIS (log scale)", fontsize=8)
        for ax in axes:
            if ax.get_legend() is not None:
                ax.get_legend().remove()  # line styles are explained in the caption
        axes[0].set_xlabel("")
    fig.tight_layout()
    save(fig, "task4")
