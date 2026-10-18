from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2, spearmanr

from graphslam.evaluation import (
    CONFIDENCE,
    average_interval,
    chi2_bounds,
    gnss_nis,
    landmark_nis,
    pose_errors,
    pose_nees,
)
from graphslam.plotting.thesis_style import thesis_figsize
from graphslam.utils import pose2_tangent_errors, rotmat2


def _ensure_ax(
    ax: plt.Axes | np.ndarray | None,
    figsize: tuple[float, float],
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kwargs,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    if ax is not None:
        fig = ax.figure if isinstance(ax, plt.Axes) else ax.flat[0].figure
        return fig, ax
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True, **subplot_kwargs)
    return fig, ax


def confidence_ellipse_2d(center, cov, scale=1.0, confidence=0.95, **kwargs):
    """Draw confidence ellipse for a 2D covariance matrix."""
    radius = np.sqrt(chi2.ppf(confidence, df=2))

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    width = 2 * np.sqrt(eigvals[0]) * radius * scale
    height = 2 * np.sqrt(eigvals[1]) * radius * scale

    return Ellipse(xy=tuple(center), width=width, height=height, angle=np.degrees(angle), **kwargs)


def draw_pose_covariances(ax, poses: np.ndarray, poses_cov: np.ndarray, cov_stride: int = 10, **kwargs) -> None:
    R_W_B = rotmat2(poses[:, 2])
    B_xy_cov = poses_cov[:, :2, :2]
    W_xy_covs = R_W_B @ B_xy_cov @ R_W_B.transpose((0, 2, 1))  # Rotate to world frame
    
    for pose, W_xy_cov in zip(poses[::cov_stride], W_xy_covs[::cov_stride]):
        ax.add_patch(
            confidence_ellipse_2d(pose[:2], W_xy_cov, fc="steelblue", alpha=0.3, ec="steelblue", **kwargs)
        )

def draw_landmark_covariances(ax, landmarks: np.ndarray, landmarks_cov: np.ndarray, alpha: float, **kwargs) -> None:
    for lm, cov in zip(landmarks, landmarks_cov):
        ax.add_patch(
            confidence_ellipse_2d(lm, cov, fc="tomato", alpha=alpha, ec="tomato", **kwargs)
        )


def plot_estimate(
    poses: np.ndarray | None = None,
    poses_cov: np.ndarray | None = None,
    poses_cov_stride: int = 10,
    landmarks: np.ndarray | None = None,
    landmarks_cov: np.ndarray | None = None,
    gnss: np.ndarray | None = None,
    poses_gt: np.ndarray | None = None,
    landmarks_gt: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the trajectory, landmarks, and optional reference data."""
    
    fig, ax = _ensure_ax(ax, figsize=(8, 8))

    kwargs.setdefault("show_legend", True)
    kwargs.setdefault("show_grid", True)
    kwargs.setdefault("equal_aspect", True)
    kwargs.setdefault("title", "MAP Estimate")
    kwargs.setdefault("traj_label", "Trajectory")
    kwargs.setdefault("x_label", "x [m]")
    kwargs.setdefault("y_label", "y [m]")

    if poses_gt is not None and len(poses_gt) > 0:
        ax.plot(poses_gt[:, 0], poses_gt[:, 1], color="black", linestyle="--", alpha=0.8, label="Ground truth trajectory", zorder=1)

    if landmarks_gt is not None and len(landmarks_gt) > 0:
        ax.scatter(landmarks_gt[:, 0], landmarks_gt[:, 1], c="0.25", marker="x", s=18, alpha=0.7, label=f"Ground truth landmarks ({len(landmarks_gt)})",zorder=2)

    if poses is not None and len(poses) > 0:
        ax.plot(poses[:, 0], poses[:, 1], color="steelblue", label=kwargs.get("traj_label"), zorder=4)
        ax.scatter(poses[0, 0], poses[0, 1], color="green", s=50, zorder=5, label="Start", marker="o", facecolors="white", edgecolors="green")
        ax.scatter(poses[-1, 0], poses[-1, 1], color="red", s=50, zorder=5, label="End", marker="s", facecolors="white", edgecolors="red")

        if poses_cov is not None:
            draw_pose_covariances(ax, poses, poses_cov, cov_stride=poses_cov_stride, zorder=2)

    if landmarks is not None and len(landmarks) > 0:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c="tomato", marker=".", label=f"Landmarks ({len(landmarks)})", zorder=3)

        if landmarks_cov is not None:
            draw_landmark_covariances(ax, landmarks, landmarks_cov, alpha=0.3, zorder=3)

    if gnss is not None and len(gnss) > 0:
        ax.scatter(gnss[:, 1], gnss[:, 2], c="gold", marker=".", alpha=0.6, label="GPS", zorder=1)

    ax.set_title(kwargs["title"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    if kwargs["show_grid"]:   
        ax.grid(lw=0.4, alpha=0.7)
    if kwargs["equal_aspect"]:
        ax.set_aspect("equal", adjustable="box") 
    if kwargs["show_legend"]:
        ax.legend()

    return fig, ax

def plot_cumulative_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_step: np.ndarray,  
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    fig, ax = _ensure_ax(ax, figsize=(6, 3), sharex=True)

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_step  = np.nan_to_num(t_step,  nan=0.0)
    t_other = np.maximum(t_step - (t_cov + t_assoc + t_opt), 0.0)

    total_step = float(np.sum(t_step))
    total_cov = float(np.sum(t_cov))
    total_assoc = float(np.sum(t_assoc))
    total_opt = float(np.sum(t_opt))
    print(
        f"Total processing times [s] -> "
        f"step: {total_step:.3f}, cov: {total_cov:.3f}, assoc: {total_assoc:.3f}, opt: {total_opt:.3f}"
    )

    t_cov_cum = np.cumsum(t_cov)
    t_assoc_cum = np.cumsum(t_assoc)
    t_opt_cum = np.cumsum(t_opt)
    t_other_cum = np.cumsum(t_other)

    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    parts = [t_cov_cum, t_assoc_cum, t_opt_cum, t_other_cum]

    parts = [p for p in parts]
    ax.stackplot(steps, *parts, labels=labels)
    ax.set_ylabel("Time (S)")
    ax.set_title(f"Cumulative Processing Time (Total: {total_step:.2f} s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    return fig, ax
        
        
def plot_per_step_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    t_assoc: np.ndarray,
    t_opt: np.ndarray,
    t_step: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-step and cumulative processing time breakdown."""
    fig, ax = _ensure_ax(ax, figsize=(6, 2.5))

    t_cov   = np.nan_to_num(t_cov,   nan=0.0)
    t_assoc = np.nan_to_num(t_assoc, nan=0.0)
    t_opt   = np.nan_to_num(t_opt,   nan=0.0)
    t_step  = np.nan_to_num(t_step,  nan=0.0)
    
    t_other = np.maximum(t_step - (t_cov + t_assoc + t_opt), 0.0)

    labels = ["Covariance extraction", "Association", "Optimisation", "Other"]
    parts = [t_cov, t_assoc, t_opt, t_other]

    parts_ms = [1000*p for p in parts]
    ax.stackplot(steps, *parts_ms, labels=labels)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Per-step Processing Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, lw=0.4)

    # Print per-step processing time statistics
    per_step_ms = t_step * 1000
    n_above_214 = int(np.sum(per_step_ms > 214)) # 214 ms is time between lidar scans in VP
    share_above_214 = (n_above_214 / len(per_step_ms) * 100.0) if len(per_step_ms) else 0.0
    print(f"Per-step Processing Time - Max: {np.max(per_step_ms):.2f} ms, "
        f"95th Quantile: {np.quantile(per_step_ms, 0.95):.2f} ms, "
        f"Average: {np.mean(per_step_ms):.2f} ms, "
        f"Median: {np.median(per_step_ms):.2f} ms, "
        f"Steps > 214 ms: {n_above_214} ({share_above_214:.2f}%)")

    return fig, ax


def _has_clique_counts(n_support: np.ndarray | None) -> bool:
    """Whether ``num_support_cliques`` was logged. GTSAM 4.3 does not provide
    it, so a stock install logs zeros.
    """
    if n_support is None:
        return False
    n_support = np.asarray(n_support, dtype=float)
    return n_support.size > 0 and np.any(np.isfinite(n_support) & (n_support > 0))


def plot_landmarks_and_timing(
    steps: np.ndarray,
    t_cov: np.ndarray,
    n_local: np.ndarray,
    n_support: np.ndarray,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time and in-view predicted landmark count."""

    show_support = _has_clique_counts(n_support)
    nrows = 3 if show_support else 2

    fig, axes = _ensure_ax(
        axes, figsize=(6, 2 * nrows), nrows=nrows, ncols=1, sharex=True
    )

    axes[0].plot(steps, t_cov*1000, lw=0.8, color="tab:blue", label="Covariance recovery time")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", axis="both", lw=0.4)
    axes[0].legend(loc="upper left")
    axes[0].tick_params(axis="x", which="both", labelbottom=False)

    axes[1].plot(steps, n_local, lw=0.8, color="tab:orange", label="# In-view landmarks")
    axes[1].set_ylabel("# In-view landmarks")
    axes[1].grid(True, lw=0.4)
    axes[1].legend()
    axes[1].set_xlabel("Scan step")

    if show_support:
        axes[1].tick_params(axis="x", which="both", labelbottom=False)
        axes[1].set_xlabel("")

        axes[2].plot(steps, n_support, lw=0.8, color="tab:green", label="# Support cliques")
        axes[2].set_ylabel("# Support cliques")
        axes[2].set_xlabel("Scan step")
        axes[2].grid(True, lw=0.4)
        axes[2].legend()

    return fig, axes


def plot_timing_vs_landmarks(
    n_local: np.ndarray,
    n_support: np.ndarray,
    t_cov: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot covariance recovery step time against in-view landmark count."""
    show_support = _has_clique_counts(n_support)
    ncols = 2 if show_support else 1

    fig, axes = _ensure_ax(ax, figsize=(4 * ncols, 4), nrows=1, ncols=ncols)
    axes = np.atleast_1d(axes)

    t_cov_ms = 1000.0 * t_cov

    rho_local, _ = spearmanr(n_local, t_cov_ms)

    def plot_panel(axis, x, y, xlabel, rho):
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid = x[valid]
        y_valid = y[valid]
        x_values = np.unique(x_valid)

        axis.scatter(
            x_valid,
            y_valid,
            alpha=0.2,
            s=12,
            color="steelblue",
            label="Samples",
            zorder=1,
        )
        if x_values.size:
            quartiles = np.array(
                [np.quantile(y_valid[x_valid == value], [0.25, 0.5, 0.75]) for value in x_values]
            )
            axis.fill_between(
                x_values,
                quartiles[:, 0],
                quartiles[:, 2],
                color="darkorange",
                alpha=0.25,
                linewidth=0,
                label="IQR",
                zorder=2,
            )
            axis.plot(
                x_values,
                quartiles[:, 1],
                color="darkorange",
                lw=2,
                marker="o",
                markersize=3,
                label="Median",
                zorder=3,
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Cov recovery time (ms)")
        axis.set_title("Spearman rho: " + r"$\rho$" + f" = {rho:.2f}")
        axis.grid(True, lw=0.4)
        axis.legend(fontsize=8)

    plot_panel(axes[0], n_local, t_cov_ms, "# In-view landmarks", rho_local)

    if show_support:
        rho_support, _ = spearmanr(n_support, t_cov_ms)
        plot_panel(axes[1], n_support, t_cov_ms, "# Support cliques", rho_support)

    return fig, axes


def plot_landmark_growth(
    steps: np.ndarray, 
    n_landmarks: np.ndarray,
    ax: plt.Axes | None = None,
    **kwargs
) -> tuple[plt.Figure, plt.Axes]:
    """Plot confirmed landmark count over time."""
    fig, ax = _ensure_ax(ax, figsize=(8, 2.5))

    ax.plot(steps, n_landmarks, **kwargs)
    ax.set_xlabel("Scan step")
    ax.set_ylabel("# landmarks")
    ax.set_title("Landmark Count Over Time")
    ax.grid(True, lw=0.4)
    return fig, ax 



def plot_pose_diagnostics(
    poses: np.ndarray | None = None,
    poses_gt: np.ndarray | None = None,
    poses_covs: np.ndarray | None = None,
    axes: plt.Axes | None = None,
    cov_frame: Literal["body", "world"] = "body",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot x, y and heading errors over time, with 1, 2 and 3 sigma envelopes.

    Errors need ``poses`` and ``poses_gt``; envelopes need ``poses_covs``.
    ``cov_frame`` is ``"body"`` (tangent-space errors and covariance) or
    ``"world"`` (world-frame x/y, covariance rotated into the world frame).
    """

    fig, axes = _ensure_ax(axes, figsize=(8, 5.2), nrows=3, ncols=1, sharex=True)

    
    if poses is None and poses_gt is None and poses_covs is None:
        raise ValueError("At least one diagnostic source must be provided.")

    plot_error = poses is not None and poses_gt is not None
    plot_cov = poses_covs is not None

    if poses_gt is not None and poses is None:
        raise ValueError("poses_gt was provided, but poses is None.")

    if cov_frame == "world" and poses_covs is not None and poses is None:
        raise ValueError("poses must be provided when cov_frame='world'.")

    # Determine common length
    lengths = []
    if poses is not None:
        lengths.append(len(poses))
    if poses_gt is not None:
        lengths.append(len(poses_gt))
    if poses_covs is not None:
        lengths.append(len(poses_covs))


    n = min(lengths)
    steps = np.arange(n)  # Assuming poses, poses_gt, and poses_covs are all aligned in time and have the same length. Adjust if needed.

    if poses is not None:
        poses = poses[:n]
    if poses_gt is not None:
        poses_gt = poses_gt[:n]
    if poses_covs is not None:
        poses_covs = poses_covs[:n]


    labels = ["x [m]", "y [m]", r"$\theta$ [deg]"]

    for ax, ylabel in zip(axes, labels):
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.65)
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.4, alpha=0.7)

    # -------------------------------------------------------------------------
    # Covariance envelopes
    # -------------------------------------------------------------------------
    if plot_cov:
        xy_covs = poses_covs[:, :2, :2]

        if cov_frame == "world":
            R = rotmat2(poses[:, 2])
            xy_covs = R @ xy_covs @ R.transpose((0, 2, 1))

        sigmas_xy = np.sqrt(
            np.maximum(np.diagonal(xy_covs, axis1=1, axis2=2), 0.0)
        )
        sigma_x = sigmas_xy[:, 0]
        sigma_y = sigmas_xy[:, 1]
        sigma_theta = np.rad2deg(
            np.sqrt(np.maximum(poses_covs[:, 2, 2], 0.0))
        )

        sigma_series = [sigma_x, sigma_y, sigma_theta]

        colors = {1: "steelblue", 2: "seagreen", 3: "tomato"}
        alphas = {1: 0.28, 2: 0.18, 3: 0.10}
        sigma_handles = []
        sigma_labels = []

        for ax_idx, (ax, sigma) in enumerate(zip(axes, sigma_series)):
            for k in (3, 2, 1):
                bound = k * sigma

                label = rf"$\pm {k}\sigma$" if ax_idx == 0 else None
                fill = ax.fill_between(
                    steps,
                    -bound,
                    bound,
                    color=colors[k],
                    alpha=alphas[k],
                    label=label,
                    linewidth=0.0,
                )
                ax.plot(bound, color=colors[k], lw=0.7, alpha=0.5)
                ax.plot(-bound, color=colors[k], lw=0.7, alpha=0.5)

                if ax_idx == 0:
                    sigma_handles.append(fill)
                    sigma_labels.append(rf"$\pm {k}\sigma$")

    # -------------------------------------------------------------------------
    # Component errors
    # -------------------------------------------------------------------------
    if plot_error:
        if cov_frame == "body":
            error = pose2_tangent_errors(poses, poses_gt)
        else:
            error = pose_errors(poses, poses_gt)
        
        x_error = error[:, 0]
        y_error = error[:, 1]
        yaw_error = error[:, 2]

        x_rmse = float(np.sqrt(np.mean(x_error**2)))
        y_rmse = float(np.sqrt(np.mean(y_error**2)))
        yaw_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(yaw_error**2))))

        error_series = [
            (x_error, f"x error, RMSE={x_rmse:.2f} m"),
            (y_error, f"y error, RMSE={y_rmse:.2f} m"),
            (np.rad2deg(yaw_error), f"yaw error, RMSE={yaw_rmse_deg:.2f} deg"),
        ]

        for ax, (err, label) in zip(axes, error_series):
            ax.plot(
                err,
                color="tab:orange",
                lw=0.9,
                label=label,
                zorder=10,
            )

    # -------------------------------------------------------------------------
    # Titles and legends
    # -------------------------------------------------------------------------
    if plot_cov and plot_error:
        title = f"Pose errors with {cov_frame}-frame covariance envelopes"
    elif plot_cov:
        title = f"{cov_frame.capitalize()}-frame pose covariance envelopes"
    else:
        title = "Pose component errors"

    axes[0].set_title(title)
    axes[-1].set_xlabel("Scan step")

    handles, labels = axes[0].get_legend_handles_labels()
    error_handles = [h for h, l in zip(handles, labels) if "error" in l.lower()]
    error_labels = [l for l in labels if "error" in l.lower()]
    if error_handles:
        error_legend = axes[0].legend(
            error_handles,
            error_labels,
            loc="upper left",
            fontsize=9,
        )
        axes[0].add_artist(error_legend)

    if plot_cov and sigma_handles:
        axes[0].legend(
            sigma_handles,
            sigma_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9,
            frameon=True,
            ncol=3,
        )

    for ax in axes[1:]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", fontsize=9)

    return fig, axes


def _plot_consistency(
    ax: plt.Axes,
    steps: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    expected: float,
    title: str,
    ylabel: str,
    average_text: str,
    interval: tuple[float, float],
    extra_text: str | None = None,
) -> None:
    """Draw one NEES/NIS figure in the shared style.

    The per-step values as a line, the 95% bounds dashed, the expected value
    dotted; the average with its interval in a box, and the share of values
    inside the bounds in the title.
    """
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    constant_bounds = np.ndim(lower) == 0
    lower = np.broadcast_to(np.asarray(lower, dtype=float), values.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), values.shape)
    inside_pct = 100.0 * np.mean((values >= lower) & (values <= upper))

    # Break the lines across long gaps (GNSS dropouts) instead of bridging them.
    if len(steps) > 2:
        gaps = np.diff(steps)
        breaks = np.flatnonzero(gaps > 10 * np.median(gaps)) + 1
        steps, values, lower, upper = (np.insert(a, breaks, np.nan) for a in (steps, values, lower, upper))

    ax.plot(steps, values, lw=1.0, color="steelblue", label=ylabel)
    bound_style = dict(lw=1.0, ls="--")
    if constant_bounds:
        ax.axhline(lower[0], color="orange", label=f"Lower {CONFIDENCE:.0%} bound", **bound_style)
        ax.axhline(upper[0], color="tomato", label=f"Upper {CONFIDENCE:.0%} bound", **bound_style)
    else:
        ax.plot(steps, lower, color="orange", label=f"Lower {CONFIDENCE:.0%} bound", **bound_style)
        ax.plot(steps, upper, color="tomato", label=f"Upper {CONFIDENCE:.0%} bound", **bound_style)
    ax.axhline(expected, lw=1.0, ls=":", color="0.3", label=f"Expected value ({expected:g})")

    text = f"{average_text}\n{CONFIDENCE:.0%} interval: [{interval[0]:.2f}, {interval[1]:.2f}]"
    if extra_text:
        text += f"\n{extra_text}"
    ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_title(f"{title} ({inside_pct:.1f}% inside the {CONFIDENCE:.0%} bounds)")
    ax.set_xlabel("Scan step")
    ax.set_ylabel(ylabel)
    ax.grid(True, lw=0.4)
    ax.legend(loc="upper left", fontsize=8)


def plot_pose_nees(
    poses: np.ndarray,
    poses_gt: np.ndarray,
    poses_covs: np.ndarray,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot NEES over time for pose estimates."""

    fig, ax = _ensure_ax(ax, figsize=(8, 3))

    n = len(poses)
    nees = pose_nees(poses, poses_gt[:n], poses_covs)
    anees = np.sum(nees) / n

    lower, upper = chi2_bounds(3)
    _plot_consistency(
        ax, np.arange(n), nees, lower, upper, expected=3,
        title="Pose NEES", ylabel="NEES",
        average_text=f"ANEES = {anees:.2f} (expected 3)",
        interval=average_interval(3 * n, n),
    )
    return fig, ax


def plot_position_nis(
    gnss: np.ndarray,
    poses: np.ndarray,
    poses_covs: np.ndarray,
    poses_times: np.ndarray,
    sigma_gnss: float = 1.0,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot NIS between GNSS samples and nearest-in-time pose estimates.

    ``sigma_gnss`` is the assumed std. dev. of each GNSS position component, set
    by ``noise.sigma_gnss`` in the config. The NIS treats GNSS as a measurement
    of the position with covariance ``sigma_gnss**2 * I``.
    """
    fig, ax = _ensure_ax(ax, figsize=(8, 3))

    nis_xy, nearest_pose_indices, _ = gnss_nis(gnss, poses, poses_covs, poses_times, sigma_gnss)
    n = len(nis_xy)
    anis_xy = np.sum(nis_xy) / n

    lower, upper = chi2_bounds(2)
    _plot_consistency(
        ax, nearest_pose_indices, nis_xy, lower, upper, expected=2,
        title="GPS position NIS", ylabel="NIS",
        average_text=f"ANIS = {anis_xy:.2f} (expected 2)",
        interval=average_interval(2 * n, n),
        extra_text=f"Assumed GPS sigma = {sigma_gnss:g} m",
    )
    return fig, ax

def plot_association(
    diagnostics: dict[str, np.ndarray],
    show_covariances: bool = True,
    axes: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one saved association diagnostic in world space and measurement space."""
    
    fig, axes = _ensure_ax(axes, figsize=(8, 6), nrows=1, ncols=2)
    world_ax, meas_ax = axes
    
    scan_step = int(diagnostics.get("scan_step")[0])
    
    pose      = np.asarray(diagnostics.get("pose"), dtype=float)
    z         = np.asarray(diagnostics.get("measurements"), dtype=float)
    z_pred    = diagnostics.get("predicted_measurements")
    assoc     = diagnostics.get("association")
    local_lms = diagnostics.get("local_landmarks")
    S         = diagnostics.get("innovation_covariance")
    
    if pose.shape != (3,):
        raise ValueError(f"Expected pose with shape (3,), got {pose.shape}")
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"Expected measurements with shape (N, 2), got {z.shape}")

    directions = np.column_stack((np.cos(z[:, 1]), np.sin(z[:, 1])))  # (N, 2)
    local_points = z[:, 0, None] * directions  # (N, 1) * (N, 2) -> (N, 2)
    z_world = pose[None, :2] + local_points @ rotmat2(pose[2]).T
    z_deg = z.copy()
    z_pred_deg = z_pred.copy()
    
    z_deg[:, 1]  = np.rad2deg(z[:, 1])
    z_pred_deg[:, 1] = np.rad2deg(z_pred[:, 1])

    ### WORLD SPACE
    world_ax.scatter(pose[0], pose[1], c="black", marker="o", s=45, label="Robot")
    heading = rotmat2(pose[2]) @ np.array([2.0, 0.0])
    world_ax.arrow(pose[0], pose[1], heading[0], heading[1], width=0.05, head_width=0.5, length_includes_head=True, color="black", alpha=0.8)

    if len(local_lms) > 0:
        world_ax.scatter(local_lms[:, 0], local_lms[:, 1], c="tomato", marker="x", s=45, label="Predicted landmarks")

    if len(z_world) > 0:
        is_ass = assoc >= 0
        world_ax.scatter(z_world[~is_ass, 0], z_world[~is_ass, 1], c="dimgray", marker=".", s=45,label="Unassociated measurements")
        world_ax.scatter(z_world[is_ass, 0], z_world[is_ass, 1], c="steelblue", marker=".", s=55, label="Associated measurements")

        for i, ass_idx in enumerate(assoc):
            if ass_idx < 0:
                continue
            world_ax.plot([z_world[i, 0], local_lms[ass_idx, 0]], [z_world[i, 1], local_lms[ass_idx, 1]], color="seagreen", lw=0.9, alpha=0.7)

    world_ax.set_title(f"Association Result, Scan {scan_step}")
    world_ax.set_xlabel("X (m)")
    world_ax.set_ylabel("Y (m)")
    world_ax.set_aspect("equal")
    world_ax.grid(True, lw=0.4)
    world_ax.legend(fontsize=8)
   
    ### MEASUREMENT SPACE
    if len(z_pred_deg) > 0:
        meas_ax.scatter(z_pred_deg[:, 0], z_pred_deg[:, 1], c="tomato", marker="x", s=45, label="Predicted")

    if len(z_deg) > 0:
        is_ass = assoc >= 0
        meas_ax.scatter(z_deg[~is_ass, 0], z_deg[~is_ass, 1], c="dimgray", marker=".", s=45, label="Unassociated")
        meas_ax.scatter(z_deg[is_ass, 0], z_deg[is_ass, 1], c="steelblue", marker=".", s=55, label="Associated")

        for i, ass_idx in enumerate(assoc):
            if ass_idx < 0:
                continue
            meas_ax.plot([z_deg[i, 0], z_pred_deg[ass_idx, 0]], [z_deg[i, 1], z_pred_deg[ass_idx, 1]], color="seagreen", lw=0.9, alpha=0.7)

    if show_covariances and S.shape == (2 * len(z_pred), 2 * len(z_pred)):
        transform = np.array([[0.0, 180.0 / np.pi], [1.0, 0.0]])
        for j, center in enumerate(z_pred_deg):
            cov_rb = S[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
            cov_br = transform @ cov_rb @ transform.T
            meas_ax.add_patch(confidence_ellipse_2d(center, cov_br, fc="none", ec="tomato", alpha=0.35, lw=0.7))

    meas_ax.set_title("Measurement Space")
    meas_ax.set_xlabel("Bearing (deg)")
    meas_ax.set_ylabel("Range (m)")
    meas_ax.grid(True, lw=0.4)
    handles, _ = meas_ax.get_legend_handles_labels()
    if handles:
        meas_ax.legend(fontsize=8)

    return fig, axes


def plot_wrong_association(
    diagnostics: list[dict[str, np.ndarray]],
    steps: dict[str, np.ndarray],
    wrong_step: int,
    window: int = 50,
    loss_label: str | None = None,
    axes: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the landmark NIS (log scale) and the association counts around a forced wrong association.

    ``steps`` is the per-step log of the run; ``window`` is the number of scan
    steps shown on either side of ``wrong_step``; ``loss_label`` names the loss
    on the landmark factors in the title.
    """
    fig, axes = _ensure_ax(
        axes, figsize=(8, 4.5), nrows=2, ncols=1, sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    nis_ax, count_ax = axes
    first, last = wrong_step - window, wrong_step + window

    nis, dofs, nis_steps = landmark_nis(diagnostics)
    lower, upper = chi2_bounds(dofs)
    shown = (nis_steps >= first) & (nis_steps <= last)
    x, values, lower, upper = nis_steps[shown], (nis / dofs)[shown], (lower / dofs)[shown], (upper / dofs)[shown]
    # Break the lines at steps where nothing was associated, so the gap shows.
    if len(x) > 2:
        gaps = np.diff(x)
        breaks = np.flatnonzero(gaps > np.median(gaps)) + 1
        x, values, lower, upper = (np.insert(np.asarray(a, dtype=float), breaks, np.nan)
                                   for a in (x, values, lower, upper))
    nis_ax.semilogy(x, values, lw=1.0, color="steelblue", marker=".", ms=3, label="NIS per DOF")
    nis_ax.semilogy(x, lower, lw=1.0, ls="--", color="orange", label=f"Lower {CONFIDENCE:.0%} bound")
    nis_ax.semilogy(x, upper, lw=1.0, ls="--", color="tomato", label=f"Upper {CONFIDENCE:.0%} bound")
    nis_ax.axhline(1, lw=1.0, ls=":", color="0.3", label="Expected value (1)")
    nis_ax.set_ylabel("NIS per DOF")
    loss = f" ({loss_label})" if loss_label else ""
    nis_ax.set_title(f"Forced wrong association at step {wrong_step}{loss}: landmark NIS (log scale)")

    scan = np.asarray(steps["scan_step"])
    associated = np.asarray(steps["num_associated_measurement"])
    total = associated + np.asarray(steps["num_unassociated_measurement"])
    shown = (scan >= first) & (scan <= last)
    count_ax.step(scan[shown], total[shown], where="mid", lw=1.0, ls=":", color="k", label="Measurements")
    count_ax.step(scan[shown], associated[shown], where="mid", lw=1.4, color="steelblue", label="Associated")
    count_ax.set_ylabel("Count")
    count_ax.set_xlabel("Scan step")
    count_ax.set_ylim(bottom=0)

    for ax in axes:
        ax.axvline(wrong_step, lw=1.0, ls=":", color="k")
        ax.grid(True, lw=0.4)
        ax.legend(loc="upper right", fontsize=8)
    count_ax.set_xlim(first, last)
    return fig, axes


def plot_landmark_nis(
    diagnostics: list[dict[str, np.ndarray]],
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot landmark measurement NIS per degree of freedom."""

    fig, ax = _ensure_ax(ax, figsize=(8, 3))

    nis, dofs, steps = landmark_nis(diagnostics)

    # Per degree of freedom, since the number of associated landmarks varies.
    lower, upper = chi2_bounds(dofs)
    D = int(np.sum(dofs))
    anis_per_dof = np.sum(nis) / D

    _plot_consistency(
        ax, steps, nis / dofs, lower / dofs, upper / dofs, expected=1,
        title="Landmark NIS per degree of freedom", ylabel="NIS per DOF",
        average_text=f"ANIS = {anis_per_dof:.2f} per DOF (expected 1)",
        interval=average_interval(D, D),
    )
    return fig, ax
