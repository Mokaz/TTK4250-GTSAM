from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# On Windows, Qt warns that it cannot set the DPI mode when the terminal has
# already set it ("SetProcessDpiAwarenessContext() failed"). It is harmless.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

import matplotlib.pyplot as plt
import numpy as np

from graphslam.loaders.victoria_park import VictoriaParkLoader
from graphslam.loaders.simulated import SimulatedDataLoader
from graphslam.config import SlamConfig
from graphslam.evaluation import (
    MapQuality,
    RunSummary,
    average_interval,
    gnss_nis,
    landmark_nis,
    map_quality,
    pose_errors,
    pose_nees,
)
from graphslam.logger import SlamLogger
from graphslam.plotting.plotting_funcs import (
    plot_estimate,
    plot_pose_diagnostics,
    plot_pose_nees,
    plot_position_nis,
    plot_landmark_growth,
    plot_landmark_nis,
    plot_landmarks_and_timing,
    plot_timing_vs_landmarks,
    plot_cumulative_timing,
    plot_per_step_timing,
)
from graphslam.plotting.thesis_style import apply_thesis_style


@dataclass
class SlamRunPlotter:
    """Utility class for plotting SLAM runs from saved logs and snapshots."""
    run_dir: Path
    steps: dict[str, np.ndarray]
    snapshots: list[dict[str, np.ndarray]]
    association: list[dict[str, np.ndarray]]
    config: SlamConfig
    gnss: np.ndarray | None = None
    gt_poses: np.ndarray | None = None
    gt_landmarks: np.ndarray | None = None
    observed_landmarks: np.ndarray | None = None

    @classmethod
    def from_run(cls, run_dir: Path | str) -> SlamRunPlotter:
        run_path = Path(run_dir)
        
        gnss = None
        gt_poses = None
        gt_landmarks = None
        observed_landmarks = None

        metadata = SlamLogger.load_metadata(run_path)
        dataset = metadata["dataset"]

        if dataset == "sim":
            loader = SimulatedDataLoader()
            gt_poses = loader.poses_gt
            gt_landmarks = loader.landmarks_gt
            observed_landmarks = loader.observed_landmarks(metadata["num_poses"])
        elif dataset == "victoria_park":
            loader = VictoriaParkLoader()
            gnss = loader.gnss_filtered
        else:
            raise ValueError(f"Unknown dataset in metadata: {dataset}")

        return cls(
            run_dir      = run_path,
            steps        = SlamLogger.load_steps(run_path),
            snapshots    = SlamLogger.load_all_snapshots(run_path),
            association  = SlamLogger.load_all_association_diagnostics(run_path),
            config       = SlamConfig.load(run_path / "config.yaml"),
            gnss         = gnss,
            gt_poses     = gt_poses,
            gt_landmarks = gt_landmarks,
            observed_landmarks = observed_landmarks,
        )
    
    @property
    def figure_dir(self) -> Path:
        path = self.run_dir / "figures"
        path.mkdir(exist_ok=True)
        return path
    
    def _finish_figure(self, result, name: str, save: bool, show: bool = True, fmt: str = "pdf") -> None:
        if result is None:
            return
        
        fig = result[0] if isinstance(result, tuple) else result
        if save:
            path = self.figure_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {path}")
        if not show:
            plt.close(fig)

    def map_quality(self) -> MapQuality | None:
        """Map landmarks against the true ones; ``None`` without ground truth."""
        if self.gt_landmarks is None:
            return None
        return map_quality(
            self.snapshots[-1].get("landmarks"), self.gt_landmarks, self.observed_landmarks
        )

    def summary(self) -> RunSummary:
        """The run's headline numbers, the same ones its figures show."""
        poses = self.snapshots[-1].get("poses")
        poses_covs = self.snapshots[-1].get("poses_covariance")
        values = {}

        if self.gt_poses is not None:
            rmse = np.sqrt(np.mean(pose_errors(poses, self.gt_poses) ** 2, axis=0))
            values.update(
                rmse_x_m=float(rmse[0]),
                rmse_y_m=float(rmse[1]),
                rmse_yaw_deg=float(np.rad2deg(rmse[2])),
            )
            nees = pose_nees(poses, self.gt_poses, poses_covs)
            values.update(
                pose_anees=float(np.mean(nees)),
                pose_anees_interval=average_interval(3 * len(nees), len(nees)),
            )

        if self.association:
            nis, dofs, _ = landmark_nis(self.association)
            total_dof = int(dofs.sum())
            if total_dof > 0:
                values.update(
                    landmark_anis_per_dof=float(nis.sum() / total_dof),
                    landmark_anis_interval=average_interval(total_dof, total_dof),
                )

        if self.gnss is not None:
            sigma_gnss = self.config.noise.sigma_gnss
            try:
                nis, _, difference = gnss_nis(
                    self.gnss, poses, poses_covs, self.steps.get("scan_time"), sigma_gnss
                )
            except ValueError:  # e.g. a run too short to overlap any GNSS sample
                pass
            else:
                values.update(
                    gnss_anis=float(np.mean(nis)),
                    gnss_anis_interval=average_interval(2 * len(nis), len(nis)),
                    sigma_gnss_m=float(sigma_gnss),
                    gnss_rms_distance_m=float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                )

        return RunSummary(map_quality=self.map_quality(), **values)

    def plot_final_snapshot(self, ax = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        quality = self.map_quality()
        if quality is not None:
            kwargs.setdefault(
                "title",
                f"MAP Estimate | landmarks: {quality.found}/{quality.observed} mapped, "
                f"{quality.duplicates} duplicates, {quality.spurious} spurious",
            )
        fig, ax = plot_estimate(
            ax            = ax,
            poses         = self.snapshots[-1].get("poses"),
            poses_cov     = self.snapshots[-1].get("poses_covariance"),
            landmarks     = self.snapshots[-1].get("landmarks"),
            landmarks_cov = self.snapshots[-1].get("landmarks_covariance"), 
            gnss          = self.gnss,
            poses_gt      = self.gt_poses,
            landmarks_gt  = self.gt_landmarks,
            poses_cov_stride = 20,
            **kwargs
        )
        return fig, ax
        
    def plot_cumulative_timing(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, axes = plot_cumulative_timing(
            ax      = ax,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_step  = self.steps.get("duration_step"),         
        )
        return fig, axes
    
    def plot_per_step_timing(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, axes = plot_per_step_timing(
            ax      = ax,
            steps   = self.steps.get("scan_step"),
            t_cov   = self.steps.get("duration_covariance_extraction"),
            t_assoc = self.steps.get("duration_association"),
            t_opt   = self.steps.get("duration_optimization"),
            t_step  = self.steps.get("duration_step"),         
        )
        return fig, axes
    
    def plot_landmarks_and_timing(self, axes = None) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        fig, axes = plot_landmarks_and_timing(
            axes      = axes,
            steps     = self.steps.get("scan_step"),
            t_cov     = self.steps.get("duration_covariance_extraction"),
            n_local   = self.steps.get("num_local_landmarks"),
            n_support = self.steps.get("num_support_cliques"),
        )
        return fig, axes

    def plot_timing_vs_landmarks(self, ax = None) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_timing_vs_landmarks(
            ax        = ax,
            n_local   = self.steps.get("num_local_landmarks"),
            n_support = self.steps.get("num_support_cliques"),
            t_cov     = self.steps.get("duration_covariance_extraction"),
        )
        return fig, ax

    def plot_landmark_growth(self, ax = None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_landmark_growth(
            ax          = ax,
            steps       = self.steps.get("scan_step"),
            n_landmarks = self.steps.get("num_landmarks"),
            **kwargs
        )
        return fig, ax
    
    def plot_pose_diagnostics(self, axes = None) -> tuple[plt.Figure, np.ndarray]:
        fig, axes = plot_pose_diagnostics(
            axes = axes,
            poses = self.snapshots[-1].get("poses"),
            poses_covs = self.snapshots[-1].get("poses_covariance"),
            poses_gt = self.gt_poses,
            cov_frame = 'world',
        )
        return fig, axes
    
    
    def plot_pose_nees(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if self.gt_poses is None:
            print("No ground truth poses found, skipping NEES plot.")
            return None
        
        fig, ax = plot_pose_nees(
            ax = ax,
            poses = self.snapshots[-1].get("poses"),
            poses_covs = self.snapshots[-1].get("poses_covariance"),
            poses_gt = self.gt_poses,
        )
        return fig, ax
    

    def plot_position_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if self.gnss is None:
            print("No GNSS reference found, skipping position NIS plot.")
            return None

        fig, ax = plot_position_nis(
            ax          = ax,
            gnss        = self.gnss,
            poses       = self.snapshots[-1].get("poses"),
            poses_covs  = self.snapshots[-1].get("poses_covariance"),
            poses_times = self.steps.get("scan_time"),
            sigma_gnss  = self.config.noise.sigma_gnss,
        )

        return fig, ax

    def plot_landmark_nis(self, ax = None) -> tuple[plt.Figure, plt.Axes] | None:
        if not self.association:
            print("No association diagnostics found, skipping NIS plot.")
            return None
        
        fig, ax = plot_landmark_nis(
            diagnostics=self.association,
        )
        return fig, ax
    
    
    # def plot_association(self, save: bool = True, fmt: str = "pdf") -> None:
    #     if not self.association:
    #         print("No association diagnostics found, skipping association plot.")
    #         return
        
    #     for diag in self.association:
    #         scan_step = int(diag["scan_step"][0])
    #         fig = plot_association(
    #             diagnostics=diag,
    #             show_covariances=True,
    #             alpha_individual=self.config.association.alpha_individual,
    #             alpha_joint=self.config.association.alpha_joint,
    #         )
    #         self._finish_figure(fig, f"association_scan_{scan_step:03d}", save, show=False)

    
    def plot_all(self, save: bool = True, show: bool = True):
        self._finish_figure( self.plot_final_snapshot(), name="final_snapshot", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_pose_diagnostics(), name="pose_diagnostics", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_pose_nees(), name="pose_nees", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmark_nis(), name="landmark_nis", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_position_nis(), name="position_nis", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_cumulative_timing(), name="timing_cumulative", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_per_step_timing(), name="timing_per_step", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmarks_and_timing(), name="landmarks_and_timing", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_timing_vs_landmarks(), name="timing_vs_landmarks", save=save, show=show, fmt="pdf" )
        self._finish_figure( self.plot_landmark_growth(), name="landmark_growth", save=save, show=show, fmt="pdf" )
        if show and plt.get_fignums():
            print(
                f"Showing {len(plt.get_fignums())} figures. Close the windows to finish, "
                "or run with --no-show-plots to skip them."
            )
            plt.show()


def main() -> None:
    apply_thesis_style()
    
    parser = argparse.ArgumentParser(description="Plot a saved SLAM run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    plotter = SlamRunPlotter.from_run(run_dir)
    print(plotter.summary())
    plotter.plot_all(save=True, show=args.show)
    

if __name__ == "__main__":
    # plotter = SlamRunPlotter.from_run("runs/real/20260527_224117")
    # plotter.plot_all(save=True, show=True)
    
    main()
    
