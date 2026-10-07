from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from graphslam.config import SlamConfig
from graphslam.evaluation import map_quality
from graphslam.loaders.simulated import SimulatedDataLoader
from graphslam.logger import SlamLogger
from graphslam.slam import run_slam


def run_sim(
    config: SlamConfig,
    output_dir: Path,
    num_steps: int | None,
    show_plots: bool = False,
    save_plots: bool = True,
    save_on_abort: bool = True,
) -> None:
    dataset = SimulatedDataLoader()
    run_slam(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        num_steps=num_steps,
        show_plots=show_plots,
        save_plots=save_plots,
        save_on_abort=save_on_abort,
    )
    report_map_quality(dataset, output_dir)


def report_map_quality(dataset: SimulatedDataLoader, output_dir: Path) -> None:
    """Print how the finished map compares with the true landmarks."""
    snapshot = SlamLogger.load_snapshot(Path(output_dir) / "snapshots" / "snap_final.npz")
    quality = map_quality(
        snapshot["landmarks"],
        dataset.landmarks_gt,
        observed=dataset.observed_landmarks(len(snapshot["poses"])),
    )
    print(f"Map quality: {quality}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor graph SLAM on the simulated data set.")
    parser.add_argument("--config", type=Path, default=Path("configs/sim_default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-show-plots", action="store_true", help="Do not display plots after the run.")
    parser.add_argument("--no-save-plots", action="store_true", help="Do not save plots to the run directory.")
    parser.add_argument(
        "--no-save-on-abort",
        action="store_true",
        help="On Ctrl+C, exit without saving. By default an interrupted run is saved up to the last completed step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("runs/sim") / datetime.now().strftime("%Y%m%d_%H%M%S")

    config = SlamConfig.load(args.config)

    run_sim(
        config=config,
        output_dir=output_dir,
        num_steps=args.steps,
        show_plots=not args.no_show_plots,
        save_plots=not args.no_save_plots,
        save_on_abort=not args.no_save_on_abort,
    )


if __name__ == "__main__":
    main()
