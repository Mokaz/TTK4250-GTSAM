"""Where the data and the run outputs live.

The data directory is looked up rather than hard-coded, so the code works
whether you run from the handout root, from a subdirectory, or from a checkout
that keeps the (large) data set one level up.
"""

from __future__ import annotations

from pathlib import Path

RUNS_ROOT = Path("runs")
FIGURES_ROOT = Path("figures")

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the first ``data/`` directory that actually exists.

    Searched, in order: the current working directory, the package root, and
    the directory above the package root.
    """
    candidates = [
        Path("data"),
        _PACKAGE_ROOT / "data",
        _PACKAGE_ROOT.parent / "data",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find the data directory. Looked in: "
        + ", ".join(str(c.resolve()) for c in candidates)
    )


def simulated_data_file() -> Path:
    return data_root() / "simulated" / "simulatedSLAM.mat"


def victoria_park_folder() -> Path:
    return data_root() / "victoria_park" / "raw"
