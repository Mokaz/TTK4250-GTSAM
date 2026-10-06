"""The configuration plumbing. Not graded, but a broken config breaks everything."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphslam.config import (
    AssociationConfig,
    BackendConfig,
    NoiseConfig,
    SlamConfig,
    TentativeLandmarkManagerConfig,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["sim_default.yaml", "real_default.yaml", "default_config.yaml"])
def test_shipped_configs_load(name: str) -> None:
    config = SlamConfig.load(CONFIGS / name)
    assert isinstance(config, SlamConfig)


def test_default_config_matches_the_dataclass_defaults() -> None:
    assert SlamConfig.load(CONFIGS / "default_config.yaml") == SlamConfig()


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    expected = SlamConfig()

    expected.save(path)

    assert SlamConfig.load(path) == expected


def test_noise_parameters_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        NoiseConfig(sigma_range=0.0)


def test_measurement_covariance_is_ordered_range_then_bearing() -> None:
    noise = NoiseConfig(sigma_range=0.5, sigma_bearing_deg=1.0)
    R = noise.range_bearing_cov_matrix

    assert R[0, 0] == pytest.approx(0.25)
    assert R[1, 1] == pytest.approx(noise.sigma_bearing_rad**2)


def test_confirmation_window_must_be_consistent() -> None:
    with pytest.raises(ValueError):
        TentativeLandmarkManagerConfig(M=5, N=2)


def test_unknown_association_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        AssociationConfig(method="magic")


def test_unknown_solver_is_rejected() -> None:
    with pytest.raises(ValueError):
        BackendConfig(solver="wishful-thinking")


def test_victoria_park_initial_heading_is_configurable() -> None:
    assert SlamConfig().victoria_park.initial_heading_deg == 36.0
    assert SlamConfig.load(CONFIGS / "real_default.yaml").victoria_park.initial_heading_deg == 36.0

    try:
        from graphslam.loaders.victoria_park import VictoriaParkLoader

        loader = VictoriaParkLoader(initial_heading_deg=10.0)
    except FileNotFoundError:
        pytest.skip("Victoria Park data set not found")

    assert loader.initial_pose[2] == pytest.approx(np.deg2rad(10.0))
