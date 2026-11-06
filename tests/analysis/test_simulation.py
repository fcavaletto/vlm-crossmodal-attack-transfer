"""The precision sketch uses a fabricated distribution and a fixed seed."""

from __future__ import annotations

import pytest

from analysis.simulation import simulate_paired_difference_precision


def test_wider_samples_have_narrower_intervals_under_the_fabricated_dgp():
    first = simulate_paired_difference_precision(
        n_scenes_grid=(12, 48),
        n_replications=40,
        n_boot=200,
        seed=20261106,
        p_plus=0.3,
        p_minus=0.05,
    )
    second = simulate_paired_difference_precision(
        n_scenes_grid=(12, 48),
        n_replications=40,
        n_boot=200,
        seed=20261106,
        p_plus=0.3,
        p_minus=0.05,
    )
    assert first == second
    assert first["data_origin"] == "fabricated_dgp"
    assert first["not_an_empirical_finding"] is True
    assert first["dgp"]["mean"] == pytest.approx(0.25)
    widths = [row["median_ci_width"] for row in first["by_n"]]
    assert widths[0] > widths[1] > 0
    assert all(0 < row["coverage"] <= 1 for row in first["by_n"])


def test_zero_variance_interval_has_width_zero_and_covers_the_mean():
    result = simulate_paired_difference_precision(
        n_scenes_grid=(8,),
        n_replications=5,
        n_boot=20,
        seed=1,
        p_plus=0.0,
        p_minus=0.0,
    )
    assert result["dgp"]["mean"] == 0
    assert result["by_n"][0]["median_ci_width"] == 0
    assert result["by_n"][0]["coverage"] == 1


def test_simulation_rejects_impossible_probabilities():
    with pytest.raises(ValueError):
        simulate_paired_difference_precision(p_plus=0.8, p_minus=0.3)
    with pytest.raises(ValueError):
        simulate_paired_difference_precision(n_scenes_grid=())
    with pytest.raises(ValueError):
        simulate_paired_difference_precision(n_replications=0)
