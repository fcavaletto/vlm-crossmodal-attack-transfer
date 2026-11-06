"""The H* precision sketch is a fabricated design with group dependence."""

from __future__ import annotations

import json
import math
from pathlib import Path

from analysis.hstar_simulation import recommend_eligible_groups, simulate_hstar_precision

# Last-bit gaps in median interval widths. One ulp near these widths is about
# 2e-16, so 1e-9 covers a unit in the last place and nothing that would move
# a reported coverage count.
MEDIAN_WIDTH_ABSOLUTE_TOLERANCE = 1e-9
MEDIAN_WIDTH_KEYS = frozenset({"median_ci_width_cluster", "median_ci_width_row"})

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = (
    ROOT
    / "src"
    / "analysis"
    / "analysis"
    / "fixtures"
    / "hstar"
    / "precision_simulation.json"
)


def test_same_seed_repeats_and_the_draw_is_labeled_fabricated():
    kwargs = {
        "n_groups_grid": (6, 12),
        "n_replications": 5,
        "n_boot": 40,
        "seed": 20261107,
    }
    first = simulate_hstar_precision(**kwargs)
    second = simulate_hstar_precision(**kwargs)
    assert first == second
    assert first["data_origin"] == "fabricated_dgp"
    assert first["not_an_empirical_finding"] is True
    assert first["empirical_claim"] is False
    assert first["known_beta_a"] == 0.25
    assert first["recommendation"]["does_not_promise_significance"] is True
    assert "does not promise" in first["recommendation"]["sentence"]
    assert "significant" not in first["recommendation"]["sentence"].lower()


def test_recommendation_uses_the_pre_specified_bar_and_can_abstain():
    meeting = {
        "by_n_groups": [
            {
                "n_groups": 6,
                "fraction_interval_returned": 1.0,
                "coverage_cluster_among_returned": 0.80,
            },
            {
                "n_groups": 12,
                "fraction_interval_returned": 1.0,
                "coverage_cluster_among_returned": 0.91,
            },
        ]
    }
    chosen = recommend_eligible_groups(meeting)
    assert chosen["recommended_minimum_eligible_groups"] == 12
    assert chosen["algebraic_minimum_groups"] == 5
    assert chosen["cited_checkpoint1_eligible_groups"] == 3
    assert chosen["cited_checkpoint1_below_algebraic_minimum"] is True
    assert chosen["does_not_promise_significance"] is True

    none = {
        "by_n_groups": [
            {
                "n_groups": 48,
                "fraction_interval_returned": 0.99,
                "coverage_cluster_among_returned": 0.99,
            }
        ]
    }
    abstain = recommend_eligible_groups(none)
    assert abstain["recommended_minimum_eligible_groups"] is None
    assert abstain["status"] == "no_grid_point_met_the_planning_bar"
    assert "does not promise" in abstain["sentence"]


def _assert_precision_match(rebuilt: object, committed: object, path: str = "$") -> None:
    """Exact agreement, except last-bit gaps on median interval widths."""
    if isinstance(committed, dict):
        assert isinstance(rebuilt, dict), path
        assert set(rebuilt) == set(committed), path
        for key in sorted(committed):
            _assert_precision_match(rebuilt[key], committed[key], f"{path}.{key}")
        return
    if isinstance(committed, list):
        assert isinstance(rebuilt, list), path
        assert len(rebuilt) == len(committed), path
        for index, (got, want) in enumerate(zip(rebuilt, committed)):
            _assert_precision_match(got, want, f"{path}[{index}]")
        return
    field = path.rsplit(".", 1)[-1]
    if field in MEDIAN_WIDTH_KEYS:
        assert isinstance(rebuilt, float) and isinstance(committed, float), path
        assert math.isfinite(rebuilt) and math.isfinite(committed), path
        gap = abs(rebuilt - committed)
        # One ulp near these widths is about 1e-16. 1e-9 admits that gap.
        assert gap <= MEDIAN_WIDTH_ABSOLUTE_TOLERANCE, (path, rebuilt, committed, gap)
        return
    assert rebuilt == committed, (path, rebuilt, committed)


def test_committed_precision_file_matches_the_generator():
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    rebuilt = simulate_hstar_precision()
    _assert_precision_match(rebuilt, committed)
    assert rebuilt["recommendation"] == committed["recommendation"]
    assert committed["recommendation"]["recommended_minimum_eligible_groups"] == 24
    assert "does not promise" in committed["recommendation"]["sentence"]
    for got, want in zip(rebuilt["by_n_groups"], committed["by_n_groups"], strict=True):
        assert got["fraction_interval_returned"] == want["fraction_interval_returned"]
        assert got["fraction_row_interval_returned"] == want["fraction_row_interval_returned"]
        assert got["cluster_status_counts"] == want["cluster_status_counts"]
        assert got["n_interval_returned"] == want["n_interval_returned"]
        assert got["coverage_cluster_among_returned"] == want["coverage_cluster_among_returned"]
        assert (
            got["coverage_cluster_counting_unreturned_as_not_covered"]
            == want["coverage_cluster_counting_unreturned_as_not_covered"]
        )
        assert got["coverage_row_among_returned"] == want["coverage_row_among_returned"]
    counts = {row["n_groups"]: row["coverage_cluster_among_returned"] for row in committed["by_n_groups"]}
    assert counts[6] < counts[12] < 0.90 <= counts[24]
    assert committed["by_n_groups"][0]["cluster_status_counts"]["flat_outcome"] == 1
    assert committed["by_n_groups"][0]["cluster_status_counts"]["near_flat_outcome"] == 2
