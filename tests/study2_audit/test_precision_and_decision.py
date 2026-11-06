"""Fabricated cluster precision, and a no-go that does not fit real scenes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from study2_audit.decision import decide_scene_count
from study2_audit.eligibility import explain_nine
from study2_audit.precision import (
    RealDataRefused,
    fit_fabricated_rows,
    lower_endpoint_excludes_zero,
    simulate_precision,
    worked_zero_residual_example,
)

ROOT = Path(__file__).resolve().parents[2]


def test_an_endpoint_of_zero_does_not_count_as_above_zero():
    assert lower_endpoint_excludes_zero(0.0) is False
    assert lower_endpoint_excludes_zero(-0.01) is False
    assert lower_endpoint_excludes_zero(0.01) is True


def test_zero_residual_example_collapses_and_is_not_evidence():
    fitted = worked_zero_residual_example()
    assert fitted["status"] == "interval_returned"
    assert fitted["interval"]["estimate"] == 1.0
    assert fitted["interval"]["ci_low"] == 1.0
    assert fitted["interval"]["ci_high"] == 1.0
    assert fitted["zero_residual_fit"] is True
    assert fitted["evidence"] is False
    assert fitted["h_star_fitted_on_real_data"] is False
    assert fitted["excludes_zero"] is True


def test_two_groups_do_not_return_a_two_parameter_interval():
    rows = []
    for group in range(2):
        for arm, alignment in enumerate((-0.5, 0.5)):
            rows.append(
                {
                    "data_origin": "fabricated",
                    "scene_id": f"fab-g{group}-s{arm}",
                    "counterfactual_group_id": f"fab-g{group}",
                    "alignment_a": alignment,
                    "strict_error": arm == 1,
                }
            )
    fitted = fit_fabricated_rows(rows, n_boot=20, seed=1)
    assert fitted["status"] == "too_few_groups"
    assert fitted["interval"] is None
    assert fitted["evidence"] is False


def test_a_constant_outcome_has_no_interval():
    rows = []
    for group in range(3):
        rows.append(
            {
                "data_origin": "fabricated",
                "scene_id": f"fab-g{group}-s0",
                "counterfactual_group_id": f"fab-g{group}",
                "alignment_a": -0.2 if group == 0 else 0.2,
                "strict_error": True,
            }
        )
    fitted = fit_fabricated_rows(rows, n_boot=20, seed=1)
    assert fitted["status"] == "flat_outcome"
    assert fitted["interval"] is None


def test_development_ids_are_refused_before_a_fit():
    row = {
        "data_origin": "fabricated",
        "scene_id": "dev3-row-00-a",
        "counterfactual_group_id": "fab-g0",
        "alignment_a": 0.1,
        "strict_error": True,
    }
    with pytest.raises(RealDataRefused):
        fit_fabricated_rows([row, row], n_boot=10, seed=1)
    row["scene_id"] = "fab-g0-s0"
    row["counterfactual_group_id"] = "cf3-row-00"
    with pytest.raises(RealDataRefused):
        fit_fabricated_rows([row], n_boot=10, seed=1)
    row["counterfactual_group_id"] = "fab-g0"
    row["data_origin"] = "development"
    with pytest.raises(RealDataRefused):
        fit_fabricated_rows([row], n_boot=10, seed=1)


def test_the_same_seed_rebuilds_the_fabricated_grid():
    kwargs = {"n_groups_grid": (9, 48), "n_replications": 15, "n_boot": 40, "seed": 20261108}
    first = simulate_precision(**kwargs)
    second = simulate_precision(**kwargs)
    assert first == second
    assert first["data_origin"] == "fabricated_dgp"
    assert first["empirical_claim"] is False
    assert first["h_star_fitted_on_real_data"] is False
    assert first["twenty_four_groups_automatically_enough"] is False
    assert "not automatically enough" in first["reading"]
    assert "not a measurement" in first["reading"]


def test_documented_grid_covers_the_coefficient_more_often_than_it_excludes_zero():
    report = simulate_precision()
    by_n = {row["n_groups"]: row for row in report["by_n_groups"]}
    assert set(by_n) == {9, 12, 24, 48}
    for n_groups, row in by_n.items():
        assert row["fraction_interval_returned"] == 1.0
        assert row["fraction_covered_among_returned"] > row["fraction_excludes_zero_among_returned"]
        assert row["fraction_excludes_zero_among_returned"] < 0.75
        assert row["n_scenes"] == n_groups * 2
    assert by_n[9]["median_width"] > by_n[48]["median_width"]
    assert by_n[9]["fraction_excludes_zero_among_returned"] < 0.5
    assert by_n[48]["median_width"] > 0.25
    assert by_n[24]["fraction_excludes_zero_among_returned"] < 0.6


def test_a_flattering_precision_grid_still_gets_a_no_go():
    gate = json.loads((ROOT / "docs/checkpoint-1.6/development_simple/gate.json").read_text(encoding="utf-8"))
    predictions = []
    with (ROOT / "docs/checkpoint-1.6/development_simple/predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                predictions.append(json.loads(line))
    eligibility = explain_nine(predictions, gate)
    precision = {
        "data_origin": "fabricated_dgp",
        "h_star_fitted_on_real_data": False,
        "by_n_groups": [
            {
                "n_groups": n_groups,
                "n_scenes": n_groups * 2,
                "fraction_interval_returned": 1.0,
                "fraction_covered_among_returned": 1.0,
                "fraction_excludes_zero_among_returned": 1.0,
                "median_width": 0.01,
                "median_width_row": 0.01,
            }
            for n_groups in (9, 24, 48)
        ],
    }
    decision = decide_scene_count(eligibility, precision)
    assert decision["decision"] == "no-go"
    assert decision["supports_h_star"] is False
    assert decision["blocks_h_star_claim"] is True
    assert decision["blocks_scene_construction"] is False
    assert decision["twenty_four_groups_automatically_enough"] is False
    assert decision["nine_groups_establish_h_star"] is False
    assert decision["n_groups_eligible"] is None
    assert decision["hypothetical_product_is_a_forecast"] is False
    assert decision["hypothetical_clean_both_product"] == 36
    assert decision["h_star_fitted"] is False
    assert decision["attacks_observed"] is False


def test_a_real_fit_flag_is_refused():
    with pytest.raises(RuntimeError):
        decide_scene_count(
            {"h_star_fitted": True, "n_groups_clean_both_primary": 9, "n_groups": 12, "n_groups_eligible": None},
            {"data_origin": "fabricated_dgp", "h_star_fitted_on_real_data": False, "by_n_groups": []},
        )
