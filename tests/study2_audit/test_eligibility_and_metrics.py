"""Nine clean-both groups are not an eligible-group count and not an H* fit."""

from __future__ import annotations

import json
from pathlib import Path

from study2_audit.eligibility import explain_nine, minimum_groups
from study2_audit.metrics import KICKOFF_METRICS, metric_catalog

ROOT = Path(__file__).resolve().parents[2]


def _predictions():
    rows = []
    path = ROOT / "docs/checkpoint-1.6/development_simple/predictions.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_algebraic_floors_follow_the_parameter_count():
    assert minimum_groups(2) == 3
    assert minimum_groups(4) == 5


def test_nine_clean_groups_do_not_establish_h_star():
    gate = json.loads((ROOT / "docs/checkpoint-1.6/development_simple/gate.json").read_text(encoding="utf-8"))
    report = explain_nine(_predictions(), gate)
    assert report["sampling_unit"] == "counterfactual_group"
    assert report["n_groups_clean_both_primary"] == 9
    assert report["n_scenes_clean_both_primary"] == 14
    assert report["n_clean_groups_with_both_scenes"] == 5
    assert report["n_clean_groups_with_one_scene"] == 4
    assert report["n_groups_eligible"] is None
    assert report["eligible_count_is_the_clean_both_count"] is False
    assert report["gradient_scene_id"] == "dev3-row-00-a"
    assert report["gradient_scene_is_clean_both_primary"] is False
    assert report["nine_is_above_both_algebraic_floors"] is True
    assert report["nine_meets_coverage_description"] is False
    assert report["twenty_four_groups_automatically_enough"] is False
    assert report["establishes_h_star"] is False
    assert report["h_star_fitted"] is False
    assert report["group_ids_clean_both_primary"] == [
        "cf3-row-00",
        "cf3-row-01",
        "cf3-row-02",
        "cf3-row-05",
        "cf3-row-06",
        "cf3-row-07",
        "cf3-row-08",
        "cf3-row-10",
        "cf3-row-11",
    ]


def test_kickoff_metrics_stay_distinct_from_h_star_and_from_summed_nll():
    catalog = metric_catalog()
    names = catalog["kickoff_names"]
    assert names == [item["name"] for item in KICKOFF_METRICS]
    assert len(names) == len(set(names)) == 6
    assert catalog["h_star"]["equals_any_kickoff_metric"] is False
    assert catalog["h_star"]["fitted_in_this_package"] is False
    assert catalog["prospective_primary_score"]["name"] == "teacher_forced_summed_nll"
    assert catalog["prospective_primary_score"]["adopted_as_checkpoint_1_6_official"] is False
    assert catalog["official_checkpoint_1_6"]["name"] not in names
    assert catalog["positive_control"]["counts_as_visual_robustness"] is False
    attack_metrics = {item["name"] for item in KICKOFF_METRICS if item["requires_attack"]}
    assert attack_metrics == {
        "attack_success_rate",
        "additional_error",
        "cross_context_attack_transfer",
    }
