"""H* estimator on fabricated rows. No development scene is fit."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.hstar import (
    DEVELOPMENT_GROUP_IDS,
    DEVELOPMENT_SCENE_IDS,
    HStarRefused,
    exclusion_reasons,
    fit_hstar,
    fit_linear_design,
    interval_covers_zero,
    interval_entirely_above_zero,
    mechanical_reading,
    minimum_groups,
    validate_hstar_rows,
)
from analysis.hstar_packet import build_worked_packet
from analysis.schemas import SchemaError

PRIMARY = ("neutral", "helpful_nonrevealing")


def _row(**overrides):
    source = overrides.pop("source_context", "neutral")
    target = overrides.pop("target_context", "helpful_nonrevealing")
    strict = overrides.pop("strict_error", False)
    row = {
        "data_origin": "fabricated",
        "scene_id": "fab-s0",
        "counterfactual_group_id": "fab-g0",
        "task_family": "attribute",
        "source_context": source,
        "target_context": target,
        "method": "pgd",
        "epsilon": 0.01,
        "step_budget": 2,
        "alignment_a": 0.0,
        "clean_correct_source": True,
        "clean_correct_target": True,
        "source_reveals_answer": False,
        "target_reveals_answer": False,
        "source_contains_answer": False,
        "target_contains_answer": False,
        "primary_contexts_invariant": True,
        "ambiguity_flagged": False,
        "gradients_finite": True,
        "gradient_source_l1": 4.0,
        "gradient_target_l1": 4.0,
        "clean_confidence": 0.6,
        "strict_error": strict,
        "invalid_target": False,
        "valid_wrong_target": strict,
        "source_strict_error": True,
    }
    row.update(overrides)
    return row


def _balanced_packet(
    *,
    n_per_family: int = 2,
    alignment_negative: float = -1.0,
    alignment_positive: float = 1.0,
    methods=("pgd", "fgsm", "noise"),
    confidence_of=None,
    magnitude_of=None,
    outcome_of=None,
    source_error_of=None,
):
    """Exact two-direction rows. Default PGD outcome is 1 only when alignment is positive."""
    families = ("attribute", "counting", "spatial")
    rows = []
    index = 0
    for family in families:
        for copy in range(n_per_family):
            group = f"fab-{family}-{copy}"
            scene = f"{group}-s0"
            for source, target, alignment in (
                ("neutral", "helpful_nonrevealing", alignment_negative),
                ("helpful_nonrevealing", "neutral", alignment_positive),
            ):
                for method in methods:
                    if outcome_of is None:
                        strict = alignment > 0
                        if method == "noise":
                            strict = not strict
                    else:
                        strict = outcome_of(method, alignment, group)
                    row = _row(
                        scene_id=scene,
                        counterfactual_group_id=group,
                        task_family=family,
                        source_context=source,
                        target_context=target,
                        method=method,
                        alignment_a=alignment,
                        strict_error=strict,
                        valid_wrong_target=strict,
                        invalid_target=False,
                        clean_confidence=0.6 if confidence_of is None else confidence_of(alignment),
                        gradient_target_l1=4.0 if magnitude_of is None else magnitude_of(alignment),
                        source_strict_error=True
                        if source_error_of is None
                        else source_error_of(group),
                    )
                    rows.append(row)
                    index += 1
    return rows


def test_hand_solved_slope_is_zero_when_alignment_is_crossed_with_the_outcome():
    rows = []
    pattern = [(0.0, False, "g0"), (1.0, False, "g0"), (0.0, True, "g1"), (1.0, True, "g1")]
    for index, (alignment, strict, group) in enumerate(pattern):
        rows.append(
            _row(
                scene_id=f"fab-hand-{index}",
                counterfactual_group_id=group,
                alignment_a=alignment,
                strict_error=strict,
                valid_wrong_target=strict,
            )
        )
    fitted = fit_hstar(rows, n_boot=20, seed=1)
    assert fitted["primary"]["status"] == "too_few_groups"
    assert fitted["primary"]["n_parameters"] == 2
    assert fitted["primary"]["minimum_groups"] == 3
    point = fitted["primary"]["diagnostic_point_estimate"]
    assert point["intercept"] == pytest.approx(0.5)
    assert point["alignment_a"] == pytest.approx(0.0)
    assert fitted["primary"]["interval"] is None
    assert fitted["confirmatory_supported"] is False


def test_hand_solved_unit_slope():
    rows = []
    pattern = [(0.0, False, "g0"), (1.0, True, "g0"), (0.0, False, "g1"), (1.0, True, "g1")]
    for index, (alignment, strict, group) in enumerate(pattern):
        rows.append(
            _row(
                scene_id=f"fab-unit-{index}",
                counterfactual_group_id=group,
                alignment_a=alignment,
                strict_error=strict,
                valid_wrong_target=strict,
            )
        )
    fitted = fit_hstar(rows, n_boot=20, seed=1)
    point = fitted["primary"]["diagnostic_point_estimate"]
    assert point["intercept"] == pytest.approx(0.0)
    assert point["alignment_a"] == pytest.approx(1.0)


def test_worked_packet_matches_the_closed_form_and_keeps_both_directions():
    fitted = fit_hstar(build_worked_packet(), n_boot=40, seed=20261107)
    primary = fitted["primary"]
    assert primary["status"] == "interval_returned"
    assert primary["n_groups"] == 6
    assert primary["n_parameters"] == 4
    assert primary["minimum_groups"] == minimum_groups(4)
    assert primary["zero_residual_fit"] is True
    point = primary["diagnostic_point_estimate"]
    assert point["intercept"] == pytest.approx(0.5)
    assert point["alignment_a"] == pytest.approx(0.5)
    assert point["task_counting"] == pytest.approx(0.0)
    assert point["task_spatial"] == pytest.approx(0.0)
    interval = primary["interval"]["alignment_a"]
    assert interval["ci_low"] == pytest.approx(0.5)
    assert interval["ci_high"] == pytest.approx(0.5)
    assert fitted["inclusion"]["direction_counts"] == {
        "neutral_to_helpful": 6,
        "helpful_to_neutral": 6,
    }
    assert fitted["inclusion"]["n_scenes_with_both_directions"] == 6
    assert fitted["inclusion"]["n_scenes_with_one_direction"] == 0
    assert fitted["controls"]["noise"]["diagnostic_point_estimate"]["alignment_a"] == pytest.approx(-0.5)
    assert fitted["logistic_sensitivity"]["status"] == "separation"
    assert fitted["logistic_sensitivity"]["interval"] is None
    assert fitted["decomposition"]["invalid_only_pattern"] is False
    assert fitted["decomposition"]["invalid_target"]["status"] == "flat_outcome"
    assert fitted["mechanical_reading"] == "interval_above_zero"
    assert fitted["confirmatory_supported"] is False
    assert fitted["not_an_empirical_finding"] is True
    assert fitted["empirical_claim"] is False
    swapped = fit_hstar(build_worked_packet(), n_boot=40, seed=20261107, reference_task="counting")
    assert swapped["primary"]["diagnostic_point_estimate"]["alignment_a"] == pytest.approx(0.5)
    assert "task_attribute" in swapped["primary"]["columns"]
    assert "task_counting" not in swapped["primary"]["columns"]


def test_exclusion_ignores_the_outcome_and_counts_each_gate():
    packet = build_worked_packet()
    excluded = [row for row in packet if row["scene_id"].startswith("fab-ex-")]
    reasons = {row["scene_id"]: exclusion_reasons(row) for row in excluded}
    assert reasons["fab-ex-source-clean"] == ["not_clean_correct_both"]
    assert reasons["fab-ex-target-clean"] == ["not_clean_correct_both"]
    assert reasons["fab-ex-contains"] == ["answer_leakage"]
    assert reasons["fab-ex-reveals"] == ["answer_leakage"]
    assert reasons["fab-ex-invariant"] == ["contexts_not_group_invariant"]
    assert reasons["fab-ex-ambiguity"] == ["ambiguity_flagged"]
    assert reasons["fab-ex-zero-gradient"] == ["gradient_not_finite_or_zero", "undefined_alignment"]
    assert reasons["fab-ex-nonfinite"] == ["gradient_not_finite_or_zero"]
    assert reasons["fab-ex-undefined-alignment"] == ["undefined_alignment"]
    assert reasons["fab-ex-context"] == ["context_outside_primary"]
    mutated = dict(excluded[0])
    mutated["strict_error"] = True
    mutated["valid_wrong_target"] = True
    assert exclusion_reasons(mutated) == exclusion_reasons(excluded[0])
    fitted = fit_hstar(packet, n_boot=20, seed=1)
    assert fitted["inclusion"]["n_rows_excluded"] == 10
    assert fitted["inclusion"]["exclusion_counts"]["answer_leakage"] == 2
    assert fitted["primary"]["n_rows"] == 12


def test_too_few_groups_flat_predictor_and_flat_outcome_return_no_interval():
    families = ("attribute", "counting", "spatial")
    few = []
    for family in families:
        for direction, alignment in (("neutral", -1.0), ("helpful_nonrevealing", 1.0)):
            other = "helpful_nonrevealing" if direction == "neutral" else "neutral"
            strict = alignment > 0
            few.append(
                _row(
                    scene_id=f"fab-few-{family}-{direction}",
                    counterfactual_group_id=f"fab-few-{family}",
                    task_family=family,
                    source_context=direction,
                    target_context=other,
                    alignment_a=alignment,
                    strict_error=strict,
                    valid_wrong_target=strict,
                )
            )
    few_fit = fit_hstar(few, n_boot=20, seed=1)
    assert few_fit["primary"]["status"] == "too_few_groups"
    assert few_fit["primary"]["n_groups"] == 3
    assert few_fit["primary"]["interval"] is None
    assert few_fit["primary"]["evidence"] is False
    assert few_fit["mechanical_reading"] == "non_identifiable"

    flat_x = _balanced_packet(methods=("pgd",))
    for row in flat_x:
        row["alignment_a"] = 0.25
        row["strict_error"] = row["source_context"] == "neutral"
        row["valid_wrong_target"] = row["strict_error"]
    flat_x_fit = fit_hstar(flat_x, n_boot=20, seed=1)
    assert flat_x_fit["primary"]["status"] == "flat_predictor"
    assert flat_x_fit["primary"]["interval"] is None

    flat_y = _balanced_packet(methods=("pgd",))
    for row in flat_y:
        row["strict_error"] = True
        row["valid_wrong_target"] = True
        row["invalid_target"] = False
    flat_y_fit = fit_hstar(flat_y, n_boot=20, seed=1)
    assert flat_y_fit["primary"]["status"] == "flat_outcome"
    assert flat_y_fit["primary"]["diagnostic_point_estimate"]["alignment_a"] == pytest.approx(0.0)
    assert flat_y_fit["primary"]["interval"] is None


def test_near_flat_outcome_is_not_identifiable_even_when_an_interval_is_returned():
    rows = _balanced_packet(methods=("pgd",), n_per_family=4)
    # 24 rows. 23 errors gives mean 23/24 > 0.95 without making the outcome constant.
    for row in rows:
        row["strict_error"] = True
        row["valid_wrong_target"] = True
        row["invalid_target"] = False
    rows[0]["strict_error"] = False
    rows[0]["valid_wrong_target"] = False
    fitted = fit_hstar(rows, n_boot=30, seed=3)
    assert fitted["primary"]["status"] == "interval_returned"
    assert fitted["primary"]["near_flat_outcome"] is True
    assert fitted["primary"]["identifiable"] is False
    assert fitted["mechanical_reading"] == "non_identifiable"
    assert fitted["confirmatory_supported"] is False


def test_multiple_budgets_are_refused():
    rows = _balanced_packet(methods=("pgd",))
    rows[0]["epsilon"] = 0.02
    fitted = fit_hstar(rows, n_boot=20, seed=1)
    assert fitted["primary"]["status"] == "multiple_budgets"
    assert fitted["primary"]["interval"] is None
    assert fitted["mechanical_reading"] == "non_identifiable"


def test_development_scenes_and_non_fabricated_rows_are_refused():
    scene_id = sorted(DEVELOPMENT_SCENE_IDS)[0]
    with pytest.raises(HStarRefused):
        validate_hstar_rows([_row(scene_id=scene_id)])
    with pytest.raises(HStarRefused):
        validate_hstar_rows([_row(scene_id="dev-not-a-committed-id")])
    group_id = sorted(DEVELOPMENT_GROUP_IDS)[0]
    with pytest.raises(HStarRefused):
        validate_hstar_rows([_row(counterfactual_group_id=group_id)])
    with pytest.raises(HStarRefused):
        validate_hstar_rows([_row(data_origin="development")])
    with pytest.raises(SchemaError):
        validate_hstar_rows([_row(unexpected=True)])
    with pytest.raises(SchemaError):
        validate_hstar_rows([_row(alignment_a=1.2)])
    duplicate = [_row(scene_id="fab-dup"), _row(scene_id="fab-dup")]
    with pytest.raises(SchemaError):
        validate_hstar_rows(duplicate)
    broken = _row(strict_error=True, invalid_target=True, valid_wrong_target=True)
    with pytest.raises(SchemaError):
        validate_hstar_rows([broken])


def test_invalid_only_pattern_and_predictor_comparison():
    def outcome(method, alignment, _group):
        positive = alignment > 0
        if method == "noise":
            positive = not positive
        return positive

    rows = _balanced_packet(methods=("pgd", "fgsm", "noise"), outcome_of=outcome)
    for row in rows:
        if row["method"] == "pgd" and row["strict_error"]:
            row["invalid_target"] = True
            row["valid_wrong_target"] = False
        elif row["method"] == "pgd":
            row["invalid_target"] = False
            row["valid_wrong_target"] = False
    fitted = fit_hstar(rows, n_boot=30, seed=4)
    assert fitted["decomposition"]["invalid_only_pattern"] is True
    assert fitted["mechanical_reading"] == "association_in_invalid_outputs"

    compared = _balanced_packet(
        methods=("pgd", "fgsm", "noise"),
        alignment_negative=-0.5,
        alignment_positive=0.5,
        confidence_of=lambda alignment: 0.5 + 0.5 * alignment,
        magnitude_of=lambda alignment: 2.0 + alignment,
        outcome_of=lambda _method, alignment, _group: alignment > 0,
    )
    comparison = fit_hstar(compared, n_boot=30, seed=5)
    block = comparison["predictor_comparison"]
    assert block["pearson_alignment_confidence"] == pytest.approx(1.0)
    assert block["pearson_alignment_magnitude"] == pytest.approx(1.0)
    assert block["high_correlation"] == ["clean_confidence", "gradient_magnitude"]
    assert block["clean_confidence"]["status"] == "interval_returned"
    assert block["gradient_magnitude"]["status"] == "interval_returned"
    assert block["joint_diagnostic"]["status"] == "singular_design"
    assert block["joint_diagnostic"]["not_a_primary_estimand"] is True
    assert "not a primary estimand" in block["note"]


def test_source_slice_sign_disagreement_uses_returned_intervals_only():
    def source_error(group):
        return int(group.rsplit("-", 1)[-1]) < 2

    def outcome(method, alignment, group):
        if method != "pgd":
            return alignment < 0
        positive = alignment > 0
        if not source_error(group):
            positive = not positive
        return positive

    rows = _balanced_packet(
        methods=("pgd", "fgsm", "noise"),
        n_per_family=4,
        source_error_of=source_error,
        outcome_of=outcome,
    )
    fitted = fit_hstar(rows, n_boot=30, seed=6)
    stratification = fitted["source_stratification"]
    assert stratification["sign_disagreement"] is True
    success = stratification["source_success"]["diagnostic_point_estimate"]["alignment_a"]
    failure = stratification["source_failure"]["diagnostic_point_estimate"]["alignment_a"]
    assert success == pytest.approx(0.5)
    assert failure == pytest.approx(-0.5)
    assert fitted["mechanical_reading"] == "pooled_coefficient_is_not_one_mechanism"


def test_cluster_interval_changes_when_dependence_is_respected():
    rng = np.random.default_rng(11)
    rows_per_group = 8
    n_groups = 10
    x_matrix = np.zeros((n_groups * rows_per_group, 2), dtype=np.float64)
    outcome = np.zeros(n_groups * rows_per_group, dtype=np.float64)
    clusters = np.empty(n_groups * rows_per_group, dtype=np.int64)
    cursor = 0
    for group in range(n_groups):
        shock = 0.35 if group % 2 == 0 else -0.35
        for _ in range(rows_per_group):
            alignment = float(rng.uniform(-0.5, 0.5))
            x_matrix[cursor, 0] = 1.0
            x_matrix[cursor, 1] = alignment
            probability = np.clip(0.5 + 0.2 * alignment + shock, 0.02, 0.98)
            outcome[cursor] = 1.0 if rng.random() < probability else 0.0
            clusters[cursor] = group
            cursor += 1
    clustered = fit_linear_design(x_matrix, outcome, clusters, n_boot=300, seed=19, alpha=0.05)
    rowwise = fit_linear_design(
        x_matrix,
        outcome,
        np.arange(outcome.shape[0]),
        n_boot=300,
        seed=19,
        alpha=0.05,
    )
    clustered_again = fit_linear_design(
        x_matrix, outcome, clusters, n_boot=300, seed=19, alpha=0.05
    )
    assert clustered == clustered_again
    assert clustered["intervals"][1]["ci_low"] != rowwise["intervals"][1]["ci_low"]
    assert clustered["intervals"][1]["ci_high"] != rowwise["intervals"][1]["ci_high"]


def test_reading_table_is_a_function_of_intervals_only():
    assert mechanical_reading(
        identifiable=False,
        pgd=(0.2, 0.4),
        fgsm=(0.2, 0.4),
        noise=(-0.2, 0.1),
        invalid_only_pattern=False,
        sign_disagreement=False,
    ) == "non_identifiable"
    assert mechanical_reading(
        identifiable=True,
        pgd=(0.2, 0.4),
        fgsm=None,
        noise=(-0.2, 0.1),
        invalid_only_pattern=False,
        sign_disagreement=False,
    ) == "controls_not_supplied"
    assert mechanical_reading(
        identifiable=True,
        pgd=(0.2, 0.4),
        fgsm=(0.1, 0.2),
        noise=(0.05, 0.3),
        invalid_only_pattern=False,
        sign_disagreement=False,
    ) == "alignment_confounded_with_difficulty"
    assert mechanical_reading(
        identifiable=True,
        pgd=(-0.1, 0.2),
        fgsm=(0.1, 0.4),
        noise=(-0.3, 0.05),
        invalid_only_pattern=False,
        sign_disagreement=False,
    ) == "fgsm_above_pgd_covers_zero"
    assert mechanical_reading(
        identifiable=True,
        pgd=(-0.2, -0.05),
        fgsm=(-0.2, 0.1),
        noise=(-0.3, 0.05),
        invalid_only_pattern=False,
        sign_disagreement=False,
    ) == "interval_below_zero"
    assert interval_entirely_above_zero(0.0, 0.2) is False
    assert interval_covers_zero(-0.1, 0.0) is True


def test_singular_design_is_reported_for_a_collinear_matrix():
    alignment = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], dtype=np.float64)
    x_matrix = np.column_stack(
        [
            np.ones(6),
            alignment,
            0.5 + 0.5 * alignment,
        ]
    )
    outcome = (alignment > 0).astype(np.float64)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    fitted = fit_linear_design(
        x_matrix,
        outcome,
        clusters,
        n_boot=20,
        seed=1,
        alpha=0.05,
        check_focal_flat=False,
    )
    assert fitted["status"] == "singular_design"
    assert fitted["intervals"] is None


def test_empty_packet_returns_no_interval_and_no_claim():
    fitted = fit_hstar([], n_boot=10, seed=1)
    assert fitted["primary"]["status"] == "no_included_rows"
    assert fitted["confirmatory_supported"] is False
    assert fitted["not_an_empirical_finding"] is True
    assert fitted["mechanical_reading"] == "non_identifiable"


def test_one_direction_can_be_included_without_the_other():
    rows = _balanced_packet(methods=("pgd", "fgsm", "noise"), n_per_family=2)
    kept = [
        row
        for row in rows
        if not (
            row["method"] == "pgd"
            and row["source_context"] == "helpful_nonrevealing"
            and row["scene_id"] == "fab-attribute-0-s0"
        )
    ]
    fitted = fit_hstar(kept, n_boot=20, seed=1)
    assert fitted["inclusion"]["n_scenes_with_one_direction"] == 1
    assert fitted["inclusion"]["direction_counts"]["helpful_to_neutral"] == 5
    assert fitted["primary"]["status"] == "interval_returned"
