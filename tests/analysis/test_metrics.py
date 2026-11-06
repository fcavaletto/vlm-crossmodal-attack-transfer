"""Metric arithmetic on toy rows and on the committed fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.metrics import (
    FLOOR_WARNING,
    additional_error_by_scene,
    additional_error_summary,
    attack_cells,
    attack_success_indicators,
    attack_success_rate,
    clean_by_context,
    conditional_transfer,
    cross_context_attack_transfer,
    mean_loss_by_context,
    pearson_correlation,
    proportion,
    sign_alignment,
)
from analysis.schemas import (
    load_jsonl,
    validate_attacks,
    validate_gradient_summaries,
    validate_gradient_vectors,
    validate_losses,
    validate_predictions,
)

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "src" / "analysis" / "analysis" / "fixtures" / "toy"


def _load():
    return {
        "predictions": validate_predictions(load_jsonl(TOY / "predictions.jsonl")),
        "losses": validate_losses(load_jsonl(TOY / "losses.jsonl")),
        "attacks": validate_attacks(load_jsonl(TOY / "attacks.jsonl")),
        "gradients": validate_gradient_summaries(load_jsonl(TOY / "gradient_summaries.jsonl")),
        "vectors": validate_gradient_vectors(load_jsonl(TOY / "gradient_vectors.jsonl")),
    }


def _prediction(scene, context, correct, invalid=False):
    return {
        "scene_id": scene,
        "context": context,
        "parsed_answer": "" if invalid else ("alpha" if correct else "beta"),
        "invalid": invalid,
        "correct": correct,
    }


def _attack(scene, before, after, invalid_after=False, **extra):
    row = {
        "scene_id": scene,
        "source_context": "neutral",
        "target_context": "neutral",
        "method": "pgd",
        "epsilon": 0.01,
        "step_budget": 1,
        "correct_before": before,
        "correct_after": after,
        "invalid_after": invalid_after,
    }
    row.update(extra)
    return row


def test_zero_denominator_is_null_not_zero():
    empty = proportion(0, 0, empty_reason="empty")
    assert empty["value"] is None
    assert empty["undefined_reason"] == "empty"
    zero_successes = proportion(0, 4, empty_reason="empty")
    assert zero_successes["value"] == 0.0
    assert zero_successes["undefined_reason"] is None
    with pytest.raises(ValueError):
        proportion(3, 2, empty_reason="empty")


def test_missing_context_does_not_become_zero_accuracy():
    rows = [
        _prediction("toy-01", "neutral", True),
        _prediction("toy-02", "neutral", False),
    ]
    clean = clean_by_context(rows)
    assert clean["neutral"]["accuracy"]["value"] == 0.5
    assert clean["contradictory"]["n_scenes"] == 0
    assert clean["contradictory"]["accuracy"]["value"] is None
    assert "no scenes" in clean["contradictory"]["accuracy"]["undefined_reason"]
    assert clean["answer_revealing"]["positive_control_only"] is True
    assert clean["neutral"]["analysis_role"] == "proposed_secondary"


def test_strict_accuracy_counts_invalid_as_incorrect():
    rows = [
        _prediction("toy-01", "neutral", False, invalid=True),
        _prediction("toy-02", "neutral", True),
    ]
    block = clean_by_context(rows)["neutral"]
    assert block["n_correct"] == 1
    assert block["n_invalid"] == 1
    assert block["accuracy"]["value"] == 0.5
    assert block["invalid_output_rate"]["value"] == 0.5


def test_invalid_attack_output_is_not_a_success_and_stays_in_the_denominator():
    rows = [
        _attack("toy-01", True, False, False),
        _attack("toy-02", True, False, True),
        _attack("toy-03", True, True, False),
        _attack("toy-04", False, False, False),
    ]
    rate = attack_success_rate(rows)
    assert rate["n_success"] == 1
    assert rate["baseline_correct_n"] == 3
    assert rate["n_invalid_after_among_baseline"] == 1
    assert rate["attack_success_rate"]["value"] == pytest.approx(1 / 3)
    nobody = attack_success_rate([_attack("toy-04", False, False, False)])
    assert nobody["attack_success_rate"]["value"] is None
    assert nobody["attack_success_rate"]["denominator"] == 0


def test_additional_error_counts_invalid_as_task_error():
    rows = [
        _attack("toy-01", True, False, True),
        _attack("toy-02", False, True, False),
        _attack("toy-03", True, True, False),
    ]
    deltas = additional_error_by_scene(rows)
    assert deltas == {"toy-01": 1.0, "toy-02": -1.0, "toy-03": 0.0}
    # error_after - error_before = correct_before - correct_after for these flags.
    for row, delta in zip(rows, deltas.values()):
        assert delta == float(row["correct_before"]) - float(row["correct_after"])
    summary = additional_error_summary(rows)
    assert summary["mean"] == 0.0
    assert additional_error_summary([])["mean"] is None


def test_floor_warning_appears_when_clean_error_is_at_least_half():
    high = additional_error_summary(
        [_attack("toy-01", False, False), _attack("toy-02", False, True)]
    )
    low = additional_error_summary(
        [_attack("toy-01", True, False), _attack("toy-02", True, True)]
    )
    assert high["floor_warning"] == FLOOR_WARNING
    assert low["floor_warning"] is None


def test_alignment_matches_an_independent_pure_python_oracle():
    source = [1, -2, 0, 0.5]
    target = [3, -1, 4, -2]

    def oracle(g_source, g_target):
        l1 = sum(abs(value) for value in g_target)
        if l1 == 0:
            return None
        dot = 0.0
        for left, right in zip(g_target, g_source):
            sign = 0.0 if right == 0 else (1.0 if right > 0 else -1.0)
            dot += left * sign
        return dot / l1

    result = sign_alignment(source, target)
    assert result["dot"] == 2
    assert result["l1_target"] == 10
    assert result["alignment_a"] == pytest.approx(oracle(source, target))
    assert result["alignment_a"] == pytest.approx(0.2)
    assert sign_alignment([1, 1], [2, 4])["alignment_a"] == 1
    assert sign_alignment([1, 1], [-2, -4])["alignment_a"] == -1
    assert sign_alignment([0, 0], [3, -1])["alignment_a"] == 0
    flat = sign_alignment([1, -1], [0, 0])
    assert flat["alignment_a"] is None
    assert flat["undefined_reason"]
    negative_zero = sign_alignment([-0.0], [4.0])
    assert negative_zero["alignment_a"] == 0.0
    shaped = sign_alignment([[1, -1], [0, 2]], [[1, 1], [3, -4]])
    assert shaped["alignment_a"] == pytest.approx(-4 / 9)


def test_alignment_rejects_bad_arrays():
    with pytest.raises(ValueError, match="shape"):
        sign_alignment([[1, 2], [3, 4]], [1, 2, 3, 4])
    with pytest.raises(ValueError, match="empty"):
        sign_alignment([], [])
    with pytest.raises(ValueError, match="finite"):
        sign_alignment([float("nan")], [1.0])
    with pytest.raises(ValueError, match="boolean"):
        sign_alignment([True, False], [1.0, -1.0])


def test_cross_context_cells_omit_same_context_and_the_positive_control():
    rows = [
        _attack("toy-01", True, False),
        _attack("toy-01", True, False, source_context="neutral", target_context="contradictory"),
        _attack(
            "toy-01",
            True,
            False,
            source_context="neutral",
            target_context="answer_revealing",
        ),
    ]
    transfer = cross_context_attack_transfer(rows)
    assert [cell["cell_id"] for cell in transfer] == [
        "pgd|0.01000000|1|neutral|contradictory"
    ]
    revealing = [
        cell
        for cell in attack_cells(rows)
        if cell["target_context"] == "answer_revealing"
    ]
    assert revealing[0]["positive_control_only"] is True
    assert revealing[0]["analysis_role"] == "positive_control"


def test_conditional_transfer_excludes_scenes_already_wrong_on_the_target():
    rows = [
        _attack("toy-01", True, False, source_context="neutral", target_context="neutral"),
        _attack("toy-02", True, False, source_context="neutral", target_context="neutral"),
        _attack("toy-01", True, False, source_context="neutral", target_context="contradictory"),
        _attack("toy-02", False, False, source_context="neutral", target_context="contradictory"),
    ]
    result = conditional_transfer(
        rows,
        method="pgd",
        epsilon=0.01,
        step_budget=1,
        source_context="neutral",
        target_context="contradictory",
    )
    assert result["analysis_role"] == "exploratory"
    assert result["n_source_success"] == 2
    assert result["n_excluded_target_not_correct_before"] == 1
    assert result["rate"]["numerator"] == 1
    assert result["rate"]["denominator"] == 1
    empty = conditional_transfer(
        [_attack("toy-01", False, False), _attack("toy-01", False, False, target_context="contradictory")],
        method="pgd",
        epsilon=0.01,
        step_budget=1,
        source_context="neutral",
        target_context="contradictory",
    )
    assert empty["rate"]["value"] is None


def test_pearson_edges():
    assert pearson_correlation([0, 1, 2], [0, 1, 2])["value"] == pytest.approx(1)
    assert pearson_correlation([0, 1, 2], [2, 1, 0])["value"] == pytest.approx(-1)
    assert pearson_correlation([1, 1, 1], [0, 1, 2])["value"] is None
    assert pearson_correlation([0, 1], [0, 1])["undefined_reason"] == "fewer than 3 paired scenes"


def test_committed_toy_fixture_has_the_worked_fractions():
    packet = _load()
    clean = clean_by_context(packet["predictions"])
    assert clean["neutral"]["n_correct"] == 5
    assert clean["neutral"]["n_invalid"] == 0
    assert clean["helpful_nonrevealing"]["n_correct"] == 4
    assert clean["contradictory"]["n_correct"] == 2
    assert clean["contradictory"]["n_invalid"] == 1
    assert clean["contradictory"]["floor_warning"] == FLOOR_WARNING
    assert clean["neutral"]["floor_warning"] is None
    assert clean["answer_revealing"]["n_correct"] == 5
    assert clean["answer_revealing"]["n_invalid"] == 1
    losses = mean_loss_by_context(packet["losses"])
    assert losses["neutral"]["mean_loss"] == pytest.approx(0.5)
    assert losses["contradictory"]["mean_loss"] == pytest.approx(0.7)
    assert losses["answer_revealing"]["mean_loss"] == pytest.approx(0.6)

    cells = {cell["cell_id"]: cell for cell in attack_cells(packet["attacks"])}
    same = cells["pgd|0.01000000|1|neutral|neutral"]
    assert same["attack_success_rate"]["n_success"] == 3
    assert same["attack_success_rate"]["baseline_correct_n"] == 5
    assert same["attack_success_rate"]["n_invalid_after_among_baseline"] == 1
    assert same["additional_error"]["mean"] == pytest.approx(4 / 6)
    indicators = attack_success_indicators(
        [
            row
            for row in packet["attacks"]
            if row["method"] == "pgd"
            and row["source_context"] == "neutral"
            and row["target_context"] == "neutral"
            and abs(row["epsilon"] - 0.01) < 1e-12
        ]
    )
    assert indicators["toy-05"] == (1, 0)
    assert additional_error_by_scene(
        [
            row
            for row in packet["attacks"]
            if row["scene_id"] == "toy-05"
            and row["method"] == "pgd"
            and row["source_context"] == "neutral"
            and row["target_context"] == "neutral"
        ]
    )["toy-05"] == 1.0

    noise = cells["noise|0.02000000|1|neutral|neutral"]
    assert noise["n_scenes"] == 1
    assert noise["attack_success_rate"]["attack_success_rate"]["value"] is None
    left = additional_error_by_scene(
        [
            row
            for row in packet["attacks"]
            if row["method"] == "pgd"
            and row["source_context"] == "neutral"
            and row["target_context"] == "contradictory"
            and abs(row["epsilon"] - 0.01) < 1e-12
        ]
    )
    right = additional_error_by_scene(
        [
            row
            for row in packet["attacks"]
            if row["method"] == "pgd"
            and row["source_context"] == "neutral"
            and row["target_context"] == "neutral"
            and abs(row["epsilon"] - 0.01) < 1e-12
        ]
    )
    differences = [left[scene] - right[scene] for scene in sorted(left)]
    assert differences == [0.0, -1.0, 0.0, -1.0, -1.0, -1.0]
    assert sum(differences) / len(differences) == pytest.approx(-4 / 6)

    transfer = conditional_transfer(
        packet["attacks"],
        method="pgd",
        epsilon=0.01,
        step_budget=1,
        source_context="helpful_nonrevealing",
        target_context="contradictory",
    )
    assert transfer["n_source_success"] == 4
    assert transfer["n_excluded_target_not_correct_before"] == 2
    assert transfer["rate"]["numerator"] == 1
    assert transfer["rate"]["denominator"] == 2

    stored = {
        (row["scene_id"], row["source_context"], row["target_context"]): row["alignment_a"]
        for row in packet["gradients"]
    }
    for row in packet["vectors"]:
        key = (row["scene_id"], row["source_context"], row["target_context"])
        recomputed = sign_alignment(row["g_source"], row["g_target"])["alignment_a"]
        if stored[key] is None:
            assert recomputed is None
        else:
            assert recomputed == pytest.approx(stored[key])
    assert stored[("toy-01", "neutral", "helpful_nonrevealing")] == pytest.approx(0.2)
    assert stored[("toy-04", "neutral", "helpful_nonrevealing")] is None
