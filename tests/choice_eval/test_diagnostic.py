"""Free generation stays a separate diagnostic. The strict parser is unchanged."""

from __future__ import annotations

import pytest

from choice_eval.colors import COLORS
from choice_eval.diagnostic import diagnose_free_generation
from choice_eval.score import score_example, summarize


def _row(peaked, generation: str, *, label: str = "red", truncated: bool = False, margin: float = 3.0):
    logits = {color: peaked(1, 0.0) for color in COLORS}
    logits[label] = peaked(1, margin)
    token_ids = {color: [1] for color in COLORS}
    return score_example(
        logits,
        token_ids,
        label=label,
        generation_text=generation,
        generation_truncated=truncated,
    )


def test_parser_keeps_correct_valid_but_wrong_and_invalid_separate():
    correct = diagnose_free_generation("Red.", truncated=False, label="red")
    wrong = diagnose_free_generation("blue", truncated=False, label="red")
    invalid = diagnose_free_generation("The color is red.", truncated=False, label="red")
    truncated = diagnose_free_generation("red", truncated=True, label="red")
    assert correct["correct"] and not correct["valid_but_wrong"] and not correct["invalid"]
    assert correct["parsed_answer"] == "red"
    assert wrong["valid_but_wrong"] and wrong["parsed_answer"] == "blue"
    assert invalid["invalid"] and invalid["parsed_answer"] is None
    assert invalid["reasons"] == ["not_exact"]
    assert truncated["invalid"] and "truncated" in truncated["reasons"]
    for row in (correct, wrong, invalid, truncated):
        assert row["is_primary"] is False
        assert row["role"] == "secondary_diagnostic"
        assert row["parser"] == "vlm_iface.parsing.parse_generation"


def test_closed_vocabulary_choice_ignores_the_generation_string(peaked):
    valid = _row(peaked, "blue")
    invalid = _row(peaked, "The color is red.")
    assert valid["chosen"] == invalid["chosen"] == "red"
    assert valid["closed_vocabulary_correct"] is True
    assert valid["free_generation"]["valid_but_wrong"] is True
    assert invalid["free_generation"]["invalid"] is True
    assert invalid["closed_vocabulary_correct"] is True


def test_summary_does_not_blend_the_two_metrics(peaked):
    rows = [
        _row(peaked, "red"),
        _row(peaked, "blue"),
        _row(peaked, "The color is red."),
        _row(peaked, "yellow", label="yellow"),
    ]
    # The fourth row's logits still favor its label, and the string is correct.
    summary = summarize(rows)
    assert summary["n"] == 4
    assert summary["closed_vocabulary_correct"] == 4
    assert summary["free_generation_correct"] == 2
    assert summary["free_generation_valid_but_wrong"] == 1
    assert summary["free_generation_invalid"] == 1
    assert summary["free_generation_strict_accuracy"] == 0.5
    assert summary["free_generation_invalid_rate"] == 0.25
    assert summary["closed_vocabulary_accuracy"] == 1.0
    assert summary["metrics_combined"] is False
    assert "blended_accuracy" not in summary
    partition = (
        summary["free_generation_correct"]
        + summary["free_generation_valid_but_wrong"]
        + summary["free_generation_invalid"]
    )
    assert partition == summary["n"]


def test_summary_requires_the_diagnostic(peaked):
    row = _row(peaked, "red")
    row["free_generation"] = None
    with pytest.raises(ValueError, match="diagnostic"):
        summarize([row])


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        diagnose_free_generation("red", truncated=False, label="purple")
