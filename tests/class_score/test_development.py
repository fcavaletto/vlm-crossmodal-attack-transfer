"""Counts from the committed v2 predictions. No weights are loaded."""

from __future__ import annotations

from pathlib import Path

from class_score.development import audit_development, in_design_band

ROOT = Path(__file__).resolve().parents[2]

# strict, class argmin, word argmin, valid-wrong, invalid. Denominator is every row.
EXPECTED = {
    "all": {
        "neutral": (26, 30, 29, 39, 7, 72),
        "helpful_nonrevealing": (22, 26, 22, 28, 22, 72),
        "contradictory": (17, 22, 24, 41, 14, 72),
        "answer_revealing": (45, 50, 53, 16, 11, 72),
    },
    "attribute": {
        "neutral": (10, 11, 13, 9, 5, 24),
        "helpful_nonrevealing": (0, 5, 6, 2, 22, 24),
        "contradictory": (10, 11, 12, 8, 6, 24),
        "answer_revealing": (20, 22, 24, 2, 2, 24),
    },
    "counting": {
        "neutral": (10, 12, 8, 12, 2, 24),
        "helpful_nonrevealing": (10, 10, 4, 14, 0, 24),
        "contradictory": (0, 0, 0, 24, 0, 24),
        "answer_revealing": (23, 23, 24, 1, 0, 24),
    },
    "spatial": {
        "neutral": (6, 7, 8, 18, 0, 24),
        "helpful_nonrevealing": (12, 11, 12, 12, 0, 24),
        "contradictory": (7, 11, 12, 9, 8, 24),
        "answer_revealing": (2, 5, 5, 13, 9, 24),
    },
}


def test_v2_counts_match_the_stored_predictions_without_dropping_rows():
    summary = audit_development(ROOT)
    assert summary["n_rows"] == 288
    assert summary["rows_dropped"] == 0
    assert summary["label"] == "DEVELOPMENT ONLY"
    assert summary["official_metric"] == "strict_free_generation"
    assert summary["free_generation_role"] == "official"
    assert summary["class_argmin_is_primary_outcome"] is False
    assert summary["embedded_reading_adopted"] is False
    assert summary["class_argmin_meets_band_for_every_family"] is False
    assert summary["band"]["strict_every_family_in_band"] is False
    for family, contexts in EXPECTED.items():
        for context, expected in contexts.items():
            cell = summary["by_family"][family][context]
            strict, class_correct, word, valid_wrong, invalid, n = expected
            assert cell == {
                "n": n,
                "strict_correct": strict,
                "valid_wrong": valid_wrong,
                "invalid": invalid,
                "class_argmin_correct": class_correct,
                "word_argmin_correct": word,
            }
            if context in {"neutral", "helpful_nonrevealing"} and family != "all":
                assert in_design_band(strict, n) is False
                assert in_design_band(class_correct, n) is False
                assert in_design_band(word, n) is False
    assert in_design_band(30, 72) is False
    assert in_design_band(26, 72) is False
    assert 30 / 72 < 0.5
    assert 26 / 72 < 0.5


def test_valid_rows_agree_with_the_class_argmin_except_four_spatial_rows():
    summary = audit_development(ROOT)
    agreement = summary["agreement"]
    assert agreement == {
        "n_valid": 234,
        "n_invalid": 54,
        "class_argmin_matches_parsed": 230,
        "word_argmin_matches_parsed": 179,
        "class_disagreement_flags": 4,
        "word_disagreement_flags": 55,
    }
    rows = summary["class_disagreement_rows"]
    assert len(rows) == 4
    assert {row["scene_id"] for row in rows} == {
        "dev2-spatial-01-a",
        "dev2-spatial-02-a",
        "dev2-spatial-04-b",
        "dev2-spatial-06-b",
    }
    assert all(row["gap"] > 1e-6 for row in rows)


def test_digit_removal_is_not_recomputed_from_pooled_losses_and_embedded_reading_stays_unselected():
    summary = audit_development(ROOT)
    digits = summary["digit_generations"]
    assert digits["n"] == 94
    assert digits["class_argmin_equals_parsed"] == 94
    assert digits["word_argmin_equals_parsed"] == 50
    gap = summary["decision_gap"]
    assert gap["class_argmin_differs_from_word_counting"] == 46
    assert gap["class_argmin_differs_from_word_color"] == 21
    embedded = summary["embedded_counterfactual"]
    assert embedded["adopted"] is False
    assert embedded["n_applicable"] == 50
    assert embedded["n_equals_answer"] == 26
    assert embedded["n_differs_from_answer"] == 24
    adopted = {
        "all": {
            "neutral": 30,
            "helpful_nonrevealing": 32,
            "contradictory": 23,
            "answer_revealing": 51,
        },
        "attribute": {
            "neutral": 14,
            "helpful_nonrevealing": 10,
            "contradictory": 10,
            "answer_revealing": 22,
        },
        "counting": {
            "neutral": 10,
            "helpful_nonrevealing": 10,
            "contradictory": 0,
            "answer_revealing": 23,
        },
        "spatial": {
            "neutral": 6,
            "helpful_nonrevealing": 12,
            "contradictory": 13,
            "answer_revealing": 6,
        },
    }
    for family, contexts in adopted.items():
        n = 72 if family == "all" else 24
        for context, correct in contexts.items():
            assert embedded["by_family"][family][context] == {"n": n, "correct_if_adopted": correct}
            if context in {"neutral", "helpful_nonrevealing"} and family != "all":
                assert in_design_band(correct, n) is False
    assert in_design_band(30, 72) is False
    assert in_design_band(32, 72) is False


def test_both_primary_clean_correct_counts():
    summary = audit_development(ROOT)
    both = summary["both_primary"]
    assert both["strict"]["all"] == {"scenes": 13, "groups": 11}
    assert both["strict"]["attribute"] == {"scenes": 0, "groups": 0}
    assert both["class_argmin"]["all"] == {"scenes": 20, "groups": 18}
    assert both["class_argmin"]["attribute"] == {"scenes": 4, "groups": 4}
    assert both["class_argmin"]["counting"] == {"scenes": 10, "groups": 8}
    assert both["class_argmin"]["spatial"] == {"scenes": 6, "groups": 6}
    assert both["word_argmin"]["attribute"] == {"scenes": 3, "groups": 3}
    assert both["word_argmin"]["counting"] == {"scenes": 3, "groups": 3}
    assert both["word_argmin"]["spatial"] == {"scenes": 6, "groups": 6}
    assert both["word_argmin"]["all"] == {"scenes": 12, "groups": 12}
