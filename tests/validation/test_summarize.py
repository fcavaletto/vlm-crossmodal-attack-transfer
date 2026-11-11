"""Counting rules for the locked v2 summary. No model is loaded."""

from __future__ import annotations

import pytest

from answer_scoring.class_loss import argmin_name
from pilot.rows import scoring_disagreement
from validation.protocol import FORMAT_LINE, LABEL, assert_not_checkpoint1, repo_root
from validation.summarize import build_summary

CONTEXTS = ("neutral", "helpful_nonrevealing", "contradictory", "answer_revealing")
PRIMARY = ("neutral", "helpful_nonrevealing")
COLORS = ("red", "blue", "green", "yellow")


def _losses(best: str, gap: float = 1.0) -> dict[str, float]:
    return {name: (0.2 if name == best else 0.2 + gap) for name in COLORS}


def _row(
    scene_id: str,
    group: str,
    context: str,
    *,
    raw: str,
    answer: str = "red",
    family: str = "attribute",
    best: str | None = None,
    ambiguity: list[str] | None = None,
    reveals: bool = False,
    context_text: str | None = None,
) -> dict:
    parsed = raw if raw in COLORS else None
    invalid = parsed is None
    correct = (not invalid) and parsed == answer
    valid_wrong = (not invalid) and not correct
    reasons: list[str] = [] if not invalid else ["not_in_vocabulary"]
    if raw == "sentence red":
        reasons = ["not_exact"]
    class_nll = _losses(best or (parsed or "blue"))
    word_nll = dict(class_nll)
    wrapped = {name: {"sum_nll": value} for name, value in class_nll.items()}
    class_argmin, _best, class_flag = scoring_disagreement(parsed, invalid, wrapped)
    word_argmin, _ignored, word_flag = scoring_disagreement(parsed, invalid, wrapped)
    if argmin_name(class_nll) != class_argmin:
        raise RuntimeError("test argmin drifted")
    text = context_text if context_text is not None else f"Shared {context} guidance."
    return {
        "label": LABEL,
        "format_line_appended": False,
        "scene_id": scene_id,
        "counterfactual_group_id": group,
        "task_family": family,
        "context": context,
        "answer": answer,
        "answer_aliases": [answer],
        "context_text": text,
        "question": "What color is the large circle?",
        "prompt": text + "\n" + "What color is the large circle?",
        "reveals_answer": reveals,
        "ambiguity_flags": [] if ambiguity is None else ambiguity,
        "raw_output": raw,
        "parsed_answer": parsed,
        "parse_reasons": reasons,
        "correct": correct,
        "valid_wrong": valid_wrong,
        "invalid": invalid,
        "truncated": False,
        "class_nll": class_nll,
        "class_argmin": class_argmin,
        "class_disagreement": class_flag,
        "word_only_nll": word_nll,
        "word_only_argmin": word_argmin,
        "word_only_disagreement": word_flag,
        "device": "mps",
    }


def _pair(group: str, left: str, right: str, *, left_raw: str, right_raw: str, **kwargs) -> list[dict]:
    rows = []
    for scene_id, raw in ((left, left_raw), (right, right_raw)):
        for context in CONTEXTS:
            rows.append(_row(scene_id, group, context, raw=raw, **kwargs))
    return rows


def _gradient(scene_id: str, group: str, *, l1: float = 10.0, finite: bool = True) -> list[dict]:
    rows = []
    for context in PRIMARY:
        rows.append(
            {
                "scene_id": scene_id,
                "counterfactual_group_id": group,
                "context": context,
                "finite": finite,
                "l1": l1 if finite else None,
                "loss": "canonical_word",
            }
        )
    return rows


def _summary(rows, gradients, *, full_pass: bool):
    scene_ids = sorted({row["scene_id"] for row in rows})
    return build_summary(
        rows,
        gradients,
        [],
        run_meta={"full_gradient_pass": full_pass, "full_gradient_pass_reason": None if full_pass else "test"},
        gate={"passed": True, "scene_id": "dev2-attribute-00-a"},
        expected_scene_ids=scene_ids,
        require_full_sample=False,
    )


def test_eligible_group_requires_both_contexts_and_a_positive_gradient():
    rows = _pair("g-clean", "s-a", "s-b", left_raw="red", right_raw="blue")
    rows += _pair("g-flat", "s-c", "s-d", left_raw="red", right_raw="blue")
    gradients = []
    gradients += _gradient("s-a", "g-clean")
    gradients += _gradient("s-b", "g-clean", l1=0.0, finite=True)
    gradients += _gradient("s-c", "g-flat", l1=0.0, finite=True)
    gradients += _gradient("s-d", "g-flat")
    # s-b is not clean-correct (raw blue, answer red). s-a is clean and finite.
    # s-c is clean but its gradient L1 is zero. s-d is not clean.
    summary = _summary(rows, gradients, full_pass=True)
    eligible = summary["eligible"]
    assert eligible["n_scenes_clean_both_primary"] == 2
    assert eligible["n_groups_clean_both_primary"] == 2
    assert eligible["n_groups_eligible"] == 1
    assert eligible["eligible_group_ids"] == ["g-clean"]
    assert eligible["meets_algebraic_minimum"] is False
    assert eligible["meets_coverage_description"] is False
    assert summary["h_star_fitted"] is False
    assert summary["attacks_run"] is False


def test_ambiguity_and_revealing_text_block_eligibility_without_dropping_the_row():
    rows = _pair("g1", "s-a", "s-b", left_raw="red", right_raw="red", ambiguity=["shared-cell"])
    gradients = _gradient("s-a", "g1") + _gradient("s-b", "g1")
    summary = _summary(rows, gradients, full_pass=True)
    assert summary["eligible"]["n_scenes_clean_both_primary"] == 2
    assert summary["eligible"]["n_groups_eligible"] == 0
    assert summary["clean_by_context"]["neutral"]["n_rows"] == 2


def test_primary_text_must_match_inside_the_group():
    rows = _pair("g1", "s-a", "s-b", left_raw="red", right_raw="red")
    for row in rows:
        if row["scene_id"] == "s-b" and row["context"] == "neutral":
            row["context_text"] = "A different neutral sentence."
            row["prompt"] = row["context_text"] + "\n" + row["question"]
    gradients = _gradient("s-a", "g1") + _gradient("s-b", "g1")
    summary = _summary(rows, gradients, full_pass=True)
    assert summary["eligible"]["n_groups_eligible"] == 0


def test_design_band_recommends_without_deleting_rows():
    rows = _pair("attr", "a1", "a2", left_raw="red", right_raw="red", family="attribute")
    # Both scenes correct is accuracy 1.0, which is above the 0.80 band.
    summary = _summary(rows, _gradient("a1", "attr") + _gradient("a2", "attr"), full_pass=True)
    attribute = summary["family_recommendations"]["by_family"]["attribute"]
    assert attribute["meets_design_target"] is False
    assert attribute["unsuitable"] is False
    assert attribute["recommendation"] == "misses_design_band"
    assert summary["family_recommendations"]["deletes_rows"] is False
    assert summary["clean_by_family_context"]["attribute"]["neutral"]["n_rows"] == 2


def test_half_the_rows_wrong_is_a_drop_recommendation_and_the_rows_remain():
    rows = _pair("g1", "s-a", "s-b", left_raw="blue", right_raw="blue")
    summary = _summary(rows, [], full_pass=False)
    attribute = summary["family_recommendations"]["by_family"]["attribute"]
    assert attribute["unsuitable"] is True
    assert attribute["recommendation"] == "drop_or_replace"
    assert summary["clean_by_family_context"]["attribute"]["neutral"]["n_correct"] == 0
    assert summary["clean_by_family_context"]["attribute"]["neutral"]["n_rows"] == 2
    assert summary["eligible"]["n_groups_eligible"] is None


def test_class_argmin_agreement_ignores_invalid_rows():
    rows = _pair("g1", "s-a", "s-b", left_raw="red", right_raw="maybe")
    for row in rows:
        if row["raw_output"] == "red":
            row["class_nll"] = _losses("blue")
            row["word_only_nll"] = dict(row["class_nll"])
            wrapped = {name: {"sum_nll": value} for name, value in row["class_nll"].items()}
            argmin, _best, flag = scoring_disagreement(row["parsed_answer"], row["invalid"], wrapped)
            row["class_argmin"] = argmin
            row["word_only_argmin"] = argmin
            row["class_disagreement"] = flag
            row["word_only_disagreement"] = flag
    summary = _summary(rows, [], full_pass=False)
    agreement = summary["agreement"]
    assert agreement["official_parse_matches_generation"]["value"] == 1.0
    assert agreement["n_valid_rows"] == 4
    assert agreement["class_argmin_matches_official_on_valid_rows"]["numerator"] == 0
    assert agreement["class_disagreement_rows"] == 4


def test_format_line_in_the_prompt_is_rejected():
    rows = _pair("g1", "s-a", "s-b", left_raw="red", right_raw="red")
    rows[0]["prompt"] = rows[0]["prompt"] + "\n" + FORMAT_LINE
    with pytest.raises(ValueError, match="format line"):
        _summary(rows, [], full_pass=False)


def test_checkpoint1_paths_are_refused():
    root = repo_root()
    with pytest.raises(RuntimeError):
        assert_not_checkpoint1(root / "docs" / "checkpoint-1" / "summary.json", root)
    assert_not_checkpoint1(root / "docs" / "checkpoint-1.5" / "development_v2" / "summary.json", root)


def test_validation_package_does_not_fit_hstar_or_name_an_attack():
    package = repo_root() / "src" / "validation" / "validation"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py")).lower()
    for banned in ("fit_linear", "analysis.hstar", "pgd", "fgsm"):
        assert banned not in text
