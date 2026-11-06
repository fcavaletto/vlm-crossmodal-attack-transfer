"""Rebuild the v2 development tables from compact rows.

This module does not load a model, does not score an image, and does not fit
H*. Counts are arithmetic on the rows the runner already stored.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from answer_scoring.class_loss import argmin_name
from answer_scoring.instruction import FORMAT_INSTRUCTION
from pilot.rows import judge_parse, scoring_disagreement
from vlm_iface.parsing import parse_generation

from .protocol import (
    ALGEBRAIC_MIN_GROUPS,
    CONTEXTS,
    COVERAGE_DESCRIPTION_GROUPS,
    DESIGN_TARGET_HIGH,
    DESIGN_TARGET_LOW,
    FORMAT_LINE,
    GRADIENT_LOSS,
    HELPFUL_TEXT_ROLE,
    LABEL,
    N_CONTEXTS,
    N_GROUPS,
    N_PER_FAMILY,
    N_SCENES,
    OFFICIAL_PARSER,
    PRIMARY_CONTEXTS,
    TASK_FAMILIES,
    UNSUITABLE_BELOW,
)

if FORMAT_INSTRUCTION != FORMAT_LINE:
    raise RuntimeError("The locked format line does not match Agent A's string.")


def proportion(numerator: int, denominator: int) -> dict[str, Any]:
    """A count and its ratio. An empty denominator is null, not zero."""
    if denominator == 0:
        return {"numerator": numerator, "denominator": 0, "value": None}
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
    }


def build_summary(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
    *,
    run_meta: Mapping[str, Any],
    gate: Mapping[str, Any],
    expected_scene_ids: Sequence[str],
    require_full_sample: bool,
) -> dict[str, Any]:
    """Scientific counts for one completed or deliberately partial gradient pass.

    ``require_full_sample`` is true for the committed 72-scene manifest. Tests
    of the counting rules pass a smaller expected list and set it false.
    """
    problems = _structure_problems(
        predictions,
        expected_scene_ids,
        require_full_sample=require_full_sample,
    )
    if problems:
        raise ValueError("; ".join(problems))
    _check_official_parse(predictions)
    _check_closed_vocabulary(predictions)
    clean_context = {context: _slice(predictions, context=context) for context in CONTEXTS}
    clean_family = {
        family: {context: _slice(predictions, context=context, family=family) for context in CONTEXTS}
        for family in TASK_FAMILIES
    }
    eligible = _eligible(predictions, gradients, run_meta)
    return {
        "label": LABEL,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
        "format_line_used": False,
        "official_parser": OFFICIAL_PARSER,
        "differentiated_gradient": GRADIENT_LOSS,
        "helpful_text_role": HELPFUL_TEXT_ROLE,
        "n_scenes": len(set(expected_scene_ids)),
        "n_groups": len({row["counterfactual_group_id"] for row in predictions}),
        "n_predictions": len(predictions),
        "clean_by_context": clean_context,
        "clean_by_family_context": clean_family,
        "counterfactual_pairs": _pairs(predictions),
        "agreement": _agreement(predictions),
        "eligible": eligible,
        "family_recommendations": _recommendations(clean_family),
        "gradients": _gradient_block(gradients, alignments, run_meta),
        "gate_passed": gate.get("passed") is True,
        "gate_scene_id": gate.get("scene_id"),
        "run_meta": dict(run_meta),
        "note": (
            "DEVELOPMENT ONLY. Strict D4 is the official score. "
            "The class-loss argmin is stored beside it and is not the official score. "
            "Image gradients are the canonical-word loss. No attack and no H* fit."
        ),
    }


def _structure_problems(
    predictions: Sequence[Mapping[str, Any]],
    expected_scene_ids: Sequence[str],
    *,
    require_full_sample: bool,
) -> list[str]:
    problems: list[str] = []
    expected = list(expected_scene_ids)
    if len(expected) != len(set(expected)):
        problems.append("expected scene ids are not unique")
    if require_full_sample and len(expected) != N_SCENES:
        problems.append(f"expected {N_SCENES} scenes, got {len(expected)}")
    found = [row["scene_id"] for row in predictions]
    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        problems.append(f"scene ids missing={missing[:8]} extra={extra[:8]}")
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_scene[row["scene_id"]].append(row)
        if row.get("label") != LABEL:
            problems.append(f"{row['scene_id']} is not labeled {LABEL}")
            break
        if row.get("format_line_appended") is not False:
            problems.append(f"{row['scene_id']} appended the format line")
            break
        if FORMAT_LINE in str(row.get("prompt", "")):
            problems.append(f"{row['scene_id']} prompt contains the format line")
            break
        if row.get("prompt") != row["context_text"] + "\n" + row["question"]:
            problems.append(f"{row['scene_id']} {row['context']} prompt is not D2")
            break
        if row.get("device") != "mps":
            problems.append(f"{row['scene_id']} is not an MPS row")
            break
    for scene_id, rows in by_scene.items():
        contexts = [row["context"] for row in rows]
        if sorted(contexts) != sorted(CONTEXTS) or len(contexts) != N_CONTEXTS:
            problems.append(f"{scene_id} does not have the four contexts")
            break
    if require_full_sample:
        groups = {row["counterfactual_group_id"] for row in predictions}
        if len(groups) != N_GROUPS:
            problems.append(f"expected {N_GROUPS} groups, got {len(groups)}")
        for family in TASK_FAMILIES:
            n_family = len({row["scene_id"] for row in predictions if row["task_family"] == family})
            if n_family != N_PER_FAMILY:
                problems.append(f"{family} has {n_family} scenes")
    return problems


def _check_official_parse(predictions: Sequence[Mapping[str, Any]]) -> None:
    """The official fields must be the strict parse of the stored generation."""
    for row in predictions:
        parsed = parse_generation(
            str(row["raw_output"]),
            truncated=bool(row["truncated"]),
            task_family=str(row["task_family"]),
        )
        if parsed.parsed_answer != row["parsed_answer"] or parsed.invalid != row["invalid"]:
            raise ValueError(f"{row['scene_id']} {row['context']} official parse does not match raw_output")
        if list(parsed.reasons) != list(row["parse_reasons"]):
            raise ValueError(f"{row['scene_id']} {row['context']} parse reasons do not match raw_output")
        judgment = judge_parse(
            parsed.parsed_answer,
            parsed.invalid,
            str(row["answer"]),
            list(row["answer_aliases"]),
        )
        if (
            judgment["correct"] != row["correct"]
            or judgment["valid_wrong"] != row["valid_wrong"]
            or judgment["invalid"] != row["invalid"]
        ):
            raise ValueError(f"{row['scene_id']} {row['context']} correct/valid-wrong/invalid disagree")
        if row["correct"] + row["valid_wrong"] + row["invalid"] != 1:
            raise ValueError(f"{row['scene_id']} {row['context']} partition is not one outcome")


def _check_closed_vocabulary(predictions: Sequence[Mapping[str, Any]]) -> None:
    for row in predictions:
        for key, flag in (
            ("class_nll", "class_argmin"),
            ("word_only_nll", "word_only_argmin"),
        ):
            scores = {name: float(value) for name, value in row[key].items()}
            if argmin_name(scores) != row[flag]:
                raise ValueError(f"{row['scene_id']} {row['context']} {flag} does not match the losses")
        for loss_key, flag_key in (
            ("class_nll", "class_disagreement"),
            ("word_only_nll", "word_only_disagreement"),
        ):
            wrapped = {name: {"sum_nll": float(value)} for name, value in row[loss_key].items()}
            _argmin, _best, flag = scoring_disagreement(
                row["parsed_answer"],
                bool(row["invalid"]),
                wrapped,
            )
            if flag != row[flag_key]:
                raise ValueError(f"{row['scene_id']} {row['context']} {flag_key} does not match the losses")


def _slice(
    predictions: Sequence[Mapping[str, Any]],
    *,
    context: str,
    family: str | None = None,
) -> dict[str, Any]:
    rows = [
        row
        for row in predictions
        if row["context"] == context and (family is None or row["task_family"] == family)
    ]
    n_correct = sum(bool(row["correct"]) for row in rows)
    n_valid_wrong = sum(bool(row["valid_wrong"]) for row in rows)
    n_invalid = sum(bool(row["invalid"]) for row in rows)
    n_rows = len(rows)
    if n_correct + n_valid_wrong + n_invalid != n_rows:
        raise ValueError(f"{family} {context} partition does not add up")
    return {
        "label": LABEL,
        "n_rows": n_rows,
        "n_correct": n_correct,
        "n_valid_wrong": n_valid_wrong,
        "n_invalid": n_invalid,
        "accuracy": proportion(n_correct, n_rows),
        "valid_wrong_rate": proportion(n_valid_wrong, n_rows),
        "invalid_rate": proportion(n_invalid, n_rows),
    }


def _pairs(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(dict)
    for row in predictions:
        grouped[row["counterfactual_group_id"]].setdefault(row["scene_id"], {})[row["context"]] = row
    by_context = {}
    for context in CONTEXTS:
        both_correct = 0
        both_valid = 0
        parsed_changed = 0
        parsed_unchanged = 0
        truths_differ = 0
        n_pairs = 0
        for scenes in grouped.values():
            if len(scenes) != 2:
                raise ValueError("a counterfactual group does not have two scenes")
            left, right = [scenes[scene_id][context] for scene_id in sorted(scenes)]
            n_pairs += 1
            truths_differ += int(left["answer"] != right["answer"])
            if left["correct"] and right["correct"]:
                both_correct += 1
            if not left["invalid"] and not right["invalid"]:
                both_valid += 1
                if left["parsed_answer"] == right["parsed_answer"]:
                    parsed_unchanged += 1
                else:
                    parsed_changed += 1
        by_context[context] = {
            "label": LABEL,
            "n_pairs": n_pairs,
            "true_answers_differ": proportion(truths_differ, n_pairs),
            "correct_on_both_arms": proportion(both_correct, n_pairs),
            "both_valid": proportion(both_valid, n_pairs),
            "parsed_answer_changed": proportion(parsed_changed, both_valid),
            "parsed_answer_unchanged": proportion(parsed_unchanged, both_valid),
        }
    return {"label": LABEL, "by_context": by_context}


def _agreement(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_rows = len(predictions)
    valid = [row for row in predictions if not row["invalid"]]
    class_match_valid = sum(row["class_argmin"] == row["parsed_answer"] for row in valid)
    word_match_valid = sum(row["word_only_argmin"] == row["parsed_answer"] for row in valid)
    class_flags = sum(bool(row["class_disagreement"]) for row in predictions)
    word_flags = sum(bool(row["word_only_disagreement"]) for row in predictions)
    return {
        "label": LABEL,
        "official_score": "strict parse of raw_output",
        "official_parse_matches_generation": proportion(n_rows, n_rows),
        "class_argmin_matches_official_on_valid_rows": proportion(class_match_valid, len(valid)),
        "word_argmin_matches_official_on_valid_rows": proportion(word_match_valid, len(valid)),
        "class_disagreement_rows": class_flags,
        "word_disagreement_rows": word_flags,
        "n_valid_rows": len(valid),
        "n_rows": n_rows,
        "note": (
            "Official versus generated answer is the strict parse of the stored string. "
            "A row reaches this summary only when that parse matches. "
            "Class-loss agreement is the argmin against that official label, on valid rows. "
            "Invalid rows are excluded from the class-loss rate, matching the Checkpoint 1 disagreement rule."
        ),
    }


def _scene_primary(predictions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    scenes: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in predictions:
        if row["context"] in PRIMARY_CONTEXTS:
            scenes[row["scene_id"]][row["context"]] = row
    return scenes


def _eligible(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    scenes = _scene_primary(predictions)
    clean_ids = []
    for scene_id in sorted(scenes):
        pair = scenes[scene_id]
        if set(pair) != set(PRIMARY_CONTEXTS):
            raise ValueError(f"{scene_id} is missing a primary context")
        if all(bool(pair[context]["correct"]) for context in PRIMARY_CONTEXTS):
            clean_ids.append(scene_id)
    group_of = {row["scene_id"]: row["counterfactual_group_id"] for row in predictions}
    clean_groups = sorted({group_of[scene_id] for scene_id in clean_ids})
    full_pass = run_meta.get("full_gradient_pass") is True
    gradient_ok = _finite_positive_scenes(gradients) if full_pass else set()
    blocked = _manifest_blocked_scenes(predictions)
    eligible_ids = [
        scene_id for scene_id in clean_ids if scene_id in gradient_ok and scene_id not in blocked
    ] if full_pass else []
    eligible_groups = sorted({group_of[scene_id] for scene_id in eligible_ids}) if full_pass else []
    n_groups = len(eligible_groups) if full_pass else None
    return {
        "label": LABEL,
        "definition": (
            "An eligible group has at least one scene that is strict-correct under both "
            "neutral and helpful_nonrevealing, whose canonical-word image gradients on "
            "those contexts are finite with L1 norm greater than zero, whose ambiguity "
            "flags are empty, whose primary contexts are marked reveals_answer false, "
            "and whose primary context strings are identical across the two scenes in "
            "the group. The count is not a fitted coefficient."
        ),
        "n_scenes_clean_both_primary": len(clean_ids),
        "n_groups_clean_both_primary": len(clean_groups),
        "scene_ids_clean_both_primary": clean_ids,
        "group_ids_clean_both_primary": clean_groups,
        "full_gradient_pass": full_pass,
        "full_gradient_pass_reason": run_meta.get("full_gradient_pass_reason"),
        "n_scenes_eligible": len(eligible_ids) if full_pass else None,
        "n_groups_eligible": n_groups,
        "eligible_scene_ids": eligible_ids if full_pass else None,
        "eligible_group_ids": eligible_groups if full_pass else None,
        "algebraic_minimum_groups": ALGEBRAIC_MIN_GROUPS,
        "coverage_description_groups": COVERAGE_DESCRIPTION_GROUPS,
        "meets_algebraic_minimum": None if n_groups is None else n_groups >= ALGEBRAIC_MIN_GROUPS,
        "meets_coverage_description": None if n_groups is None else n_groups >= COVERAGE_DESCRIPTION_GROUPS,
    }


def _manifest_blocked_scenes(predictions: Sequence[Mapping[str, Any]]) -> set[str]:
    """Scenes Agent D would exclude before reading the outcome.

    Ambiguity and ``reveals_answer`` come from the manifest. Primary-context
    strings must match inside the counterfactual group. None of these checks
    read a model score.
    """
    blocked: set[str] = set()
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(dict)
    for row in predictions:
        grouped[row["counterfactual_group_id"]].setdefault(row["scene_id"], {})[row["context"]] = row
        if row["context"] not in PRIMARY_CONTEXTS:
            continue
        if row.get("ambiguity_flags"):
            blocked.add(row["scene_id"])
        if row.get("reveals_answer") is not False:
            blocked.add(row["scene_id"])
    for scenes in grouped.values():
        for context in PRIMARY_CONTEXTS:
            texts = {scenes[scene_id][context]["context_text"] for scene_id in scenes}
            if len(texts) != 1:
                blocked.update(scenes)
    return blocked


def _finite_positive_scenes(gradients: Sequence[Mapping[str, Any]]) -> set[str]:
    by_scene: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in gradients:
        if row["context"] in PRIMARY_CONTEXTS:
            by_scene[row["scene_id"]][row["context"]] = row
    ok: set[str] = set()
    for scene_id, pair in by_scene.items():
        if set(pair) != set(PRIMARY_CONTEXTS):
            continue
        good = True
        for context in PRIMARY_CONTEXTS:
            row = pair[context]
            l1 = row.get("l1")
            if row.get("finite") is not True or l1 is None or float(l1) <= 0.0:
                good = False
            if row.get("loss") != GRADIENT_LOSS:
                good = False
        if good:
            ok.add(scene_id)
    return ok


def _recommendations(clean_family: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    by_family = {}
    for family in TASK_FAMILIES:
        primary = {}
        accuracies = []
        for context in PRIMARY_CONTEXTS:
            cell = clean_family[family][context]
            primary[context] = cell["accuracy"]
            accuracies.append(cell["accuracy"]["value"])
        unsuitable = any(value is not None and value < UNSUITABLE_BELOW for value in accuracies)
        in_band = all(
            value is not None and DESIGN_TARGET_LOW <= value <= DESIGN_TARGET_HIGH for value in accuracies
        )
        if unsuitable:
            recommendation = "drop_or_replace"
        elif in_band:
            recommendation = "retain"
        else:
            recommendation = "misses_design_band"
        by_family[family] = {
            "primary_accuracy": primary,
            "unsuitable": unsuitable,
            "meets_design_target": in_band,
            "recommendation": recommendation,
        }
    return {
        "label": LABEL,
        "design_band": [DESIGN_TARGET_LOW, DESIGN_TARGET_HIGH],
        "unsuitable_below": UNSUITABLE_BELOW,
        "deletes_rows": False,
        "by_family": by_family,
        "note": (
            "The band is the locked reading of about 70 percent. "
            "A family below one half in either primary context is recommended for "
            "drop or replace. Its rows stay in the table."
        ),
    }


def _gradient_block(
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
    run_meta: Mapping[str, Any],
) -> dict[str, Any]:
    by_context = {}
    for context in PRIMARY_CONTEXTS:
        rows = [row for row in gradients if row["context"] == context]
        finite = [row for row in rows if row.get("finite") is True]
        by_context[context] = {
            "n_rows": len(rows),
            "n_finite": len(finite),
            "n_nonpositive_l1": sum(
                row.get("finite") is True and (row.get("l1") is None or float(row["l1"]) <= 0.0) for row in rows
            ),
            "loss": GRADIENT_LOSS,
        }
    defined = [row for row in alignments if row.get("alignment_a") is not None]
    return {
        "label": LABEL,
        "loss": GRADIENT_LOSS,
        "full_gradient_pass": run_meta.get("full_gradient_pass") is True,
        "by_context": by_context,
        "n_alignment_rows": len(alignments),
        "n_alignment_defined": len(defined),
        "regressed": False,
    }
