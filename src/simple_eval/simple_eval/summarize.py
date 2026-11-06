"""Counts for the row-color clean evaluation. No model and no H* fit."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from vlm_iface.parsing import parse_generation

from answer_scoring.class_loss import argmin_name
from pilot.rows import judge_parse, scoring_disagreement

LABEL = "DEVELOPMENT ONLY"
CONTEXTS = ("neutral", "helpful_nonrevealing", "contradictory", "answer_revealing")
PRIMARY = ("neutral", "helpful_nonrevealing")
PARSE_FAMILY = "attribute"
DESIGN_LOW = 0.60
DESIGN_HIGH = 0.80
UNSUITABLE_BELOW = 0.50


def proportion(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"numerator": numerator, "denominator": 0, "value": None}
    return {"numerator": numerator, "denominator": denominator, "value": numerator / denominator}


def build_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    run_meta: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    _check(predictions)
    by_context = {context: _slice(predictions, context) for context in CONTEXTS}
    clean_ids, clean_groups = _clean_both(predictions)
    primary = {context: by_context[context]["accuracy"]["value"] for context in PRIMARY}
    unsuitable = any(value is not None and value < UNSUITABLE_BELOW for value in primary.values())
    in_band = all(value is not None and DESIGN_LOW <= value <= DESIGN_HIGH for value in primary.values())
    valid = [row for row in predictions if not row["invalid"]]
    class_match = sum(row["class_argmin"] == row["parsed_answer"] for row in valid)
    return {
        "label": LABEL,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
        "official_metric": "strict_free_generation",
        "n_scenes": len({row["scene_id"] for row in predictions}),
        "n_groups": len({row["counterfactual_group_id"] for row in predictions}),
        "n_predictions": len(predictions),
        "clean_by_context": by_context,
        "counterfactual_pairs": _pairs(predictions),
        "class_argmin_matches_official_on_valid_rows": proportion(class_match, len(valid)),
        "n_valid_rows": len(valid),
        "n_scenes_clean_both_primary": len(clean_ids),
        "n_groups_clean_both_primary": len(clean_groups),
        "scene_ids_clean_both_primary": clean_ids,
        "group_ids_clean_both_primary": clean_groups,
        "meets_design_band": in_band,
        "unsuitable": unsuitable,
        "recommendation": "do_not_adopt" if unsuitable else ("retain_for_later_review" if in_band else "outside_band"),
        "gate_attempted": gate.get("attempted"),
        "gate_passed": gate.get("passed"),
        "run_meta": dict(run_meta),
    }


def _check(predictions: Sequence[Mapping[str, Any]]) -> None:
    if len({row["scene_id"] for row in predictions}) != 24:
        raise ValueError("the row-color table is not 24 scenes")
    for row in predictions:
        if row["task_family"] != "row_color":
            raise ValueError("task family is not row_color")
        if row["prompt"] != row["context_text"] + "\n" + row["question"]:
            raise ValueError("prompt is not the locked join")
        if "Reply using only the answer" in row["prompt"]:
            raise ValueError("format line was appended")
        parsed = parse_generation(str(row["raw_output"]), truncated=bool(row["truncated"]), task_family=PARSE_FAMILY)
        if parsed.parsed_answer != row["parsed_answer"] or parsed.invalid != row["invalid"]:
            raise ValueError(f"{row['scene_id']} {row['context']} parse does not match the generation")
        judgment = judge_parse(parsed.parsed_answer, parsed.invalid, str(row["answer"]), list(row["answer_aliases"]))
        if judgment["correct"] != row["correct"] or judgment["valid_wrong"] != row["valid_wrong"]:
            raise ValueError(f"{row['scene_id']} {row['context']} judgment does not match")
        if argmin_name({name: float(value) for name, value in row["class_nll"].items()}) != row["class_argmin"]:
            raise ValueError("class argmin does not match the stored losses")
        _argmin, _best, flag = scoring_disagreement(
            row["parsed_answer"],
            bool(row["invalid"]),
            {name: {"sum_nll": float(value)} for name, value in row["class_nll"].items()},
        )
        if flag != row["class_disagreement"]:
            raise ValueError("class disagreement flag does not match")


def _slice(predictions: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    rows = [row for row in predictions if row["context"] == context]
    n_correct = sum(bool(row["correct"]) for row in rows)
    n_valid_wrong = sum(bool(row["valid_wrong"]) for row in rows)
    n_invalid = sum(bool(row["invalid"]) for row in rows)
    n_class = sum(row["class_argmin"] == row["answer"] for row in rows)
    return {
        "n_rows": len(rows),
        "n_correct": n_correct,
        "n_valid_wrong": n_valid_wrong,
        "n_invalid": n_invalid,
        "accuracy": proportion(n_correct, len(rows)),
        "valid_wrong_rate": proportion(n_valid_wrong, len(rows)),
        "invalid_rate": proportion(n_invalid, len(rows)),
        "class_argmin_accuracy": proportion(n_class, len(rows)),
    }


def _pairs(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(dict)
    for row in predictions:
        grouped[row["counterfactual_group_id"]].setdefault(row["scene_id"], {})[row["context"]] = row
    out = {}
    for context in CONTEXTS:
        both = 0
        for scenes in grouped.values():
            arms = [scenes[scene_id][context] for scene_id in sorted(scenes)]
            if len(arms) == 2 and arms[0]["correct"] and arms[1]["correct"]:
                both += 1
        out[context] = proportion(both, len(grouped))
    return out


def _clean_both(predictions: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    scenes: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    group_of = {}
    for row in predictions:
        group_of[row["scene_id"]] = row["counterfactual_group_id"]
        if row["context"] in PRIMARY:
            scenes[row["scene_id"]][row["context"]] = row
    clean = []
    for scene_id in sorted(scenes):
        pair = scenes[scene_id]
        if all(bool(pair[context]["correct"]) for context in PRIMARY):
            clean.append(scene_id)
    groups = sorted({group_of[scene_id] for scene_id in clean})
    return clean, groups
