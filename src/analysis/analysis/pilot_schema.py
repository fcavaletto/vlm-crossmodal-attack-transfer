"""Checkpoint 1 rows. The Checkpoint 0 toy schema is unchanged.

A pilot prediction stores the generation, the three-way score, and the sum
negative log-likelihood of the canonical answer. Invalid, valid-but-wrong, and
correct are mutually exclusive. An invalid row is a task error and is not a
correct answer.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .schemas import CONTEXTS, SchemaError, _finite_number, _require_bool, _require_id

LABEL = "DEVELOPMENT ONLY"
PREPROCESS = "float32_shared"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MODEL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
TASK_FAMILIES: tuple[str, ...] = ("attribute", "counting", "spatial")
PRIMARY_CONTEXTS: tuple[str, ...] = ("neutral", "helpful_nonrevealing")
MEAN_NLL_TOLERANCE = 1e-5

PILOT_PREDICTION_FIELDS = (
    "label",
    "scene_id",
    "counterfactual_group_id",
    "task_family",
    "context",
    "answer",
    "answer_aliases",
    "context_text",
    "question",
    "prompt",
    "n_objects",
    "ambiguity_flags",
    "raw_output",
    "parsed_answer",
    "parse_reasons",
    "correct",
    "valid_wrong",
    "invalid",
    "sum_nll",
    "token_count",
    "mean_nll",
    "library_mean_abs_diff",
    "scored_argmin",
    "scored_argmin_sum_nll",
    "scoring_generation_disagreement",
    "device",
    "model_id",
    "model_revision",
    "preprocess",
)

PILOT_GRADIENT_FIELDS = (
    "label",
    "scene_id",
    "counterfactual_group_id",
    "task_family",
    "context",
    "finite",
    "nonzero_fraction",
    "l1",
    "l2",
    "linf",
    "height",
    "width",
    "seconds",
    "repeat_max_abs_diff",
    "device",
    "model_id",
    "model_revision",
    "preprocess",
)

PILOT_ALIGNMENT_FIELDS = (
    "label",
    "scene_id",
    "counterfactual_group_id",
    "task_family",
    "source_context",
    "target_context",
    "alignment_a",
    "undefined_reason",
    "source_l1",
    "target_l1",
    "source_finite",
    "target_finite",
    "source_l1_zero",
    "source_gradient_excludes_h_star",
    "device",
    "model_id",
    "model_revision",
)


def validate_pilot_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked = [_copy_fields(row, PILOT_PREDICTION_FIELDS, "pilot prediction") for row in rows]
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(checked, start=1):
        where = f"pilot prediction {index}"
        _require_common(row, where)
        _require_context(row["context"], where)
        _require_family(row["task_family"], where)
        _require_text(row["answer"], f"{where} answer")
        _require_text(row["context_text"], f"{where} context_text")
        _require_text(row["question"], f"{where} question")
        _require_text(row["prompt"], f"{where} prompt")
        if row["prompt"] != row["context_text"] + "\n" + row["question"]:
            raise SchemaError(f"{where}: prompt must be the context, a newline, and the question")
        row["answer_aliases"] = _string_list(row["answer_aliases"], f"{where} answer_aliases")
        if row["answer"] not in row["answer_aliases"]:
            raise SchemaError(f"{where}: answer must be one of answer_aliases")
        row["ambiguity_flags"] = _string_list(row["ambiguity_flags"], f"{where} ambiguity_flags", allow_empty=True)
        row["n_objects"] = _positive_int(row["n_objects"], f"{where} n_objects")
        row["parse_reasons"] = _string_list(row["parse_reasons"], f"{where} parse_reasons", allow_empty=True)
        _require_bool(row["correct"], f"{where} correct")
        _require_bool(row["valid_wrong"], f"{where} valid_wrong")
        _require_bool(row["invalid"], f"{where} invalid")
        if int(row["correct"]) + int(row["valid_wrong"]) + int(row["invalid"]) != 1:
            raise SchemaError(f"{where}: correct, valid_wrong, and invalid must partition the row")
        if row["invalid"]:
            if row["parsed_answer"] is not None:
                raise SchemaError(f"{where}: an invalid row has no parsed answer")
            if not row["parse_reasons"]:
                raise SchemaError(f"{where}: an invalid row needs a parse reason")
        else:
            _require_text(row["parsed_answer"], f"{where} parsed_answer")
            matched = row["parsed_answer"] == row["answer"] or row["parsed_answer"] in row["answer_aliases"]
            if row["correct"] and not matched:
                raise SchemaError(f"{where}: a correct parse must match the answer or an alias")
            if row["valid_wrong"] and matched:
                raise SchemaError(f"{where}: a valid-wrong parse must not match the answer or an alias")
        row["sum_nll"] = _finite_number(row["sum_nll"], f"{where} sum_nll")
        row["token_count"] = _positive_int(row["token_count"], f"{where} token_count")
        row["mean_nll"] = _finite_number(row["mean_nll"], f"{where} mean_nll")
        if abs(row["mean_nll"] - row["sum_nll"] / row["token_count"]) > MEAN_NLL_TOLERANCE:
            raise SchemaError(f"{where}: mean_nll must be sum_nll / token_count")
        row["library_mean_abs_diff"] = _nonnegative_finite(row["library_mean_abs_diff"], f"{where} library_mean_abs_diff")
        _require_text(row["scored_argmin"], f"{where} scored_argmin")
        row["scored_argmin_sum_nll"] = _finite_number(row["scored_argmin_sum_nll"], f"{where} scored_argmin_sum_nll")
        _require_bool(row["scoring_generation_disagreement"], f"{where} scoring_generation_disagreement")
        if row["invalid"] and row["scoring_generation_disagreement"]:
            raise SchemaError(f"{where}: an invalid generation is not also a scoring disagreement")
        if not isinstance(row["raw_output"], str):
            raise SchemaError(f"{where}: raw_output must be a string")
        key = (row["scene_id"], row["context"])
        if key in seen:
            raise SchemaError(f"{where}: duplicate scene and context {key!r}")
        seen[key] = index
    return checked


def validate_pilot_gradients(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked = [_copy_fields(row, PILOT_GRADIENT_FIELDS, "pilot gradient") for row in rows]
    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(checked, start=1):
        where = f"pilot gradient {index}"
        _require_common(row, where)
        _require_family(row["task_family"], where)
        if row["context"] not in PRIMARY_CONTEXTS:
            raise SchemaError(f"{where}: gradient context must be one of {PRIMARY_CONTEXTS}")
        _require_bool(row["finite"], f"{where} finite")
        row["nonzero_fraction"] = _unit_interval(row["nonzero_fraction"], f"{where} nonzero_fraction")
        row["l1"] = _norm(row["l1"], row["finite"], f"{where} l1")
        row["l2"] = _norm(row["l2"], row["finite"], f"{where} l2")
        row["linf"] = _norm(row["linf"], row["finite"], f"{where} linf")
        row["height"] = _positive_int(row["height"], f"{where} height")
        row["width"] = _positive_int(row["width"], f"{where} width")
        row["seconds"] = _nonnegative_finite(row["seconds"], f"{where} seconds")
        if row["repeat_max_abs_diff"] is not None:
            row["repeat_max_abs_diff"] = _nonnegative_finite(
                row["repeat_max_abs_diff"], f"{where} repeat_max_abs_diff"
            )
        if row["finite"] and row["l1"] == 0 and row["nonzero_fraction"] != 0:
            raise SchemaError(f"{where}: a zero L1 gradient has a zero nonzero fraction")
        key = (row["scene_id"], row["context"])
        if key in seen:
            raise SchemaError(f"{where}: duplicate scene and context {key!r}")
        seen[key] = index
    return checked


def validate_pilot_alignments(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checked = [_copy_fields(row, PILOT_ALIGNMENT_FIELDS, "pilot alignment") for row in rows]
    seen: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(checked, start=1):
        where = f"pilot alignment {index}"
        _require_common(row, where)
        _require_family(row["task_family"], where)
        if row["source_context"] not in PRIMARY_CONTEXTS or row["target_context"] not in PRIMARY_CONTEXTS:
            raise SchemaError(f"{where}: alignment contexts must be {PRIMARY_CONTEXTS}")
        if row["source_context"] == row["target_context"]:
            raise SchemaError(f"{where}: the pilot stores the two cross directions, not a same-context pair")
        _require_bool(row["source_finite"], f"{where} source_finite")
        _require_bool(row["target_finite"], f"{where} target_finite")
        _require_bool(row["source_l1_zero"], f"{where} source_l1_zero")
        _require_bool(row["source_gradient_excludes_h_star"], f"{where} source_gradient_excludes_h_star")
        row["source_l1"] = _norm(row["source_l1"], row["source_finite"], f"{where} source_l1")
        row["target_l1"] = _norm(row["target_l1"], row["target_finite"], f"{where} target_l1")
        source_is_zero = row["source_finite"] and row["source_l1"] == 0
        if row["source_l1_zero"] != source_is_zero:
            raise SchemaError(f"{where}: source_l1_zero must agree with a finite zero source")
        excluded = row["source_l1_zero"] or (not row["source_finite"])
        if row["source_gradient_excludes_h_star"] != excluded:
            raise SchemaError(f"{where}: source exclusion must follow a zero or non-finite source gradient")
        if row["alignment_a"] is None:
            if not isinstance(row["undefined_reason"], str) or row["undefined_reason"] == "":
                raise SchemaError(f"{where}: an undefined alignment needs a reason")
        else:
            row["alignment_a"] = _finite_number(row["alignment_a"], f"{where} alignment_a")
            if abs(row["alignment_a"]) > 1 + 1e-8:
                raise SchemaError(f"{where}: alignment_a must lie in [-1, 1]")
            if row["undefined_reason"] is not None:
                raise SchemaError(f"{where}: a defined alignment has no undefined reason")
            if (not row["source_finite"]) or (not row["target_finite"]) or row["target_l1"] == 0:
                raise SchemaError(f"{where}: alignment_a must be null when a gradient is unusable")
        key = (row["scene_id"], row["source_context"], row["target_context"])
        if key in seen:
            raise SchemaError(f"{where}: duplicate alignment {key!r}")
        seen[key] = index
    return checked


def validate_pilot_bundle(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate the three files and check that they describe the same scenes."""
    prediction_rows = validate_pilot_predictions(predictions)
    gradient_rows = validate_pilot_gradients(gradients)
    alignment_rows = validate_pilot_alignments(alignments)
    scenes = {row["scene_id"] for row in prediction_rows}
    if not scenes:
        raise SchemaError("pilot bundle is empty")
    identity: dict[str, tuple[str, str]] = {}
    for row in prediction_rows:
        key = (row["counterfactual_group_id"], row["task_family"])
        if row["scene_id"] in identity and identity[row["scene_id"]] != key:
            raise SchemaError(f"{row['scene_id']}: group or family changed across contexts")
        identity[row["scene_id"]] = key
        contexts = {item["context"] for item in prediction_rows if item["scene_id"] == row["scene_id"]}
        if contexts != set(CONTEXTS):
            raise SchemaError(f"{row['scene_id']}: predictions must contain every context")
    for row in (*gradient_rows, *alignment_rows):
        if row["scene_id"] not in identity:
            raise SchemaError(f"{row['scene_id']}: gradient row has no prediction")
        if identity[row["scene_id"]] != (row["counterfactual_group_id"], row["task_family"]):
            raise SchemaError(f"{row['scene_id']}: gradient identity does not match the prediction")
    gradient_index = {(row["scene_id"], row["context"]): row for row in gradient_rows}
    for scene_id in scenes:
        for context in PRIMARY_CONTEXTS:
            if (scene_id, context) not in gradient_index:
                raise SchemaError(f"{scene_id}: missing {context} gradient")
    for row in alignment_rows:
        source = gradient_index[(row["scene_id"], row["source_context"])]
        target = gradient_index[(row["scene_id"], row["target_context"])]
        if not _same_norm(row["source_l1"], source["l1"]) or not _same_norm(row["target_l1"], target["l1"]):
            raise SchemaError(f"{row['scene_id']}: alignment norms do not match the gradient file")
        if row["source_finite"] != source["finite"] or row["target_finite"] != target["finite"]:
            raise SchemaError(f"{row['scene_id']}: alignment finiteness does not match the gradient file")
    return {
        "predictions": prediction_rows,
        "gradients": gradient_rows,
        "alignments": alignment_rows,
    }


def _copy_fields(row: Mapping[str, Any], fields: Sequence[str], where: str) -> dict[str, Any]:
    missing = [field for field in fields if field not in row]
    extra = sorted(set(row) - set(fields))
    if missing or extra:
        raise SchemaError(f"{where}: missing {missing or '[]'}, unknown {extra or '[]'}")
    return dict(row)


def _require_common(row: Mapping[str, Any], where: str) -> None:
    if row["label"] != LABEL:
        raise SchemaError(f"{where}: label must be {LABEL!r}")
    if row["device"] != "mps":
        raise SchemaError(f"{where}: device must be 'mps'")
    if row["model_id"] != MODEL_ID or row["model_revision"] != MODEL_REVISION:
        raise SchemaError(f"{where}: model pin does not match the approved checkpoint")
    if "preprocess" in row and row["preprocess"] != PREPROCESS:
        raise SchemaError(f"{where}: preprocess must be {PREPROCESS!r}")
    _require_id(row["scene_id"], f"{where} scene_id")
    _require_id(row["counterfactual_group_id"], f"{where} counterfactual_group_id")


def _require_context(value: Any, where: str) -> None:
    if value not in CONTEXTS:
        raise SchemaError(f"{where}: context must be one of {CONTEXTS}")


def _require_family(value: Any, where: str) -> None:
    if value not in TASK_FAMILIES:
        raise SchemaError(f"{where}: task_family must be one of {TASK_FAMILIES}")


def _require_text(value: Any, where: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise SchemaError(f"{where} must be a non-empty string")


def _string_list(value: Any, where: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise SchemaError(f"{where} must be a list of strings")
    if not value and not allow_empty:
        raise SchemaError(f"{where} must not be empty")
    if any(not isinstance(item, str) or item == "" for item in value):
        raise SchemaError(f"{where} must contain only non-empty strings")
    return list(value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{where} must be an integer >= 1")
    return value


def _norm(value: Any, finite: bool, where: str) -> float | None:
    if not finite:
        if value is not None:
            raise SchemaError(f"{where} must be null when the gradient is not finite")
        return None
    return _nonnegative_finite(value, where)


def _same_norm(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= 1e-6


def _nonnegative_finite(value: Any, where: str) -> float:
    number = _finite_number(value, where)
    if number < 0 or math.isnan(number):
        raise SchemaError(f"{where} must be >= 0")
    return number


def _unit_interval(value: Any, where: str) -> float:
    number = _nonnegative_finite(value, where)
    if number > 1:
        raise SchemaError(f"{where} must be <= 1")
    return number
