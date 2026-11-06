"""Estimands on validated rows.

Symbols used by the gradient alignment:

- ``g_s`` is the image-gradient of the candidate loss under the source context.
- ``g_t`` is the image-gradient under the evaluation (target) context.
- ``sign`` is elementwise: positive → 1, negative → -1, and 0 stays 0.
- ``||g_t||_1`` is the sum of absolute values of ``g_t``.
- ``A(s, t) = dot(g_t, sign(g_s)) / ||g_t||_1``.

A zero denominator is undefined. It is stored as null, not as zero.
Invalid generation is not an attack success. It still counts as a task error
when additional error is computed, because the answer is not correct.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import (
    CONTEXTS,
    MULTI_METHOD,
    MULTI_SOURCE,
    POSITIVE_CONTROL_CONTEXT,
    SchemaError,
    epsilon_key,
)

FLOOR_WARNING = (
    "Clean error is at least half the scenes in this cell. Additional error "
    "cannot be +1 on scenes that are already wrong, so a low additional error "
    "is not by itself evidence that an attack did nothing."
)

ANALYSIS_PRIMARY = "proposed_primary"
ANALYSIS_SECONDARY = "proposed_secondary"
ANALYSIS_EXPLORATORY = "exploratory"
ANALYSIS_POSITIVE_CONTROL = "positive_control"
ANALYSIS_DESCRIPTIVE = "descriptive_not_an_endpoint"


def proportion(numerator: int, denominator: int, *, empty_reason: str) -> dict[str, Any]:
    """A rate. ``value`` is null when ``denominator`` is 0."""
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        raise ValueError("counts must be integers")
    if numerator < 0 or denominator < 0:
        raise ValueError("counts must be non-negative")
    if numerator > denominator:
        raise ValueError("numerator cannot exceed denominator")
    if denominator == 0:
        return {
            "numerator": int(numerator),
            "denominator": 0,
            "value": None,
            "undefined_reason": empty_reason,
        }
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator,
        "undefined_reason": None,
    }


def clean_by_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Strict clean accuracy and invalid-output rate, one block per context.

    Accuracy is the number of scenes with ``correct`` true, divided by the
    number of scenes in that context. An invalid row must already be marked
    not correct, so an invalid output lowers accuracy and raises the invalid
    rate. A context with no rows has a null rate.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_prediction_flags(row)
        grouped[row["context"]].append(row)
    unknown = sorted(set(grouped) - set(CONTEXTS))
    if unknown:
        raise SchemaError(f"unknown context {unknown[0]!r}")
    return {context: _clean_block(context, grouped.get(context, [])) for context in CONTEXTS}


def additional_error_by_scene(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Scene → attacked task-error minus clean task-error.

    Task error is 1 when ``correct`` is false and 0 when it is true. Invalid
    output is not correct, so it contributes to this error. The result is in
    {-1, 0, +1} for a single row.
    """
    values: dict[str, float] = {}
    for row in rows:
        _require_attack_flags(row)
        scene_id = row["scene_id"]
        if scene_id in values:
            raise SchemaError(f"duplicate scene_id {scene_id!r} in one attack cell")
        error_before = 0.0 if row["correct_before"] else 1.0
        error_after = 0.0 if row["correct_after"] else 1.0
        values[scene_id] = error_after - error_before
    return values


def attack_success_indicators(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, int]]:
    """Scene → ``(denominator, numerator)`` for attack success.

    The denominator is 1 when the scene was correct before the attack.
    The numerator is 1 only when that scene is also valid and incorrect after.
    An invalid later output stays in the denominator and is not a success.
    """
    values: dict[str, tuple[int, int]] = {}
    for row in rows:
        _require_attack_flags(row)
        scene_id = row["scene_id"]
        if scene_id in values:
            raise SchemaError(f"duplicate scene_id {scene_id!r} in one attack cell")
        denominator = 1 if row["correct_before"] else 0
        success = (
            denominator == 1 and (not row["correct_after"]) and (not row["invalid_after"])
        )
        values[scene_id] = (denominator, 1 if success else 0)
    return values


def attack_success_rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indicators = attack_success_indicators(rows)
    denominator = sum(item[0] for item in indicators.values())
    numerator = sum(item[1] for item in indicators.values())
    invalid_among_baseline = 0
    for row in rows:
        if row["correct_before"] and row["invalid_after"]:
            invalid_among_baseline += 1
    rate = proportion(
        numerator,
        denominator,
        empty_reason="no scene was correct before the attack",
    )
    return {
        "n_scenes": len(indicators),
        "baseline_correct_n": denominator,
        "n_success": numerator,
        "n_invalid_after_among_baseline": invalid_among_baseline,
        "attack_success_rate": rate,
        "definition": (
            "Among scenes correct before the attack, the fraction with a valid "
            "and incorrect output after. Invalid outputs are not successes and "
            "remain in the denominator."
        ),
    }


def additional_error_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = additional_error_by_scene(rows)
    if not values:
        return {
            "n_scenes": 0,
            "mean": None,
            "clean_error_rate": None,
            "attacked_error_rate": None,
            "floor_warning": None,
            "undefined_reason": "no scenes",
        }
    clean_errors = []
    attacked_errors = []
    for row in rows:
        clean_errors.append(0.0 if row["correct_before"] else 1.0)
        attacked_errors.append(0.0 if row["correct_after"] else 1.0)
    clean_error_rate = float(np.mean(np.asarray(clean_errors, dtype=np.float64)))
    attacked_error_rate = float(np.mean(np.asarray(attacked_errors, dtype=np.float64)))
    mean = float(np.mean(np.asarray(list(values.values()), dtype=np.float64)))
    return {
        "n_scenes": len(values),
        "mean": mean,
        "clean_error_rate": clean_error_rate,
        "attacked_error_rate": attacked_error_rate,
        "floor_warning": FLOOR_WARNING if clean_error_rate >= 0.5 else None,
        "undefined_reason": None,
    }


def sign_alignment(g_source: Any, g_target: Any) -> dict[str, Any]:
    """Directional agreement ``A(s, t)``.

    Arrays must share a shape. They are then flattened in C order. An empty
    array is an error. A target gradient whose L1 norm is 0 makes ``A``
    undefined. A source gradient of all zeros yields ``A = 0`` when the target
    norm is positive, because every sign is 0.
    """
    source = _gradient_array(g_source, "g_source")
    target = _gradient_array(g_target, "g_target")
    if source.shape != target.shape:
        raise ValueError(
            f"gradient shapes must match, got {source.shape} and {target.shape}"
        )
    if source.size == 0:
        raise ValueError("gradient is empty")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("gradients must be finite")
    flat_source = source.reshape(-1)
    flat_target = target.reshape(-1)
    signs = _element_sign(flat_source)
    l1 = float(np.abs(flat_target).sum())
    if l1 == 0.0:
        return {
            "alignment_a": None,
            "dot": None,
            "l1_target": 0.0,
            "undefined_reason": "target gradient L1 norm is zero",
        }
    dot = float(np.dot(flat_target, signs))
    return {
        "alignment_a": dot / l1,
        "dot": dot,
        "l1_target": l1,
        "undefined_reason": None,
    }


def mean_alignment(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean of stored ``alignment_a`` values by source and target.

    Null alignments are excluded from the mean and counted. They are not
    replaced by zero. A pair with no defined value has a null mean.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if "alignment_a" not in row:
            raise SchemaError("alignment row is missing alignment_a")
        grouped[(row["source_context"], row["target_context"])].append(row)
    blocks = []
    for source, target in sorted(grouped):
        group = grouped[(source, target)]
        defined = []
        n_undefined = 0
        seen: set[str] = set()
        for row in group:
            scene_id = row["scene_id"]
            if scene_id in seen:
                raise SchemaError(f"duplicate alignment scene {scene_id!r} for {(source, target)}")
            seen.add(scene_id)
            value = row["alignment_a"]
            if value is None:
                n_undefined += 1
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaError("alignment_a must be a number or null")
            defined.append(float(value))
        if defined:
            mean = float(np.mean(np.asarray(defined, dtype=np.float64)))
            undefined_reason = None
        else:
            mean = None
            undefined_reason = "no defined alignments"
        blocks.append(
            {
                "source_context": source,
                "target_context": target,
                "positive_control_only": _pair_is_positive_control(source, target),
                "n_scenes": len(group),
                "n_defined": len(defined),
                "n_undefined": n_undefined,
                "mean": mean,
                "undefined_reason": undefined_reason,
                "defined_by_scene": {
                    row["scene_id"]: None if row["alignment_a"] is None else float(row["alignment_a"])
                    for row in sorted(group, key=lambda item: item["scene_id"])
                    if row["alignment_a"] is not None
                },
            }
        )
    return blocks


def attack_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One summary per method, budget, source, and target."""
    grouped: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_attack_flags(row)
        key = (
            row["method"],
            epsilon_key(row["epsilon"]),
            int(row["step_budget"]),
            row["source_context"],
            row["target_context"],
        )
        grouped[key].append(row)
    cells = []
    for key in sorted(grouped):
        method, key_eps, step_budget, source, target = key
        group = grouped[key]
        success = attack_success_rate(group)
        added = additional_error_summary(group)
        cells.append(
            {
                "cell_id": f"{method}|{key_eps}|{step_budget}|{source}|{target}",
                "method": method,
                "epsilon": float(group[0]["epsilon"]),
                "epsilon_key": key_eps,
                "step_budget": step_budget,
                "source_context": source,
                "target_context": target,
                "cross_context": source != target,
                "positive_control_only": _pair_is_positive_control(source, target),
                "analysis_role": _cell_role(method, source, target),
                "research_question": _cell_question(method, source, target),
                "n_scenes": len(group),
                "attack_success_rate": success,
                "additional_error": added,
            }
        )
    return cells


def cross_context_attack_transfer(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attack cells whose source context and evaluation context differ.

    The transfer summary for a cell is that cell's attack success rate and
    additional error. Same-context cells are the reference and are omitted
    here. ``source_context == "multi"`` is cross-context relative to a single
    target because the perturbation was not built for that target alone.
    """
    return [
        cell
        for cell in attack_cells(rows)
        if cell["cross_context"] and not cell["positive_control_only"]
    ]


def conditional_transfer(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    epsilon: float,
    step_budget: int,
    source_context: str,
    target_context: str,
) -> dict[str, Any]:
    """Exploratory transfer rate among source successes that were clean on the target.

    A same-context success is a valid incorrect output after the attack on
    scenes that were correct before it, with source and target both equal to
    ``source_context``. Those scenes enter the target cell only when the target
    context was also correct before the attack. Invalid outputs are not
    successes. Scenes already incorrect on the target are excluded and counted,
    not treated as transfer failures.
    """
    if source_context == target_context:
        raise ValueError("conditional transfer requires two different contexts")
    same = _select_cell(
        rows,
        method=method,
        epsilon=epsilon,
        step_budget=step_budget,
        source_context=source_context,
        target_context=source_context,
    )
    cross = _select_cell(
        rows,
        method=method,
        epsilon=epsilon,
        step_budget=step_budget,
        source_context=source_context,
        target_context=target_context,
    )
    same_by_scene = {row["scene_id"]: row for row in same}
    cross_by_scene = {row["scene_id"]: row for row in cross}
    source_successes = []
    for scene_id, row in same_by_scene.items():
        denominator, numerator = attack_success_indicators([row])[scene_id]
        if denominator == 1 and numerator == 1:
            source_successes.append(scene_id)
    excluded = []
    eligible = []
    successes = []
    missing_target = []
    for scene_id in sorted(source_successes):
        target_row = cross_by_scene.get(scene_id)
        if target_row is None:
            missing_target.append(scene_id)
            continue
        if not target_row["correct_before"]:
            excluded.append(scene_id)
            continue
        eligible.append(scene_id)
        _, numerator = attack_success_indicators([target_row])[scene_id]
        if numerator == 1:
            successes.append(scene_id)
    if len(source_successes) == 0:
        empty_reason = "no same-context success"
    else:
        empty_reason = "no source success was correct on the target before the attack"
    rate = proportion(len(successes), len(eligible), empty_reason=empty_reason)
    return {
        "analysis_role": ANALYSIS_EXPLORATORY,
        "method": method,
        "epsilon": float(epsilon),
        "epsilon_key": epsilon_key(epsilon),
        "step_budget": int(step_budget),
        "source_context": source_context,
        "target_context": target_context,
        "positive_control_only": _pair_is_positive_control(source_context, target_context),
        "n_source_success": len(source_successes),
        "n_missing_target_row": len(missing_target),
        "n_excluded_target_not_correct_before": len(excluded),
        "n_eligible": len(eligible),
        "n_target_success": len(successes),
        "rate": rate,
        "definition": (
            "Among same-context successes that were correct on the target "
            "before the attack, the fraction with a valid incorrect target "
            "output after. Invalid outputs are not successes. Scenes already "
            "incorrect on the target are excluded."
        ),
    }


def mean_loss_by_context(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Descriptive means. Loss is not a primary endpoint in this plan."""
    grouped: dict[str, list[float]] = defaultdict(list)
    seen: dict[tuple[str, str], None] = {}
    for row in rows:
        key = (row["scene_id"], row["context"])
        if key in seen:
            raise SchemaError(f"duplicate loss row {key!r}")
        seen[key] = None
        if isinstance(row["loss"], bool) or not isinstance(row["loss"], (int, float)):
            raise SchemaError("loss must be a finite number")
        grouped[row["context"]].append(float(row["loss"]))
    blocks = {}
    for context in CONTEXTS:
        values = grouped.get(context, [])
        if not values:
            blocks[context] = {
                "n_scenes": 0,
                "mean_loss": None,
                "undefined_reason": "no scenes for this context",
            }
        else:
            blocks[context] = {
                "n_scenes": len(values),
                "mean_loss": float(np.mean(np.asarray(values, dtype=np.float64))),
                "undefined_reason": None,
            }
    return blocks


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """Pearson correlation. Undefined when fewer than 3 pairs or variance is 0."""
    if len(xs) != len(ys):
        raise ValueError("correlation inputs must have the same length")
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 3:
        return {
            "n": int(x.size),
            "value": None,
            "undefined_reason": "fewer than 3 paired scenes",
        }
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("correlation inputs must be finite")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"n": int(x.size), "value": None, "undefined_reason": "zero variance"}
    value = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(value):
        return {"n": int(x.size), "value": None, "undefined_reason": "non-finite correlation"}
    return {"n": int(x.size), "value": value, "undefined_reason": None}


def _clean_block(context: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    n_correct = 0
    n_invalid = 0
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id in seen:
            raise SchemaError(f"duplicate scene_id {scene_id!r} in context {context}")
        seen.add(scene_id)
        n_correct += int(row["correct"])
        n_invalid += int(row["invalid"])
    n_scenes = len(rows)
    accuracy = proportion(
        n_correct,
        n_scenes,
        empty_reason="no scenes for this context",
    )
    invalid_rate = proportion(
        n_invalid,
        n_scenes,
        empty_reason="no scenes for this context",
    )
    positive = context == POSITIVE_CONTROL_CONTEXT
    clean_error_rate = None if accuracy["value"] is None else 1.0 - accuracy["value"]
    scene_correct = {row["scene_id"]: 1.0 if row["correct"] else 0.0 for row in rows}
    scene_invalid = {row["scene_id"]: 1.0 if row["invalid"] else 0.0 for row in rows}
    return {
        "context": context,
        "analysis_role": ANALYSIS_POSITIVE_CONTROL if positive else ANALYSIS_SECONDARY,
        "positive_control_only": positive,
        "scoring": "strict_invalid_counts_as_incorrect",
        "n_scenes": n_scenes,
        "n_correct": n_correct,
        "n_invalid": n_invalid,
        "accuracy": accuracy,
        "invalid_output_rate": invalid_rate,
        "clean_error_rate": clean_error_rate,
        "floor_warning": (
            FLOOR_WARNING if clean_error_rate is not None and clean_error_rate >= 0.5 else None
        ),
        "scene_correct": scene_correct,
        "scene_invalid": scene_invalid,
    }


def _cell_role(method: str, source: str, target: str) -> str:
    if _pair_is_positive_control(source, target):
        return ANALYSIS_POSITIVE_CONTROL
    if method == MULTI_METHOD or source == MULTI_SOURCE:
        return ANALYSIS_SECONDARY
    return ANALYSIS_SECONDARY


def _cell_question(method: str, source: str, target: str) -> str:
    if _pair_is_positive_control(source, target):
        return "positive_control"
    if method == MULTI_METHOD or source == MULTI_SOURCE:
        return "RQ3"
    if source != target:
        return "RQ2_RQ3"
    return "RQ1_RQ3"


def _pair_is_positive_control(source: str, target: str) -> bool:
    return source == POSITIVE_CONTROL_CONTEXT or target == POSITIVE_CONTROL_CONTEXT


def _select_cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    epsilon: float,
    step_budget: int,
    source_context: str,
    target_context: str,
) -> list[Mapping[str, Any]]:
    key_eps = epsilon_key(epsilon)
    selected = []
    for row in rows:
        if row["method"] != method:
            continue
        if epsilon_key(row["epsilon"]) != key_eps:
            continue
        if int(row["step_budget"]) != int(step_budget):
            continue
        if row["source_context"] != source_context or row["target_context"] != target_context:
            continue
        selected.append(row)
    return selected


def _require_prediction_flags(row: Mapping[str, Any]) -> None:
    if not isinstance(row.get("correct"), bool) or not isinstance(row.get("invalid"), bool):
        raise SchemaError("prediction correct and invalid must be booleans")
    if row["invalid"] and row["correct"]:
        raise SchemaError("an invalid output cannot be correct")
    if "scene_id" not in row or "context" not in row:
        raise SchemaError("prediction row is missing scene_id or context")


def _require_attack_flags(row: Mapping[str, Any]) -> None:
    for field in ("correct_before", "correct_after", "invalid_after"):
        if not isinstance(row.get(field), bool):
            raise SchemaError(f"attack {field} must be a boolean")
    if row["invalid_after"] and row["correct_after"]:
        raise SchemaError("an invalid attacked output cannot be correct")
    for field in ("scene_id", "source_context", "target_context", "method"):
        if field not in row:
            raise SchemaError(f"attack row is missing {field}")


def _gradient_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    return array


def _element_sign(values: np.ndarray) -> np.ndarray:
    """Sign with exact zeros, including negative zero, mapped to 0."""
    signs = np.zeros(values.shape, dtype=np.float64)
    signs[values > 0] = 1.0
    signs[values < 0] = -1.0
    return signs
