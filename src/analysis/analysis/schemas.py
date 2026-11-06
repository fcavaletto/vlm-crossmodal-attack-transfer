"""JSONL contracts for later study tables.

The field names are the Checkpoint 0 interface. This module checks them.
It does not score a model, and it does not read images.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CONTEXTS: tuple[str, ...] = (
    "neutral",
    "helpful_nonrevealing",
    "contradictory",
    "answer_revealing",
)
POSITIVE_CONTROL_CONTEXT = "answer_revealing"
MULTI_SOURCE = "multi"
MULTI_METHOD = "pgd_multi"
METHODS: frozenset[str] = frozenset({"pgd", "fgsm", "noise", "pgd_multi"})

# Bookkeeping identity for an attack budget. Two floats that agree at 8
# decimal places share a cell. This is not a claim about perceptual distance.
EPSILON_DECIMALS = 8

PREDICTION_FIELDS = (
    "scene_id",
    "context",
    "parsed_answer",
    "invalid",
    "correct",
)
LOSS_FIELDS = ("scene_id", "context", "loss")
GRADIENT_SUMMARY_FIELDS = (
    "scene_id",
    "source_context",
    "target_context",
    "alignment_a",
)
GRADIENT_VECTOR_FIELDS = (
    "scene_id",
    "source_context",
    "target_context",
    "g_source",
    "g_target",
)
ATTACK_FIELDS = (
    "scene_id",
    "source_context",
    "target_context",
    "method",
    "epsilon",
    "step_budget",
    "correct_before",
    "correct_after",
    "invalid_after",
)


class SchemaError(ValueError):
    """A row broke the Checkpoint 0 field contract."""


def epsilon_key(epsilon: float) -> str:
    """Stable cell key for a non-negative finite budget label."""
    number = _finite_number(epsilon, "epsilon")
    if number < 0:
        raise SchemaError("epsilon must be >= 0")
    return f"{number:.{EPSILON_DECIMALS}f}"


def load_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load objects from a UTF-8 JSONL file. Blank lines are ignored."""
    text = Path(path).read_text(encoding="utf-8")
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(obj, dict):
            raise SchemaError(f"{path}:{line_number}: each line must be a JSON object")
        rows.append((line_number, obj))
    return rows


def validate_predictions(rows: Sequence[tuple[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    checked = [_check_fields(row, PREDICTION_FIELDS, f"predictions line {line}") for line, row in rows]
    for (line, _), row in zip(rows, checked):
        _require_id(row["scene_id"], f"predictions line {line} scene_id")
        _require_context(row["context"], f"predictions line {line} context")
        if not isinstance(row["parsed_answer"], str):
            raise SchemaError(f"predictions line {line}: parsed_answer must be a string")
        _require_bool(row["invalid"], f"predictions line {line} invalid")
        _require_bool(row["correct"], f"predictions line {line} correct")
        if row["invalid"] and row["correct"]:
            raise SchemaError(
                f"predictions line {line}: an invalid output cannot be correct"
            )
    _reject_duplicates(
        checked,
        rows,
        lambda row: (row["scene_id"], row["context"]),
        "predictions",
    )
    return checked


def validate_losses(rows: Sequence[tuple[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    checked = [_check_fields(row, LOSS_FIELDS, f"losses line {line}") for line, row in rows]
    for (line, _), row in zip(rows, checked):
        _require_id(row["scene_id"], f"losses line {line} scene_id")
        _require_context(row["context"], f"losses line {line} context")
        row["loss"] = _finite_number(row["loss"], f"losses line {line} loss")
    _reject_duplicates(
        checked,
        rows,
        lambda row: (row["scene_id"], row["context"]),
        "losses",
    )
    return checked


def validate_gradient_summaries(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    checked = [
        _check_fields(row, GRADIENT_SUMMARY_FIELDS, f"gradient summaries line {line}")
        for line, row in rows
    ]
    for (line, _), row in zip(rows, checked):
        _require_id(row["scene_id"], f"gradient summaries line {line} scene_id")
        _require_alignment_context(
            row["source_context"], f"gradient summaries line {line} source_context"
        )
        _require_context(row["target_context"], f"gradient summaries line {line} target_context")
        row["alignment_a"] = _optional_alignment(
            row["alignment_a"], f"gradient summaries line {line} alignment_a"
        )
    _reject_duplicates(
        checked,
        rows,
        lambda row: (row["scene_id"], row["source_context"], row["target_context"]),
        "gradient summaries",
    )
    return checked


def validate_gradient_vectors(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Audit file. Not part of the study schema. Tiny toy vectors only in tests."""
    checked = [
        _check_fields(row, GRADIENT_VECTOR_FIELDS, f"gradient vectors line {line}")
        for line, row in rows
    ]
    for (line, _), row in zip(rows, checked):
        _require_id(row["scene_id"], f"gradient vectors line {line} scene_id")
        _require_alignment_context(
            row["source_context"], f"gradient vectors line {line} source_context"
        )
        _require_context(row["target_context"], f"gradient vectors line {line} target_context")
        row["g_source"] = _vector(row["g_source"], f"gradient vectors line {line} g_source")
        row["g_target"] = _vector(row["g_target"], f"gradient vectors line {line} g_target")
    _reject_duplicates(
        checked,
        rows,
        lambda row: (row["scene_id"], row["source_context"], row["target_context"]),
        "gradient vectors",
    )
    return checked


def validate_attacks(rows: Sequence[tuple[int, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    checked = [_check_fields(row, ATTACK_FIELDS, f"attacks line {line}") for line, row in rows]
    for (line, _), row in zip(rows, checked):
        where = f"attacks line {line}"
        _require_id(row["scene_id"], f"{where} scene_id")
        _require_method(row["method"], f"{where} method")
        _require_source(row["source_context"], f"{where} source_context")
        _require_context(row["target_context"], f"{where} target_context")
        _require_multi_consistency(row, where)
        row["epsilon"] = _finite_number(row["epsilon"], f"{where} epsilon")
        if row["epsilon"] < 0:
            raise SchemaError(f"{where}: epsilon must be >= 0")
        row["step_budget"] = _nonnegative_int(row["step_budget"], f"{where} step_budget")
        _require_bool(row["correct_before"], f"{where} correct_before")
        _require_bool(row["correct_after"], f"{where} correct_after")
        _require_bool(row["invalid_after"], f"{where} invalid_after")
        if row["invalid_after"] and row["correct_after"]:
            raise SchemaError(f"{where}: an invalid attacked output cannot be correct")
    _reject_duplicates(
        checked,
        rows,
        lambda row: (
            row["scene_id"],
            row["source_context"],
            row["target_context"],
            row["method"],
            epsilon_key(row["epsilon"]),
            row["step_budget"],
        ),
        "attacks",
    )
    return checked


def _check_fields(
    row: Mapping[str, Any], fields: Iterable[str], where: str
) -> dict[str, Any]:
    allowed = tuple(fields)
    missing = [field for field in allowed if field not in row]
    extra = sorted(set(row) - set(allowed))
    if missing or extra:
        raise SchemaError(f"{where}: missing {missing or '[]'}, unknown {extra or '[]'}")
    return dict(row)


def _reject_duplicates(
    checked: Sequence[Mapping[str, Any]],
    located: Sequence[tuple[int, Mapping[str, Any]]],
    key_fn,
    label: str,
) -> None:
    seen: dict[tuple, int] = {}
    for row, (line, _) in zip(checked, located):
        key = key_fn(row)
        if key in seen:
            raise SchemaError(f"{label}: duplicate key {key!r} on lines {seen[key]} and {line}")
        seen[key] = line


def _require_id(value: Any, where: str) -> None:
    if not isinstance(value, str) or value == "" or value != value.strip() or any(
        char.isspace() for char in value
    ):
        raise SchemaError(f"{where} must be a non-empty string without whitespace")


def _require_context(value: Any, where: str) -> None:
    if value not in CONTEXTS:
        raise SchemaError(f"{where} must be one of {CONTEXTS}")


def _require_alignment_context(value: Any, where: str) -> None:
    if value not in CONTEXTS and value != MULTI_SOURCE:
        raise SchemaError(f"{where} must be one of {CONTEXTS} or {MULTI_SOURCE!r}")


def _require_source(value: Any, where: str) -> None:
    _require_alignment_context(value, where)


def _require_method(value: Any, where: str) -> None:
    if value not in METHODS:
        raise SchemaError(f"{where} must be one of {sorted(METHODS)}")


def _require_multi_consistency(row: Mapping[str, Any], where: str) -> None:
    is_multi_source = row["source_context"] == MULTI_SOURCE
    is_multi_method = row["method"] == MULTI_METHOD
    if is_multi_source != is_multi_method:
        raise SchemaError(
            f"{where}: method {MULTI_METHOD!r} and source_context {MULTI_SOURCE!r} "
            "must be used together"
        )


def _require_bool(value: Any, where: str) -> None:
    if not isinstance(value, bool):
        raise SchemaError(f"{where} must be a boolean")


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{where} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"{where} must be finite")
    return number


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{where} must be an integer >= 0")
    return value


def _optional_alignment(value: Any, where: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, where)
    if abs(number) > 1 + 1e-8:
        raise SchemaError(f"{where} must lie in [-1, 1] or be null")
    return number


def _vector(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) == 0:
        raise SchemaError(f"{where} must be a non-empty list of finite numbers")
    return [_finite_number(item, where) for item in value]
