"""Read the stored Checkpoint 1 generations. Do not rewrite them.

The counts below re-parse ``raw_output`` with the unchanged strict parser and
place the embedded-label reading beside that parse. They are descriptions of
the committed file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from vlm_iface.parsing import parse_generation

from .reading import side_by_side

PREDICTIONS = Path("docs") / "checkpoint-1" / "predictions.jsonl"


def load_predictions(root: Path) -> list[dict[str, Any]]:
    path = root / PREDICTIONS
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 96:
        raise RuntimeError(f"Expected 96 stored predictions, found {len(rows)}.")
    return rows


def audit_predictions(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize invalid outputs and scoring/generation disagreements."""
    mismatches: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in rows:
        reading = side_by_side(str(row["raw_output"]), truncated=False, task_family=str(row["task_family"]))
        reparsed = parse_generation(str(row["raw_output"]), truncated=False, task_family=str(row["task_family"]))
        if (
            reparsed.invalid != row["invalid"]
            or reparsed.parsed_answer != row["parsed_answer"]
            or list(reparsed.reasons) != list(row["parse_reasons"])
            or reading["strict_parsed_answer"] != row["parsed_answer"]
        ):
            mismatches.append(f"{row['scene_id']} {row['context']}")
        if row["invalid"]:
            invalid_rows.append(_invalid_view(row, reading))
        if row["scoring_generation_disagreement"]:
            disagreements.append(_disagreement_view(row))
    return {
        "n_rows": len(rows),
        "n_reparse_mismatches": len(mismatches),
        "reparse_mismatches": mismatches,
        "n_invalid": len(invalid_rows),
        "invalid_reason_counts": dict(Counter(
            "|".join(row["parse_reasons"]) for row in rows if row["invalid"]
        )),
        "invalid_by_family": _invalid_by_family(invalid_rows),
        "invalid_embedded_equals_answer": sum(row["alternative_matches_answer"] is True for row in invalid_rows),
        "invalid_embedded_differs_from_answer": sum(row["alternative_matches_answer"] is False for row in invalid_rows),
        "invalid_embedded_equals_word_argmin": sum(
            row["alternative_parsed_answer"] == row["scored_argmin"] for row in invalid_rows
        ),
        "invalid_embedded_differs_from_word_argmin": sum(
            row["alternative_applicable"] and row["alternative_parsed_answer"] != row["scored_argmin"]
            for row in invalid_rows
        ),
        "invalid_not_embedded": sum(not row["alternative_applicable"] for row in invalid_rows),
        "n_scoring_generation_disagreement": len(disagreements),
        "disagreement_by_family": dict(Counter(row["task_family"] for row in disagreements)),
        "disagreement_digit_generations": sum(row["raw_is_digit"] for row in disagreements),
        "disagreement_case_color_generations": sum(row["raw_is_case_variant"] for row in disagreements),
        "alternative_selected": False,
        "invalid_rows": invalid_rows,
        "disagreement_rows": disagreements,
        "official_counts": _official_counts(rows),
    }


def _invalid_view(row: Mapping[str, Any], reading: Mapping[str, Any]) -> dict[str, Any]:
    alternative = reading["alternative_parsed_answer"]
    return {
        "scene_id": row["scene_id"],
        "context": row["context"],
        "task_family": row["task_family"],
        "answer": row["answer"],
        "raw_output": row["raw_output"],
        "parse_reasons": list(row["parse_reasons"]),
        "scored_argmin": row["scored_argmin"],
        "alternative_parsed_answer": alternative,
        "alternative_applicable": reading["alternative_applicable"],
        "alternative_matches_answer": alternative == row["answer"] if alternative is not None else None,
        "alternative_selected": False,
    }


def _disagreement_view(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(row["raw_output"]).strip()
    return {
        "scene_id": row["scene_id"],
        "context": row["context"],
        "task_family": row["task_family"],
        "answer": row["answer"],
        "raw_output": row["raw_output"],
        "parsed_answer": row["parsed_answer"],
        "correct": row["correct"],
        "scored_argmin": row["scored_argmin"],
        "sum_nll": row["sum_nll"],
        "scored_argmin_sum_nll": row["scored_argmin_sum_nll"],
        "gap": float(row["sum_nll"]) - float(row["scored_argmin_sum_nll"]),
        "raw_is_digit": raw in {"1", "2", "3", "4"},
        "raw_is_case_variant": raw in {"Red", "Blue", "Green", "Yellow", "RED", "BLUE", "GREEN", "YELLOW"},
    }


def _invalid_by_family(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    tallies: dict[str, dict[str, int]] = {}
    for family in ("attribute", "counting", "spatial"):
        group = [row for row in rows if row["task_family"] == family]
        tallies[family] = {
            "n": len(group),
            "embedded_equals_answer": sum(row["alternative_matches_answer"] is True for row in group),
            "embedded_differs_from_answer": sum(row["alternative_matches_answer"] is False for row in group),
            "embedded_equals_word_argmin": sum(
                row["alternative_parsed_answer"] == row["scored_argmin"] for row in group
            ),
        }
    return tallies


def _official_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "correct": sum(bool(row["correct"]) for row in rows),
        "valid_wrong": sum(bool(row["valid_wrong"]) for row in rows),
        "invalid": sum(bool(row["invalid"]) for row in rows),
        "disagreement": sum(bool(row["scoring_generation_disagreement"]) for row in rows),
    }
