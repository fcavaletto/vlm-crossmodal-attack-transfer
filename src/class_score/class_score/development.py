"""Counts from the committed v2 development predictions.

The file is read as stored. No row is deleted. Accuracy denominators are every
row in the cell. These counts describe that SmolVLM2 MPS run. They do not
select the scoring rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from .decision import argmin_class, is_disagreement
from .generation import score_free_generation
from .protocol import (
    CLASS_ARGMIN_IS_PRIMARY,
    CLASS_LOSS_ROLE,
    CONTEXTS,
    DESIGN_BAND,
    EMBEDDED_READING_ADOPTED,
    FAMILIES,
    FREE_GENERATION_ROLE,
    N_DEVELOPMENT_ROWS,
    OFFICIAL_METRIC,
    PRIMARY_CONTEXTS,
    SCENES_PER_FAMILY,
)
from .sensitivity import embedded_read_counterfactual
from .vocabulary import task_vocabulary

PREDICTIONS = Path("docs/checkpoint-1.5/development_v2/predictions.jsonl")
DIGIT_GENERATIONS = frozenset({"1", "2", "3", "4"})


def predictions_path(root: Path | None = None) -> Path:
    base = root if root is not None else Path(__file__).resolve().parents[3]
    return base / PREDICTIONS


def load_predictions(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != N_DEVELOPMENT_ROWS:
        raise ValueError(f"expected {N_DEVELOPMENT_ROWS} rows, found {len(rows)}")
    return rows


def in_design_band(correct: int, n: int) -> bool:
    if n <= 0:
        raise ValueError("a rate needs a positive denominator")
    low, high = DESIGN_BAND
    rate = correct / n
    return low <= rate <= high


def audit_predictions(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """Check the stored file against the strict parser and the argmin rule, then count."""
    if len(rows) != N_DEVELOPMENT_ROWS:
        raise ValueError(f"expected {N_DEVELOPMENT_ROWS} rows, found {len(rows)}")
    _check_rows(rows)
    by_family = {family: {context: _cell(rows, family, context) for context in CONTEXTS} for family in FAMILIES}
    by_family["all"] = {context: _cell(rows, None, context) for context in CONTEXTS}
    for family in (*FAMILIES, "all"):
        for context in CONTEXTS:
            cell = by_family[family][context]
            expected_n = N_DEVELOPMENT_ROWS // 4 if family == "all" else SCENES_PER_FAMILY
            if cell["n"] != expected_n:
                raise ValueError(f"{family} {context} has {cell['n']} rows, expected {expected_n}")
    agreement = _agreement(rows)
    band = _band(by_family)
    return {
        "label": "DEVELOPMENT ONLY",
        "n_rows": len(rows),
        "rows_dropped": 0,
        "official_metric": OFFICIAL_METRIC,
        "free_generation_role": FREE_GENERATION_ROLE,
        "class_loss_role": CLASS_LOSS_ROLE,
        "class_argmin_is_primary_outcome": CLASS_ARGMIN_IS_PRIMARY,
        "embedded_reading_adopted": EMBEDDED_READING_ADOPTED,
        "design_band": list(DESIGN_BAND),
        "by_family": by_family,
        "agreement": agreement,
        "band": band,
        "class_argmin_meets_band_for_every_family": band["class_argmin_every_family_in_band"],
        "both_primary": _both_primary(rows),
        "digit_generations": _digit_generations(rows),
        "decision_gap": _decision_gap(rows),
        "embedded_counterfactual": embedded_read_counterfactual(rows),
        "class_disagreement_rows": _named(rows, "class_disagreement"),
    }


def audit_development(root: Path | None = None) -> dict[str, object]:
    return audit_predictions(load_predictions(predictions_path(root)))


def _check_rows(rows: list[Mapping[str, object]]) -> None:
    scenes: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["label"] != "DEVELOPMENT ONLY":
            raise ValueError("a row is missing the development label")
        if row["model_id"] != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
            raise ValueError("a row is not the pinned SmolVLM2 checkpoint")
        if row["model_revision"] != "7b375e1b73b11138ff12fe22c8f2822d8fe03467":
            raise ValueError("a row is not the pinned SmolVLM2 revision")
        if row["device"] != "mps":
            raise ValueError("a row is not an MPS measurement")
        if row["format_line_appended"] is not False:
            raise ValueError("the v2 file was scored without the format line")
        if row["task_family"] not in FAMILIES or row["context"] not in CONTEXTS:
            raise ValueError("unexpected task family or context")
        scenes[str(row["scene_id"])] += 1
        free = score_free_generation(
            str(row["raw_output"]),
            truncated=bool(row["truncated"]),
            task_family=str(row["task_family"]),
            answer=str(row["answer"]),
        )
        if free["parsed_answer"] != row["parsed_answer"] or free["invalid"] != row["invalid"]:
            raise ValueError(f"{row['scene_id']} {row['context']} strict parse does not match")
        if list(free["reasons"]) != list(row["parse_reasons"]):
            raise ValueError(f"{row['scene_id']} {row['context']} parse reasons do not match")
        if (
            free["correct"] != row["correct"]
            or free["valid_wrong"] != row["valid_wrong"]
            or free["invalid"] != row["invalid"]
        ):
            raise ValueError(f"{row['scene_id']} {row['context']} partition does not match")
        if int(bool(row["correct"])) + int(bool(row["valid_wrong"])) + int(bool(row["invalid"])) != 1:
            raise ValueError(f"{row['scene_id']} {row['context']} is not one outcome")
        vocabulary = set(task_vocabulary(str(row["task_family"])))
        for key, flag in (("class_nll", "class_argmin"), ("word_only_nll", "word_only_argmin")):
            scores = {name: float(value) for name, value in row[key].items()}
            if set(scores) != vocabulary:
                raise ValueError(f"{row['scene_id']} {row['context']} {key} is not the task vocabulary")
            if argmin_class(scores) != row[flag]:
                raise ValueError(f"{row['scene_id']} {row['context']} {flag} does not match the losses")
            if any(value != value or abs(value) == float("inf") for value in scores.values()):
                raise ValueError(f"{row['scene_id']} {row['context']} has a non-finite loss")
        if is_disagreement(row["parsed_answer"], bool(row["invalid"]), _scores(row, "class_nll")) != bool(
            row["class_disagreement"]
        ):
            raise ValueError(f"{row['scene_id']} {row['context']} class disagreement flag does not match")
        if is_disagreement(row["parsed_answer"], bool(row["invalid"]), _scores(row, "word_only_nll")) != bool(
            row["word_only_disagreement"]
        ):
            raise ValueError(f"{row['scene_id']} {row['context']} word disagreement flag does not match")
    if len(scenes) != N_DEVELOPMENT_ROWS // len(CONTEXTS):
        raise ValueError("scene count is not 72")
    if any(count != len(CONTEXTS) for count in scenes.values()):
        raise ValueError("a scene does not have four contexts")


def _scores(row: Mapping[str, object], key: str) -> dict[str, float]:
    return {name: float(value) for name, value in row[key].items()}


def _cell(rows: list[Mapping[str, object]], family: str | None, context: str) -> dict[str, int]:
    subset = [
        row
        for row in rows
        if row["context"] == context and (family is None or row["task_family"] == family)
    ]
    strict = sum(bool(row["correct"]) for row in subset)
    valid_wrong = sum(bool(row["valid_wrong"]) for row in subset)
    invalid = sum(bool(row["invalid"]) for row in subset)
    if strict + valid_wrong + invalid != len(subset):
        raise ValueError(f"{family} {context} partition does not add up")
    return {
        "n": len(subset),
        "strict_correct": strict,
        "valid_wrong": valid_wrong,
        "invalid": invalid,
        "class_argmin_correct": sum(row["class_argmin"] == row["answer"] for row in subset),
        "word_argmin_correct": sum(row["word_only_argmin"] == row["answer"] for row in subset),
    }


def _agreement(rows: list[Mapping[str, object]]) -> dict[str, int]:
    valid = [row for row in rows if not row["invalid"]]
    return {
        "n_valid": len(valid),
        "n_invalid": sum(bool(row["invalid"]) for row in rows),
        "class_argmin_matches_parsed": sum(row["class_argmin"] == row["parsed_answer"] for row in valid),
        "word_argmin_matches_parsed": sum(row["word_only_argmin"] == row["parsed_answer"] for row in valid),
        "class_disagreement_flags": sum(bool(row["class_disagreement"]) for row in rows),
        "word_disagreement_flags": sum(bool(row["word_only_disagreement"]) for row in rows),
    }


def _band(by_family: Mapping[str, Mapping[str, Mapping[str, int]]]) -> dict[str, object]:
    cells = {}
    class_ok = True
    strict_ok = True
    for family in FAMILIES:
        cells[family] = {}
        for context in PRIMARY_CONTEXTS:
            cell = by_family[family][context]
            entry = {
                metric: {
                    "correct": cell[column],
                    "n": cell["n"],
                    "in_design_band": in_design_band(cell[column], cell["n"]),
                }
                for metric, column in (
                    ("strict", "strict_correct"),
                    ("class_argmin", "class_argmin_correct"),
                    ("word_argmin", "word_argmin_correct"),
                )
            }
            cells[family][context] = entry
            class_ok = class_ok and entry["class_argmin"]["in_design_band"]
            strict_ok = strict_ok and entry["strict"]["in_design_band"]
    return {
        "primary_contexts": list(PRIMARY_CONTEXTS),
        "by_family": cells,
        "strict_every_family_in_band": strict_ok,
        "class_argmin_every_family_in_band": class_ok,
    }


def _both_primary(rows: list[Mapping[str, object]]) -> dict[str, dict[str, dict[str, int]]]:
    scenes: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if row["context"] in PRIMARY_CONTEXTS:
            scenes[str(row["scene_id"])][str(row["context"])] = row
    for contexts in scenes.values():
        neutral = contexts["neutral"]
        helpful = contexts["helpful_nonrevealing"]
        if neutral["task_family"] != helpful["task_family"]:
            raise ValueError(f"{neutral['scene_id']} changes task family across contexts")
        if set(contexts) != set(PRIMARY_CONTEXTS):
            raise ValueError(f"{neutral['scene_id']} is missing a primary context")
    result: dict[str, dict[str, dict[str, int]]] = {}
    for metric, predicate in (
        ("strict", lambda row: bool(row["correct"])),
        ("class_argmin", lambda row: row["class_argmin"] == row["answer"]),
        ("word_argmin", lambda row: row["word_only_argmin"] == row["answer"]),
    ):
        result[metric] = {}
        for family in (*FAMILIES, "all"):
            n_scenes = 0
            groups: set[str] = set()
            for contexts in scenes.values():
                neutral = contexts["neutral"]
                helpful = contexts["helpful_nonrevealing"]
                if family != "all" and neutral["task_family"] != family:
                    continue
                if predicate(neutral) and predicate(helpful):
                    n_scenes += 1
                    groups.add(str(neutral["counterfactual_group_id"]))
            result[metric][family] = {"scenes": n_scenes, "groups": len(groups)}
    return result


def _digit_generations(rows: list[Mapping[str, object]]) -> dict[str, int]:
    digits = [
        row
        for row in rows
        if row["task_family"] == "counting" and str(row["raw_output"]).strip() in DIGIT_GENERATIONS
    ]
    return {
        "n": len(digits),
        "strict_correct": sum(bool(row["correct"]) for row in digits),
        "class_argmin_correct": sum(row["class_argmin"] == row["answer"] for row in digits),
        "word_argmin_correct": sum(row["word_only_argmin"] == row["answer"] for row in digits),
        "class_argmin_equals_parsed": sum(row["class_argmin"] == row["parsed_answer"] for row in digits),
        "word_argmin_equals_parsed": sum(row["word_only_argmin"] == row["parsed_answer"] for row in digits),
    }


def _decision_gap(rows: list[Mapping[str, object]]) -> dict[str, int]:
    counting = [row for row in rows if row["task_family"] == "counting"]
    color = [row for row in rows if row["task_family"] != "counting"]
    return {
        "class_argmin_differs_from_word_counting": sum(
            row["class_argmin"] != row["word_only_argmin"] for row in counting
        ),
        "class_argmin_differs_from_word_color": sum(
            row["class_argmin"] != row["word_only_argmin"] for row in color
        ),
    }


def _named(rows: list[Mapping[str, object]], flag: str) -> list[dict[str, object]]:
    named = []
    for row in rows:
        if not row[flag]:
            continue
        scores = _scores(row, "class_nll")
        parsed = str(row["parsed_answer"])
        chosen = str(row["class_argmin"])
        named.append(
            {
                "scene_id": row["scene_id"],
                "context": row["context"],
                "task_family": row["task_family"],
                "parsed_answer": parsed,
                "class_argmin": chosen,
                "answer": row["answer"],
                "gap": scores[parsed] - scores[chosen],
            }
        )
    return named


def main() -> None:
    summary = audit_development()
    print(json.dumps(summary, indent=2, sort_keys=True))
