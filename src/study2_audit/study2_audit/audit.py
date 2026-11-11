"""Assemble the Checkpoint 1.7 audit.

The Qwen files, the row-color table, and the published three-family precision
file are read. Nothing is written back to ``docs/checkpoint-1.6``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cluster import descriptive_cluster_mean, descriptive_group_rate, scene_iid_mean
from .decision import decide_scene_count
from .eligibility import explain_nine
from .metrics import metric_catalog, partition_counts
from .precision import DEFAULT_BOOT, DEFAULT_GRID, DEFAULT_REPLICATIONS, DEFAULT_SEED, simulate_precision
from .qwen_record import QWEN_SUCCESS_RELATIVE, audit_qwen_records
from .reproducibility import check_reproducibility

PRIMARY = ("neutral", "helpful_nonrevealing")
CONTEXTS = ("neutral", "helpful_nonrevealing", "contradictory", "answer_revealing")
DESCRIPTIVE_SEED = 20261108
DESCRIPTIVE_BOOT = 4000


def build_audit(
    root: Path,
    *,
    n_replications: int = DEFAULT_REPLICATIONS,
    n_boot: int = DEFAULT_BOOT,
    n_groups_grid: Sequence[int] = DEFAULT_GRID,
    seed: int = DEFAULT_SEED,
    n_boot_descriptive: int = DESCRIPTIVE_BOOT,
) -> dict[str, Any]:
    """Run the audit. The precision grid is fabricated. The tables are read."""

    root = Path(root)
    before = (root / QWEN_SUCCESS_RELATIVE).read_bytes()
    qwen = audit_qwen_records(root)
    reproducibility = check_reproducibility(root)
    predictions = _read_jsonl(root / "docs/checkpoint-1.6/development_simple/predictions.jsonl")
    summary = json.loads((root / "docs/checkpoint-1.6/development_simple/summary.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "docs/checkpoint-1.6/development_simple/gate.json").read_text(encoding="utf-8"))
    eligibility = explain_nine(predictions, gate)
    intervals = descriptive_intervals(predictions, n_boot=n_boot_descriptive, seed=DESCRIPTIVE_SEED)
    precision = simulate_precision(
        n_groups_grid=n_groups_grid,
        n_replications=n_replications,
        n_boot=n_boot,
        seed=seed,
    )
    decision = decide_scene_count(eligibility, precision)
    after = (root / QWEN_SUCCESS_RELATIVE).read_bytes()
    return {
        "label": "DEVELOPMENT ONLY",
        "checkpoint": "checkpoint-1.7",
        "h_star_fitted": False,
        "model_inference": False,
        "attacks_run": False,
        "original_qwen_json_rewritten": before != after,
        "qwen_records_match_pins": qwen["bytes_unchanged"],
        "reproducibility_ok": reproducibility["ok"],
        "qwen": qwen,
        "reproducibility": reproducibility,
        "metrics": metric_catalog(),
        "partition": partition_counts(predictions),
        "ceilings": ceilings(summary),
        "eligibility": eligibility,
        "descriptive_intervals": intervals,
        "published_three_family_precision": published_three_family(root),
        "precision": precision,
        "decision": decision,
    }


def descriptive_intervals(
    predictions: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Cluster intervals for the clean table. They are not H* coefficients."""

    cluster_of = {row["scene_id"]: row["counterfactual_group_id"] for row in predictions}
    by_context = {}
    for context in CONTEXTS:
        values = {
            row["scene_id"]: 1.0 if row["correct"] else 0.0
            for row in predictions
            if row["context"] == context
        }
        by_context[context] = {
            "estimand": "strict accuracy, mean over scenes",
            "cluster": descriptive_cluster_mean(values, cluster_of, n_boot=n_boot, seed=seed),
            "scene_iid": scene_iid_mean(values, n_boot=n_boot, seed=seed),
            "both_arms_correct": descriptive_group_rate(
                _both_arms(predictions, context),
                n_boot=n_boot,
                seed=seed,
            ),
        }
    clean_scene, clean_group = _clean_indicators(predictions)
    return {
        "seed": seed,
        "n_boot": n_boot,
        "confirmatory_seed": False,
        "h_star_fitted": False,
        "sampling_unit": "counterfactual_group",
        "by_context": by_context,
        "scene_clean_both_primary": descriptive_cluster_mean(
            clean_scene,
            cluster_of,
            n_boot=n_boot,
            seed=seed,
        ),
        "group_has_a_clean_both_scene": descriptive_group_rate(clean_group, n_boot=n_boot, seed=seed),
    }


def ceilings(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Bounds that a later reading of this table has to keep in view."""

    cells = summary["clean_by_context"]
    notes = []
    for context, cell in cells.items():
        n = cell["n_rows"]
        error = n - cell["n_correct"]
        notes.append(
            {
                "context": context,
                "strict_correct": cell["n_correct"],
                "valid_wrong": cell["n_valid_wrong"],
                "invalid": cell["n_invalid"],
                "class_argmin_correct": cell["class_argmin_accuracy"]["numerator"],
                "clean_error_rate": error / n,
                "additional_error_floor_warning": error / n >= 0.5,
            }
        )
    revealing = cells["answer_revealing"]
    return {
        "by_context": notes,
        "attack_rates_observed": False,
        "additional_error_observed": False,
        "parser_ceiling": {
            "context": "answer_revealing",
            "strict_correct": revealing["n_correct"],
            "invalid": revealing["n_invalid"],
            "class_argmin_correct": revealing["class_argmin_accuracy"]["numerator"],
            "detail": (
                "Every revealing-context generation is invalid under the strict parser. "
                "The class-loss argmin matches the label on 22 of 24 scenes. "
                "The strict rate of zero is the parser rejecting sentences. "
                "It is not a finding that the image was ignored, and it is not a reason "
                "to replace the official score."
            ),
        },
        "near_flat_band": {"low": 0.05, "high": 0.95},
        "design_band": {"low": 0.60, "high": 0.80, "applies_to": "strict accuracy on the two primary contexts"},
    }


def published_three_family(root: Path) -> dict[str, Any]:
    """Cite the committed three-family precision file. Do not refit it."""

    path = root / "src/analysis/analysis/fixtures/hstar/precision_simulation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    recommendation = payload["recommendation"]
    return {
        "path": "src/analysis/analysis/fixtures/hstar/precision_simulation.json",
        "refit": False,
        "recommended_minimum_eligible_groups": recommendation["recommended_minimum_eligible_groups"],
        "does_not_promise_significance": recommendation["does_not_promise_significance"],
        "sentence_says_does_not_promise": "does not promise" in recommendation["sentence"],
        "cited_checkpoint1_eligible_groups": recommendation["cited_checkpoint1_eligible_groups"],
        "algebraic_minimum_groups": recommendation["algebraic_minimum_groups"],
        "known_beta_a": payload["known_beta_a"],
    }


def _both_arms(predictions: Sequence[Mapping[str, Any]], context: str) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for row in predictions:
        if row["context"] == context:
            grouped.setdefault(row["counterfactual_group_id"], []).append(bool(row["correct"]))
    return {group_id: 1.0 if values and all(values) else 0.0 for group_id, values in grouped.items()}


def _clean_indicators(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    scenes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        scenes.setdefault(row["scene_id"], {})[row["context"]] = row
    scene_values = {}
    group_hits: dict[str, bool] = {}
    for scene_id, contexts in scenes.items():
        clean = all(bool(contexts[name]["correct"]) for name in PRIMARY)
        scene_values[scene_id] = 1.0 if clean else 0.0
        group_id = contexts["neutral"]["counterfactual_group_id"]
        group_hits[group_id] = group_hits.get(group_id, False) or clean
    group_values = {group_id: 1.0 if hit else 0.0 for group_id, hit in group_hits.items()}
    return scene_values, group_values


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
