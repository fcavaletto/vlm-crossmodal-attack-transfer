"""Source-success and conditional-transfer counts. No model and no regression."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .protocol import (
    EVAL_CONTEXTS,
    NEAR_EDGE,
    SOURCE_CONTEXT,
    TARGET_CONTEXT,
    configurations,
)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _view(block: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        return {"decision": None, "margin": None, "nll": None}
    decision = block.get("decision")
    return {
        "decision": decision if isinstance(decision, str) and decision else None,
        "margin": _finite(block.get("margin")),
        "nll": _finite(block.get("nll")),
    }


def scene_flags(label: str, clean: Mapping[str, Any], perturbed: Mapping[str, Any]) -> dict[str, Any]:
    """Flags for one scene. A missing decision stays missing."""
    clean_source = _view(clean.get(SOURCE_CONTEXT) if isinstance(clean, Mapping) else None)
    clean_target = _view(clean.get(TARGET_CONTEXT) if isinstance(clean, Mapping) else None)
    pert_source = _view(perturbed.get(SOURCE_CONTEXT) if isinstance(perturbed, Mapping) else None)
    pert_target = _view(perturbed.get(TARGET_CONTEXT) if isinstance(perturbed, Mapping) else None)
    source_clean = None if clean_source["decision"] is None else clean_source["decision"] == label
    target_clean = None if clean_target["decision"] is None else clean_target["decision"] == label

    if source_clean is not True or pert_source["decision"] is None:
        source_flip = None
    else:
        source_flip = pert_source["decision"] != label
    if target_clean is not True or pert_target["decision"] is None:
        target_flip = None
    else:
        target_flip = pert_target["decision"] != label

    if source_flip is not True:
        if source_clean is False:
            status = "not_source_clean"
        elif source_flip is False:
            status = "not_source_success"
        else:
            status = "missing"
        transfer = None
    elif target_clean is False:
        status = "excluded_target_not_clean"
        transfer = None
    elif target_flip is None:
        status = "missing"
        transfer = None
    elif target_flip is True:
        status = "transfer"
        transfer = True
    else:
        status = "no_transfer"
        transfer = False
    return {
        "source_clean_correct": source_clean,
        "target_clean_correct": target_clean,
        "source_flip": source_flip,
        "target_flip": target_flip,
        "transfer": transfer,
        "transfer_status": status,
    }


def edge_position(numerator: int | None, denominator: int | None) -> str:
    """Near the floor is 0 or 1. Near the ceiling is the denominator or one less."""
    if not isinstance(numerator, int) or isinstance(numerator, bool):
        return "undefined"
    if not isinstance(denominator, int) or isinstance(denominator, bool):
        return "undefined"
    if denominator < 2 or numerator < 0 or numerator > denominator:
        return "undefined"
    if numerator <= NEAR_EDGE:
        return "near_floor"
    if numerator >= denominator - NEAR_EDGE:
        return "near_ceiling"
    return "interior"


def classify_configuration(block: Mapping[str, Any]) -> dict[str, Any]:
    """Judgment from the two counts. Alignment is not an argument."""
    source_position = edge_position(
        block.get("n_source_success"),
        block.get("n_source_success_denominator"),
    )
    transfer_position = edge_position(
        block.get("n_conditional_transfer"),
        block.get("n_conditional_transfer_denominator"),
    )
    adequate = source_position in {"interior", "near_ceiling"}
    variation = transfer_position == "interior"
    return {
        "source_position": source_position,
        "transfer_position": transfer_position,
        "adequate_source_success": adequate,
        "transfer_variation": variation,
        "adequate_source_and_transfer_variation": adequate and variation,
        "alignment_used": False,
        "epsilon_raised": False,
        "steps_added": False,
    }


def configuration_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One block per recorded method, budget, and step. Order follows the protocol grid."""
    blocks = []
    for config in configurations():
        selected = [
            row
            for row in rows
            if row.get("method") == config["method"]
            and row.get("epsilon") == config["epsilon"]
            and row.get("steps") == config["steps"]
        ]
        blocks.append(_count_block(config, selected))
    return blocks


def _count_block(config: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_source_clean = 0
    n_source_success = 0
    n_source_failure = 0
    n_source_missing = 0
    n_source_not_clean = 0
    n_source_clean_missing = 0
    n_transfer = 0
    n_no_transfer = 0
    n_excluded = 0
    n_transfer_missing = 0
    n_target_clean = 0
    n_target_flip = 0
    n_target_failure = 0
    n_target_missing = 0
    increases: list[float] = []
    for row in rows:
        flags = scene_flags(str(row["answer"]), row.get("clean") or {}, row.get("perturbed") or {})
        if flags["source_clean_correct"] is True:
            n_source_clean += 1
            if flags["source_flip"] is True:
                n_source_success += 1
            elif flags["source_flip"] is False:
                n_source_failure += 1
            else:
                n_source_missing += 1
        elif flags["source_clean_correct"] is False:
            n_source_not_clean += 1
        else:
            n_source_clean_missing += 1
        status = flags["transfer_status"]
        if status == "transfer":
            n_transfer += 1
        elif status == "no_transfer":
            n_no_transfer += 1
        elif status == "excluded_target_not_clean":
            n_excluded += 1
        elif flags["source_flip"] is True:
            n_transfer_missing += 1
        if flags["target_clean_correct"] is True:
            n_target_clean += 1
            if flags["target_flip"] is True:
                n_target_flip += 1
            elif flags["target_flip"] is False:
                n_target_failure += 1
            else:
                n_target_missing += 1
        increase = _increase(row)
        if flags["source_clean_correct"] is True and increase is not None:
            increases.append(increase)
    source_denominator = n_source_success + n_source_failure
    transfer_denominator = n_transfer + n_no_transfer
    target_denominator = n_target_flip + n_target_failure
    block: dict[str, Any] = {
        "method": config["method"],
        "epsilon": config["epsilon"],
        "epsilon_rgb": config["epsilon_rgb"],
        "steps": config["steps"],
        "configuration": f"{config['method']}|{config['epsilon']}|{config['steps']}",
        "n_scenes": len(rows),
        "n_source_clean_correct": n_source_clean,
        "n_source_not_clean": n_source_not_clean,
        "n_source_clean_missing": n_source_clean_missing,
        "n_source_success": n_source_success,
        "n_source_failure": n_source_failure,
        "n_source_missing": n_source_missing,
        "n_source_success_denominator": source_denominator,
        "n_conditional_transfer": n_transfer,
        "n_conditional_transfer_failure": n_no_transfer,
        "n_conditional_transfer_denominator": transfer_denominator,
        "n_excluded_target_not_clean": n_excluded,
        "n_transfer_missing": n_transfer_missing,
        "n_target_clean_correct": n_target_clean,
        "n_target_flip": n_target_flip,
        "n_target_flip_denominator": target_denominator,
        "n_target_missing": n_target_missing,
        "median_labeled_nll_increase_on_source_clean": (
            None if not increases else float(np.median(np.asarray(increases, dtype=np.float64)))
        ),
        "n_labeled_nll_increase": len(increases),
        "definition": (
            "Source success is a correct clean neutral answer whose official neutral decision "
            "leaves the label. Conditional transfer is among those source successes that were "
            "also correct on the clean helpful answer, the fraction whose official helpful "
            "decision leaves the label. Scenes already wrong on the helpful answer are excluded. "
            "A missing decision stays out of both the numerator and the denominator."
        ),
    }
    block.update(classify_configuration(block))
    return block


def _increase(row: Mapping[str, Any]) -> float | None:
    clean = _view((row.get("clean") or {}).get(SOURCE_CONTEXT))
    perturbed = _view((row.get("perturbed") or {}).get(SOURCE_CONTEXT))
    if clean["nll"] is None or perturbed["nll"] is None:
        return None
    return perturbed["nll"] - clean["nll"]


def flatten_scenes(scenes: Mapping[str, Any], scene_ids: list[str]) -> list[dict[str, Any]]:
    """One row per finished scene and recorded configuration. Unfinished scenes are omitted."""
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = scenes.get(scene_id)
        if not isinstance(scene, Mapping) or scene.get("finished") is not True:
            continue
        clean = scene.get("clean") or {}
        for config in configurations():
            block = _lookup(scene, str(config["method"]), str(config["epsilon"]), int(config["steps"]))
            perturbed = (block or {}).get("perturbed") or {}
            flags = scene_flags(str(scene["answer"]), clean, perturbed)
            error = (block or {}).get("error") or scene.get("error")
            rows.append(
                {
                    "scene_id": scene_id,
                    "counterfactual_group_id": scene.get("counterfactual_group_id"),
                    "answer": scene.get("answer"),
                    "method": config["method"],
                    "epsilon": config["epsilon"],
                    "epsilon_rgb": config["epsilon_rgb"],
                    "steps": config["steps"],
                    "source_context": SOURCE_CONTEXT,
                    "evaluation_contexts": list(EVAL_CONTEXTS),
                    "clean": clean,
                    "perturbed": {
                        context: _view(perturbed.get(context)) for context in EVAL_CONTEXTS
                    },
                    "source_flip": flags["source_flip"],
                    "target_flip": flags["target_flip"],
                    "transfer": flags["transfer"],
                    "transfer_status": flags["transfer_status"],
                    "linf_before_round_trip": None if block is None else block.get("linf_before_round_trip"),
                    "linf_after_round_trip": None if block is None else block.get("linf_after_round_trip"),
                    "linf_stored_vs_projected": None if block is None else block.get("linf_stored_vs_projected"),
                    "seconds": None if block is None else block.get("seconds"),
                    "memory": None if block is None else block.get("memory"),
                    "os_available_bytes": None if block is None else block.get("os_available_bytes"),
                    "error": error,
                    "h_star_fitted": False,
                }
            )
    return rows


def _lookup(scene: Mapping[str, Any], method: str, epsilon_key: str, steps: int) -> dict[str, Any] | None:
    epsilons = scene.get("epsilons")
    if not isinstance(epsilons, Mapping):
        return None
    block = epsilons.get(epsilon_key)
    if not isinstance(block, Mapping):
        return None
    if method == "pgd":
        recorded = block.get("pgd")
        if not isinstance(recorded, Mapping):
            return None
        found = recorded.get(str(steps))
    else:
        found = block.get(method)
    return found if isinstance(found, dict) else None


def analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Complete official decisions in the Checkpoint 0 attack-row shape.

    A missing decision is left out. It is not encoded as an incorrect answer.
    """
    encoded: list[dict[str, Any]] = []
    for row in rows:
        for context in EVAL_CONTEXTS:
            clean = _view((row.get("clean") or {}).get(context))
            perturbed = _view((row.get("perturbed") or {}).get(context))
            if clean["decision"] is None or perturbed["decision"] is None:
                continue
            label = str(row["answer"])
            encoded.append(
                {
                    "scene_id": row["scene_id"],
                    "method": row["method"],
                    "epsilon": float(row["epsilon_rgb"]),
                    "step_budget": int(row["steps"]),
                    "source_context": SOURCE_CONTEXT,
                    "target_context": context,
                    "correct_before": clean["decision"] == label,
                    "correct_after": perturbed["decision"] == label,
                    "invalid_after": False,
                }
            )
    return encoded
