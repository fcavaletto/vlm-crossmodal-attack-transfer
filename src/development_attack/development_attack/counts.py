"""Counts for the frozen attack and its three comparisons. No regression."""

from __future__ import annotations

from typing import Any, Mapping

from attack_calibration.counts import _count_block

from .protocol import EPSILON_KEY, EVAL_CONTEXTS, SOURCE_CONTEXT, configurations


def flatten_scenes(scenes: Mapping[str, Any], scene_ids: list[str]) -> list[dict[str, Any]]:
    """One row per finished scene and recorded configuration."""
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = scenes.get(scene_id)
        if not isinstance(scene, Mapping) or scene.get("finished") is not True:
            continue
        clean = scene.get("clean") or {}
        for config in configurations():
            block = _lookup(scene, str(config["method"]), int(config["steps"]))
            perturbed = (block or {}).get("perturbed") or {}
            rows.append(
                {
                    "scene_id": scene_id,
                    "counterfactual_group_id": scene.get("counterfactual_group_id"),
                    "answer": scene.get("answer"),
                    "method": config["method"],
                    "epsilon": config["epsilon"],
                    "epsilon_rgb": config["epsilon_rgb"],
                    "steps": config["steps"],
                    "role": config["role"],
                    "source_context": SOURCE_CONTEXT,
                    "evaluation_contexts": list(EVAL_CONTEXTS),
                    "clean": clean,
                    "perturbed": perturbed,
                    "linf_before_round_trip": None if block is None else block.get("linf_before_round_trip"),
                    "linf_after_round_trip": None if block is None else block.get("linf_after_round_trip"),
                    "seconds": None if block is None else block.get("seconds"),
                    "error": None if block is None else block.get("error"),
                    "recomputed_neutral_gradient_l1": scene.get("recomputed_neutral_gradient_l1"),
                    "h_star_fitted": False,
                }
            )
    return rows


def configuration_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for config in configurations():
        selected = [
            row
            for row in rows
            if row.get("method") == config["method"]
            and row.get("epsilon") == config["epsilon"]
            and row.get("steps") == config["steps"]
        ]
        block = _count_block(config, selected)
        block["role"] = config["role"]
        blocks.append(block)
    return blocks


def _lookup(scene: Mapping[str, Any], method: str, steps: int) -> dict[str, Any] | None:
    epsilons = scene.get("epsilons")
    if not isinstance(epsilons, Mapping):
        return None
    block = epsilons.get(EPSILON_KEY)
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
