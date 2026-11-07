"""All 96 development_v4 scenes. This module does not load a model."""

from __future__ import annotations

from typing import Any

from attack_feasibility.steps import scene_seed

from .protocol import N_GROUPS, N_SCENES, SAMPLE_RULE, SEED_PREFIX


def select_development(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Both arms of every counterfactual group, in group-id order.

    A group count other than 48 returns no scenes. Alignment is not read.
    """
    groups = sorted({str(record["counterfactual_group_id"]) for record in records})
    result: dict[str, Any] = {
        "ok": False,
        "rule": SAMPLE_RULE,
        "n_groups": len(groups),
        "n_scenes": 0,
        "expected_group_count": N_GROUPS,
        "expected_scene_count": N_SCENES,
        "selected_group_ids": [],
        "scenes": [],
        "reason": None,
    }
    if len(groups) != N_GROUPS or len(records) != N_SCENES:
        result["reason"] = (
            f"The development manifest has {len(groups)} groups and {len(records)} scenes, "
            f"not {N_GROUPS} groups and {N_SCENES} scenes. No attack was run."
        )
        return result
    scenes: list[dict[str, Any]] = []
    for group_id in groups:
        arms = sorted(
            (record for record in records if record["counterfactual_group_id"] == group_id),
            key=lambda record: str(record["scene_id"]),
        )
        if len(arms) != 2:
            result["reason"] = (
                f"{group_id} has {len(arms)} scenes. Both arms are required, and no replacement arm was chosen."
            )
            result["selected_group_ids"] = groups
            return result
        for record in arms:
            if record.get("split") != "development" or record.get("task_family") != "referent_color":
                result["reason"] = (
                    f"{record.get('scene_id')} is outside the development referent_color sample. No attack was run."
                )
                result["selected_group_ids"] = groups
                return result
            scene_id = str(record["scene_id"])
            scenes.append(
                {
                    "scene_id": scene_id,
                    "group_id": group_id,
                    "answer": str(record["answer"]),
                    "seed": scene_seed(SEED_PREFIX, scene_id),
                }
            )
    if len(scenes) != N_SCENES:
        result["reason"] = "The arm rule did not return 96 scenes. No attack was run."
        result["selected_group_ids"] = groups
        return result
    result["ok"] = True
    result["selected_group_ids"] = groups
    result["scenes"] = scenes
    result["n_scenes"] = len(scenes)
    return result
