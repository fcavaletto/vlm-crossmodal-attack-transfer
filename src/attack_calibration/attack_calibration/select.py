"""Group stride from the development manifest. This module does not load a model."""

from __future__ import annotations

from typing import Any

from attack_feasibility.steps import scene_seed

from .protocol import (
    EXPECTED_GROUP_COUNT,
    FROZEN_GROUP_IDS,
    FROZEN_SCENES,
    GROUP_INDEXES,
    RULE_ID,
    SEED_PREFIX,
    CalibrationStop,
)


def select_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Indexes 0, 3, ..., 45 of the lexicographic group ids, both arms.

    A group count other than 48 returns no scenes. It does not shorten the index list.
    """
    groups = sorted({str(record["counterfactual_group_id"]) for record in records})
    result: dict[str, Any] = {
        "ok": False,
        "rule": RULE_ID,
        "n_groups": len(groups),
        "expected_group_count": EXPECTED_GROUP_COUNT,
        "group_ids": groups,
        "selected_group_ids": [],
        "scenes": [],
    }
    if len(groups) != EXPECTED_GROUP_COUNT:
        result["reason"] = (
            f"The development manifest has {len(groups)} counterfactual groups, not "
            f"{EXPECTED_GROUP_COUNT}. No attack was run."
        )
        return result
    selected = [groups[index] for index in GROUP_INDEXES]
    scenes: list[dict[str, Any]] = []
    for group_id in selected:
        arms = sorted(
            (record for record in records if record["counterfactual_group_id"] == group_id),
            key=lambda record: str(record["scene_id"]),
        )
        if len(arms) != 2:
            result["reason"] = (
                f"{group_id} has {len(arms)} scenes. Both arms are required, and no replacement arm was chosen."
            )
            result["selected_group_ids"] = selected
            return result
        for record in arms:
            scene_id = str(record["scene_id"])
            scenes.append(
                {
                    "scene_id": scene_id,
                    "group_id": group_id,
                    "answer": str(record["answer"]),
                    "seed": scene_seed(SEED_PREFIX, scene_id),
                }
            )
    result["ok"] = True
    result["reason"] = None
    result["selected_group_ids"] = selected
    result["scenes"] = scenes
    return result


def assert_frozen_selection(selected: dict[str, Any]) -> None:
    """Stop before a forward when the manifest rule and the protocol table disagree."""
    if selected.get("ok") is not True:
        raise CalibrationStop(
            "stopped_group_count",
            str(selected.get("reason") or "The group rule did not return a sample."),
        )
    got_groups = tuple(selected["selected_group_ids"])
    if got_groups != FROZEN_GROUP_IDS:
        raise CalibrationStop(
            "stopped_scene_rule",
            "The group stride does not match the protocol table. No perturbation was run.",
        )
    got = tuple(
        (row["scene_id"], row["group_id"], row["answer"], row["seed"]) for row in selected["scenes"]
    )
    if got != FROZEN_SCENES:
        raise CalibrationStop(
            "stopped_scene_rule",
            "The scene list does not match the protocol table. No perturbation was run.",
        )
