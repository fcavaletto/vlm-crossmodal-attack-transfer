"""Scene rules from ATTACK_PROTOCOL.md. This module does not load a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import (
    COLOR_ORDER,
    PRIOR_PROGRESS_RELATIVE,
    STAGE_A_RULE_ID,
    STAGE_A_SCENES,
    STAGE_B_RULE_ID,
    STAGE_B_SCENE_IDS,
)


def load_prior_progress(root: Path) -> dict[str, Any]:
    path = root / PRIOR_PROGRESS_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError("Checkpoint 1.8 progress is not a completed record.")
    if payload.get("attacks_run") is True:
        raise RuntimeError("Checkpoint 1.8 progress says an attack already ran.")
    return payload


def select_stage_a(progress: dict[str, Any]) -> list[dict[str, Any]]:
    """Lowest and highest Checkpoint 1.8 alignment within each labeled color.

    The sample is exploratory. Alignment is read from the stored record only.
    """
    scenes = progress["scenes"]
    if len(scenes) != 96:
        raise RuntimeError(f"expected 96 Checkpoint 1.8 scenes, found {len(scenes)}")
    by_color: dict[str, list[dict[str, Any]]] = {color: [] for color in COLOR_ORDER}
    for scene_id, block in scenes.items():
        answer = block.get("answer")
        alignment = block.get("alignment_neutral_to_helpful")
        if answer not in by_color:
            raise RuntimeError(f"{scene_id} has an unexpected label {answer!r}")
        if not isinstance(alignment, (int, float)) or isinstance(alignment, bool):
            raise RuntimeError(f"{scene_id} has no stored alignment")
        if not -1.0 <= float(alignment) <= 1.0:
            raise RuntimeError(f"{scene_id} alignment is outside [-1, 1]")
        by_color[answer].append(
            {
                "scene_id": scene_id,
                "answer": answer,
                "alignment_neutral_to_helpful": float(alignment),
            }
        )
    selected: list[dict[str, Any]] = []
    for color in COLOR_ORDER:
        items = sorted(by_color[color], key=lambda row: (row["alignment_neutral_to_helpful"], row["scene_id"]))
        if len(items) < 2:
            raise RuntimeError(f"{color} does not have two scenes")
        low = dict(items[0])
        high = dict(items[-1])
        if low["scene_id"] == high["scene_id"]:
            raise RuntimeError(f"{color} extremes are the same scene")
        low["stratum"] = "lowest_alignment_within_color"
        high["stratum"] = "highest_alignment_within_color"
        selected.append(low)
        selected.append(high)
    return selected


def select_stage_b(progress: dict[str, Any]) -> list[str]:
    """Every sixth scene id after a lexicographic sort. Indexes 0, 6, ..., 90."""
    scene_ids = sorted(progress["scenes"])
    if len(scene_ids) != 96:
        raise RuntimeError(f"expected 96 Checkpoint 1.8 scenes, found {len(scene_ids)}")
    return [scene_ids[index] for index in range(0, 96, 6)]


def assert_frozen_selection(root: Path) -> dict[str, Any]:
    """Stop before a forward if the code and the protocol table disagree."""
    progress = load_prior_progress(root)
    stage_a = select_stage_a(progress)
    got = [
        (row["scene_id"], row["answer"], row["stratum"], row["alignment_neutral_to_helpful"])
        for row in stage_a
    ]
    if got != list(STAGE_A_SCENES):
        raise RuntimeError(
            "Stage A selection does not match the protocol table. "
            f"rule={STAGE_A_RULE_ID} got={got!r}"
        )
    stage_b = select_stage_b(progress)
    if tuple(stage_b) != STAGE_B_SCENE_IDS:
        raise RuntimeError(
            "Stage B selection does not match the protocol table. "
            f"rule={STAGE_B_RULE_ID} got={stage_b!r}"
        )
    return {"stage_a": stage_a, "stage_b": stage_b, "stage_a_rule": STAGE_A_RULE_ID, "stage_b_rule": STAGE_B_RULE_ID}
