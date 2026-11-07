"""Stored Checkpoint 1.8 scalars for the calibration scenes.

The attack does not call this module. The numbers are not a sample rule and not a regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import PRIOR_PROGRESS_RELATIVE


def load_prior(root: Path) -> dict[str, Any]:
    payload = json.loads((root / PRIOR_PROGRESS_RELATIVE).read_text(encoding="utf-8"))
    if payload.get("attacks_run") is True:
        raise RuntimeError("Checkpoint 1.8 progress says an attack already ran.")
    return payload


def describe_scenes(prior: dict[str, Any], scene_ids: list[str]) -> list[dict[str, Any]]:
    """Alignment, clean margins, and neutral gradient L1. Missing stays missing."""
    scenes = prior.get("scenes") or {}
    rows = []
    for scene_id in scene_ids:
        block = scenes.get(scene_id) or {}
        neutral = block.get("neutral") or {}
        helpful = block.get("helpful_nonrevealing") or {}
        gradient = neutral.get("gradient") or {}
        rows.append(
            {
                "scene_id": scene_id,
                "answer": block.get("answer"),
                "counterfactual_group_id": block.get("counterfactual_group_id"),
                "alignment_neutral_to_helpful": _number(block.get("alignment_neutral_to_helpful")),
                "neutral_margin": _number(neutral.get("margin")),
                "helpful_margin": _number(helpful.get("margin")),
                "neutral_gradient_l1": _number(gradient.get("l1")),
                "used_to_choose_the_sample": False,
                "used_to_choose_a_configuration": False,
                "regressed_on_transfer": False,
            }
        )
    return rows


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
