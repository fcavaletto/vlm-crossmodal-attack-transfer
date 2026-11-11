"""Summaries from compact records. H* is not fit."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from dev_pilot.lock import N_SCENES
from dev_pilot.summarize import eligible_groups

from .protocol import PRIMARY_CONTEXTS
from .records import distribution, norms_summary

EXPECTED_MEASUREMENTS = N_SCENES * len(PRIMARY_CONTEXTS)


def scene_list(scenes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One block per scene id, in scene-id order."""
    return [scenes[scene_id] for scene_id in sorted(scenes)]


def measurement_complete(scenes: dict[str, dict[str, Any]], scene_ids: list[str]) -> bool:
    """True when every scene has both primary contexts, including recorded misses."""
    if len(scene_ids) != N_SCENES or len(set(scene_ids)) != N_SCENES:
        return False
    for scene_id in scene_ids:
        block = scenes.get(scene_id)
        if not isinstance(block, dict):
            return False
        for context in PRIMARY_CONTEXTS:
            entry = block.get(context)
            if not isinstance(entry, dict) or "gradient" not in entry:
                return False
    return True


def missing_gradients(scenes: dict[str, dict[str, Any]], scene_ids: list[str]) -> list[dict[str, Any]]:
    """Every scene-context whose RGB gradient is absent. A zero gradient is not listed."""
    missing: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        block = scenes.get(scene_id)
        for context in PRIMARY_CONTEXTS:
            entry = block.get(context) if isinstance(block, dict) else None
            gradient = entry.get("gradient") if isinstance(entry, dict) else None
            if isinstance(gradient, dict) and gradient.get("present") is True:
                continue
            reason = "not recorded"
            if isinstance(gradient, dict) and gradient.get("reason"):
                reason = str(gradient["reason"])
            elif isinstance(entry, dict) and entry.get("stopped_reason"):
                reason = str(entry["stopped_reason"])
            missing.append({"scene_id": scene_id, "context": context, "reason": reason})
    return missing


def eligibility_view(scenes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The mapping ``eligible_groups`` already knows how to read."""
    view: dict[str, dict[str, Any]] = {}
    for scene_id, block in scenes.items():
        item: dict[str, Any] = {"alignment": block.get("eligibility_alignment")}
        for context in PRIMARY_CONTEXTS:
            gradient = block.get(context, {}).get("gradient", {})
            finite = gradient.get("finite") is True
            item[context] = {
                "finite": True if finite else False if gradient.get("present") is True else None,
                "l1": gradient.get("l1") if finite else None,
            }
        view[scene_id] = item
    return view


def alignment_summary(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Both directed alignments. Undefined and missing values are counted, not averaged."""
    blocks = scene_list(scenes)
    return {
        "neutral_to_helpful": _alignment_distribution(blocks, "alignment_neutral_to_helpful"),
        "helpful_to_neutral": _alignment_distribution(blocks, "alignment_helpful_to_neutral"),
        "regressed_on_attack_outcomes": False,
        "h_star_fitted": False,
    }


def margin_summary(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Clean labeled-color margins. A missing loss is not a margin of zero."""
    per_context = {}
    for context in PRIMARY_CONTEXTS:
        values: list[float | None] = []
        missing = 0
        for block in scene_list(scenes):
            entry = block.get(context)
            margin = entry.get("margin") if isinstance(entry, dict) else None
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                values.append(None)
                missing += 1
            else:
                values.append(float(margin))
        per_context[context] = distribution(values, undefined=0, missing=missing)
    return per_context


def build_summary(
    *,
    scenes: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    scene_ids: list[str],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Compact result. Raw gradient arrays are not accepted in ``scenes``."""
    _reject_raw_arrays(scenes)
    complete = measurement_complete(scenes, scene_ids)
    missing = missing_gradients(scenes, scene_ids)
    gradients = eligibility_view(scenes) if complete else None
    eligible = eligible_groups(records, rows, gradients)
    return {
        "label": "DEVELOPMENT ONLY",
        "attacks_run": False,
        "h_star_fitted": False,
        "perturbations_run": False,
        "measurement_complete": complete,
        "expected_measurements": EXPECTED_MEASUREMENTS,
        "recorded_scene_count": len(scenes),
        "gate": gate,
        "any_gradient_missing": bool(missing) or not complete,
        "missing_gradients": missing,
        "n_missing_gradients": len(missing) if complete else EXPECTED_MEASUREMENTS - _present_count(scenes),
        "n_zero_gradients": _count_flag(scenes, finite=True, nonzero=False),
        "n_nonfinite_gradients": _count_nonfinite(scenes),
        "alignment": alignment_summary(scenes),
        "norms": norms_summary(scene_list(scenes)),
        "clean_margins": margin_summary(scenes),
        "eligible": eligible,
        "n_groups_eligible": eligible["n_groups_eligible"],
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _alignment_distribution(blocks: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values: list[float | None] = []
    undefined = 0
    missing = 0
    for block in blocks:
        if block.get("undefined_reason") == "a gradient is missing" or _context_missing(block):
            values.append(None)
            missing += 1
            continue
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            values.append(None)
            undefined += 1
            continue
        values.append(float(value))
    return distribution(values, undefined=undefined, missing=missing)


def _context_missing(block: dict[str, Any]) -> bool:
    for context in PRIMARY_CONTEXTS:
        gradient = block.get(context, {}).get("gradient", {}) if isinstance(block.get(context), dict) else {}
        if gradient.get("present") is not True:
            return True
    return False


def _present_count(scenes: dict[str, dict[str, Any]]) -> int:
    count = 0
    for block in scenes.values():
        for context in PRIMARY_CONTEXTS:
            gradient = block.get(context, {}).get("gradient") if isinstance(block.get(context), dict) else None
            if isinstance(gradient, dict) and gradient.get("present") is True:
                count += 1
    return count


def _count_flag(scenes: dict[str, dict[str, Any]], *, finite: bool, nonzero: bool) -> int:
    count = 0
    for block in scenes.values():
        for context in PRIMARY_CONTEXTS:
            gradient = block.get(context, {}).get("gradient") if isinstance(block.get(context), dict) else None
            if not isinstance(gradient, dict):
                continue
            if gradient.get("finite") is finite and gradient.get("nonzero") is nonzero:
                count += 1
    return count


def _count_nonfinite(scenes: dict[str, dict[str, Any]]) -> int:
    count = 0
    for block in scenes.values():
        for context in PRIMARY_CONTEXTS:
            gradient = block.get(context, {}).get("gradient") if isinstance(block.get(context), dict) else None
            if isinstance(gradient, dict) and gradient.get("present") is True and gradient.get("finite") is False:
                count += 1
    return count


def _reject_raw_arrays(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"gradient_values", "g_source", "g_target", "pixel_values", "rgb"}:
                raise ValueError(f"refusing to store raw values under {key}")
            _reject_raw_arrays(item)
    elif isinstance(value, list):
        if len(value) > 64 and value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            raise ValueError("refusing to store a long numeric array")
        for item in value:
            _reject_raw_arrays(item)
