"""Four-scene gate. Scene choice does not read a gradient or a score."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from choice_eval.colors import COLORS

from .protocol import GATE_RULE_ID, PRIMARY_CONTEXTS


def gate_scene_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Lexicographically first ``scene_id`` of each labeled color.

    Sort key and color order are the only inputs. A field named like a
    gradient is not read. ``COLORS`` is red, blue, green, yellow.
    """
    chosen: dict[str, str] = {}
    ordered = sorted(records, key=lambda record: str(record["scene_id"]))
    for record in ordered:
        answer = record.get("answer")
        if answer not in COLORS or answer in chosen:
            continue
        chosen[str(answer)] = str(record["scene_id"])
    missing = [color for color in COLORS if color not in chosen]
    if missing:
        raise RuntimeError(f"the gate rule found no scene for {missing}")
    return [chosen[color] for color in COLORS]


def gate_scene_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """The four selected scenes with the color that selected them."""
    ids = gate_scene_ids(records)
    by_id = {str(record["scene_id"]): record for record in records}
    return [
        {
            "scene_id": scene_id,
            "answer": str(by_id[scene_id]["answer"]),
            "rule": GATE_RULE_ID,
        }
        for scene_id in ids
    ]


def judge_gate(measurements: Mapping[str, Mapping[str, Any]], scene_ids: Sequence[str]) -> dict[str, Any]:
    """Pass only when every selected scene has a finite nonzero gradient in both primary contexts.

    A stability or finite-difference disagreement is not this judgment.
    A missing gradient is not treated as zero.
    """
    failures: list[dict[str, str]] = []
    for scene_id in scene_ids:
        block = measurements.get(scene_id)
        if not isinstance(block, Mapping):
            for context in PRIMARY_CONTEXTS:
                failures.append(
                    {
                        "scene_id": scene_id,
                        "context": context,
                        "reason": "no measurement was recorded",
                    }
                )
            continue
        for context in PRIMARY_CONTEXTS:
            entry = block.get(context)
            gradient = entry.get("gradient") if isinstance(entry, Mapping) else None
            reason = _invalid_reason(gradient)
            if reason is not None:
                failures.append({"scene_id": scene_id, "context": context, "reason": reason})
    return {
        "rule": GATE_RULE_ID,
        "scene_ids": list(scene_ids),
        "contexts": list(PRIMARY_CONTEXTS),
        "passed": not failures,
        "failures": failures,
        "requires": "finite and nonzero RGB gradient of the labeled-color summed NLL",
    }


def _invalid_reason(gradient: object) -> str | None:
    if not isinstance(gradient, Mapping):
        return "no gradient record"
    if gradient.get("present") is not True:
        reason = gradient.get("reason")
        return str(reason) if reason else "gradient is missing"
    if gradient.get("finite") is not True:
        return "gradient is not finite"
    if gradient.get("nonzero") is not True:
        return "gradient is zero"
    return None
