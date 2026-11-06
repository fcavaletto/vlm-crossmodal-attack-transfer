"""Checkpoint 1 strict parser, kept as the official free-generation metric.

The parser is ``vlm_iface.parsing.parse_generation``. A sentence that merely
contains one vocabulary item stays invalid. That embedded label is returned
beside the official judgment and is not selected.
"""

from __future__ import annotations

import re

from vlm_iface.parsing import alias_to_canonical, normalize_answer, parse_generation

from .protocol import EMBEDDED_READING_ADOPTED, FREE_GENERATION_ROLE, OFFICIAL_METRIC
from .vocabulary import task_vocabulary


def score_free_generation(
    text: str,
    *,
    truncated: bool,
    task_family: str,
    answer: str,
) -> dict[str, object]:
    """Official correct / valid-wrong / invalid partition for one greedy string."""
    if answer not in task_vocabulary(task_family):
        raise ValueError(f"{answer!r} is not a {task_family} class")
    parsed = parse_generation(text, truncated=truncated, task_family=task_family)
    if parsed.invalid and parsed.parsed_answer is not None:
        raise ValueError("an invalid parse has no parsed answer")
    if parsed.invalid:
        judgment = {"correct": False, "valid_wrong": False, "invalid": True}
    elif parsed.parsed_answer == answer:
        judgment = {"correct": True, "valid_wrong": False, "invalid": False}
    else:
        if parsed.parsed_answer not in task_vocabulary(task_family):
            raise ValueError(f"parsed answer {parsed.parsed_answer!r} is outside the vocabulary")
        judgment = {"correct": False, "valid_wrong": True, "invalid": False}
    embedded = embedded_single_label(text, truncated=truncated, task_family=task_family)
    return {
        "metric": OFFICIAL_METRIC,
        "role": FREE_GENERATION_ROLE,
        "is_secondary": False,
        "text": parsed.text,
        "parsed_answer": parsed.parsed_answer,
        "reasons": list(parsed.reasons),
        "correct": judgment["correct"],
        "valid_wrong": judgment["valid_wrong"],
        "invalid": judgment["invalid"],
        "embedded_label": embedded,
        "embedded_applicable": embedded is not None,
        "embedded_adopted": EMBEDDED_READING_ADOPTED,
    }


def embedded_single_label(text: str, *, truncated: bool, task_family: str) -> str | None:
    """The single vocabulary item inside a ``not_exact`` string, if there is one.

    Truncation, a refusal, or any other reason keeps this reading unavailable.
    """
    strict = parse_generation(text, truncated=truncated, task_family=task_family)
    if strict.reasons != ("not_exact",):
        return None
    aliases = alias_to_canonical(task_family)
    found: list[str] = []
    for token in re.split(r"[^a-z0-9]+", normalize_answer(strict.text)):
        if not token:
            continue
        canonical = aliases.get(token)
        if canonical is not None and canonical not in found:
            found.append(canonical)
    if len(found) == 1:
        return found[0]
    return None


def adopt_embedded_reading(*_args: object, **_kwargs: object) -> None:
    """Refuse the unselected sentence reading."""
    raise RuntimeError(
        "The embedded single-label reading is not an outcome. "
        "Describe it with embedded_read_counterfactual and leave the strict parse in place."
    )
