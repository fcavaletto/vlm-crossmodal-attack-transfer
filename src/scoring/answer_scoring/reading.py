"""Strict parse, kept beside one embedded-label reading.

The embedded reading is recorded when D4's reason is exactly ``not_exact`` and
the string contains one vocabulary item. It is not the official parse. Calling
it official would reassign rows that Checkpoint 1 already stored, which would
choose a parser from those outputs.
"""

from __future__ import annotations

import re

from vlm_iface.parsing import alias_to_canonical, normalize_answer, parse_generation


def side_by_side(text: str, *, truncated: bool, task_family: str) -> dict[str, object]:
    """Return the strict result and, separately, the embedded single label."""
    strict = parse_generation(text, truncated=truncated, task_family=task_family)
    alternative = _embedded_label(strict.text, task_family) if strict.reasons == ("not_exact",) else None
    return {
        "strict_text": strict.text,
        "strict_parsed_answer": strict.parsed_answer,
        "strict_invalid": strict.invalid,
        "strict_reasons": list(strict.reasons),
        "alternative_name": "embedded_single_label",
        "alternative_selected": False,
        "alternative_parsed_answer": alternative,
        "alternative_applicable": alternative is not None,
    }


def _embedded_label(text: str, task_family: str) -> str | None:
    normalized = normalize_answer(text)
    aliases = alias_to_canonical(task_family)
    found: list[str] = []
    for token in re.split(r"[^a-z0-9]+", normalized):
        if not token:
            continue
        canonical = aliases.get(token)
        if canonical is not None and canonical not in found:
            found.append(canonical)
    if len(found) == 1:
        return found[0]
    return None
