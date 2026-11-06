"""Strict free-generation parse, kept as a secondary diagnostic.

The parser is ``vlm_iface.parsing.parse_generation``, unchanged. Correct,
valid-but-wrong, and invalid are a partition of greedy strings. They are not
closed-vocabulary accuracy. A sentence that merely contains one color stays
invalid. This module does not adopt that embedded reading.
"""

from __future__ import annotations

from vlm_iface.parsing import parse_generation

from .colors import require_color
from .protocol import (
    COLOR_PARSER_TASK_FAMILY,
    FREE_GENERATION_IS_PRIMARY,
    FREE_GENERATION_ROLE,
)


def diagnose_free_generation(text: str, *, truncated: bool, label: str) -> dict[str, object]:
    """Parse one greedy string on the four color names."""
    require_color(label)
    if not isinstance(truncated, bool):
        raise TypeError("truncated must be a bool")
    parsed = parse_generation(text, truncated=truncated, task_family=COLOR_PARSER_TASK_FAMILY)
    if parsed.invalid:
        correct, valid_but_wrong = False, False
    elif parsed.parsed_answer == label:
        correct, valid_but_wrong = True, False
    else:
        if parsed.parsed_answer is None:
            raise RuntimeError("a valid parse is missing its color")
        require_color(parsed.parsed_answer)
        correct, valid_but_wrong = False, True
    if int(correct) + int(valid_but_wrong) + int(parsed.invalid) != 1:
        raise RuntimeError("free-generation flags are not a partition")
    return {
        "role": FREE_GENERATION_ROLE,
        "is_primary": FREE_GENERATION_IS_PRIMARY,
        "metric": "strict_free_generation",
        "parser": "vlm_iface.parsing.parse_generation",
        "parser_task_family": COLOR_PARSER_TASK_FAMILY,
        "text": parsed.text,
        "parsed_answer": parsed.parsed_answer,
        "reasons": list(parsed.reasons),
        "correct": correct,
        "valid_but_wrong": valid_but_wrong,
        "invalid": parsed.invalid,
    }
