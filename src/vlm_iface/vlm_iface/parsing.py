"""Parse a greedy generation against the task vocabulary.

Checkpoint 1 decision D4 replaces the Checkpoint 0 format-only parser.

Normalization is only case, whitespace, surrounding quotes, and terminal
punctuation. The normalized string must be exactly one task-vocabulary item or
a documented alias. Attribute and spatial questions in the current scenes ask
for a color, so their vocabulary is the four color names. Counting uses the
count words ``one`` through ``four``, and the digit strings ``1`` through ``4``
are aliases of those words.

``parsed_answer`` is the canonical word. A digit such as ``2`` is stored as
``two``. The raw generation stays in ``text``. A recognized color or count that
is not the scene's answer is still a valid parse; the caller marks it
valid-but-wrong. This function does not see the scene label.

Option-list echoes, more than one vocabulary item, refusals, and every other
string are invalid. Invalid is a task error. It is not an attack success.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TASK_FAMILIES: tuple[str, ...] = ("attribute", "counting", "spatial")
COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
COUNT_WORDS: tuple[str, ...] = ("one", "two", "three", "four")
COUNT_DIGIT_ALIASES: dict[str, str] = {"1": "one", "2": "two", "3": "three", "4": "four"}

_REFUSAL = re.compile(
    r"\b(i don't know|i do not know|i cannot|i can't|cannot tell|can't tell|not sure|unsure|unknown)\b"
)
_ALNUM = re.compile(r"[a-z0-9]")
_TERMINAL = re.compile(r"[.!?,;:]+$")
_QUOTES = "\"'`“”‘’"


@dataclass(frozen=True)
class ParseResult:
    """``as_contract`` is the dict ``generate`` returns. ``reasons`` stays with the caller."""

    text: str
    parsed_answer: str | None
    invalid: bool
    reasons: tuple[str, ...]

    def as_contract(self) -> dict[str, str | bool | None]:
        return {
            "text": self.text,
            "parsed_answer": self.parsed_answer,
            "invalid": self.invalid,
        }


def task_vocabulary(task_family: str) -> tuple[str, ...]:
    """Canonical answer words for one current task family, in a fixed order."""
    if task_family in {"attribute", "spatial"}:
        return COLORS
    if task_family == "counting":
        return COUNT_WORDS
    raise ValueError(f"unknown task family: {task_family}")


def alias_to_canonical(task_family: str) -> dict[str, str]:
    """Normalized surface form to the canonical vocabulary word."""
    canonical = {word: word for word in task_vocabulary(task_family)}
    if task_family == "counting":
        canonical.update(COUNT_DIGIT_ALIASES)
    return canonical


def normalize_answer(text: str) -> str:
    """Case, collapsed whitespace, surrounding quotes, and terminal punctuation."""
    collapsed = re.sub(r"\s+", " ", text.strip()).casefold()
    previous = None
    while previous != collapsed:
        previous = collapsed
        collapsed = collapsed.strip().strip(_QUOTES).strip()
        collapsed = _TERMINAL.sub("", collapsed).strip()
    return collapsed


def parse_generation(text: str, *, truncated: bool, task_family: str) -> ParseResult:
    """Parse ``text`` for ``task_family``.

    ``truncated`` is true when generation stopped without an end token. A
    truncated string is invalid even when the visible text is a vocabulary item.
    """
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"unknown task family: {task_family}")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    visible = text.strip()
    reasons: list[str] = []
    if truncated:
        reasons.append("truncated")
    if visible == "":
        reasons.append("empty")
        return ParseResult(visible, None, True, tuple(reasons))
    if "\n" in visible or "\r" in visible:
        reasons.append("multiline")
        return ParseResult(visible, None, True, tuple(reasons))
    normalized = normalize_answer(visible)
    if normalized == "":
        reasons.append("empty")
    elif _ALNUM.search(normalized) is None:
        reasons.append("no_alphanumeric")
    elif _REFUSAL.search(normalized):
        reasons.append("refusal")
    else:
        aliases = alias_to_canonical(task_family)
        if normalized in aliases and not reasons:
            return ParseResult(visible, aliases[normalized], False, ())
        if normalized in aliases:
            return ParseResult(visible, None, True, tuple(reasons))
        hits = _vocabulary_hits(normalized, aliases)
        if len(hits) >= 2:
            reasons.append("multiple_labels")
        elif len(hits) == 1:
            reasons.append("not_exact")
        else:
            reasons.append("not_in_vocabulary")
    return ParseResult(visible, None, True, tuple(reasons))


def _vocabulary_hits(normalized: str, aliases: dict[str, str]) -> list[str]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    found: list[str] = []
    for token in tokens:
        canonical = aliases.get(token)
        if canonical is not None and canonical not in found:
            found.append(canonical)
    return found
