"""Parse a greedy generation into a short answer, or mark it invalid.

``invalid`` means the string is not a single short answer. It does not mean the
answer is factually wrong, and it is not an attack success. Correctness is a
later comparison against the scene label and its aliases.

The length cap is provisional. This study's answers are not a closed global
vocabulary, unlike Study 1's color and shape lists. A long spatial phrase is
rejected until the research lead freezes the answer format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_ANSWER_TOKENS = 4
MAX_ANSWER_CHARS = 32

_PREFIX = re.compile(r"^(answer|response)\s*:\s*")
_REFUSAL = re.compile(
    r"\b(i don't know|i do not know|i cannot|i can't|cannot tell|can't tell|not sure|unsure|unknown)\b"
)
_ALNUM = re.compile(r"[a-z0-9]")


@dataclass(frozen=True)
class ParseResult:
    """``as_contract`` is the dict ``generate`` returns. ``reasons`` stays internal."""

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


def parse_generation(text: str, *, truncated: bool) -> ParseResult:
    """Parse ``text``. ``truncated`` is true when generation stopped without an end token."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    visible = text.strip()
    reasons: list[str] = []
    normalized = ""
    if truncated:
        reasons.append("truncated")
    if visible == "":
        reasons.append("empty")
    elif "\n" in visible or "\r" in visible:
        reasons.append("multiline")
    else:
        normalized = _normalize(visible)
        if normalized == "":
            reasons.append("empty")
        elif _ALNUM.search(normalized) is None:
            reasons.append("no_alphanumeric")
        elif _REFUSAL.search(normalized):
            reasons.append("refusal")
        elif len(normalized) > MAX_ANSWER_CHARS or len(normalized.split()) > MAX_ANSWER_TOKENS:
            reasons.append("too_long")
    if reasons:
        return ParseResult(visible, None, True, tuple(reasons))
    return ParseResult(visible, normalized, False, ())


def _normalize(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    collapsed = _PREFIX.sub("", collapsed).strip()
    collapsed = collapsed.strip("\"'`()[]{}")
    collapsed = re.sub(r"[.!?,;:]+$", "", collapsed).strip()
    return collapsed
