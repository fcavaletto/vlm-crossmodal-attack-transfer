"""Normalization for the answer-leakage substring test.

This is not the model's answer parser. Agent C owns parsing of generated text.
The rule here is only: after normalization, is the answer string a substring
of the context string?

Steps of ``normalize_for_leakage``:

1. Unicode casefold (so ``RED`` and ``red`` match).
2. Replace every run of characters that are not ASCII letters or digits with a
   single space.
3. Strip the ends.

The test is a raw substring test on that result, not a whole-word test.
``one`` is a substring of ``none``. Context sentences are written so the raw
substring test passes, which is the stricter reading of the Checkpoint 0 contract.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_for_leakage(text: str) -> str:
    """Return the leakage-test form of ``text``."""
    folded = text.casefold()
    spaced = _NON_ALNUM.sub(" ", folded)
    return spaced.strip()


def leakage_aliases(text: str, aliases: list[str]) -> list[str]:
    """Aliases whose normalized form is a substring of normalized ``text``."""
    norm = normalize_for_leakage(text)
    found: list[str] = []
    for alias in aliases:
        needle = normalize_for_leakage(alias)
        if not needle:
            raise ValueError(f"alias {alias!r} normalizes to an empty string")
        if needle in norm:
            found.append(alias)
    return found
