"""Leakage normalization.

The steps match ``scene_gen.normalize.normalize_for_leakage``. Tests compare
the two functions on the same strings. This is not a model parser.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_for_leakage(text: str) -> str:
    """Casefold, turn non-alphanumeric runs into one space, and strip."""
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
