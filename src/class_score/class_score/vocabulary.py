"""D4 candidate answers.

Attribute and spatial answers are the four color names. Counting answers are
the words one through four. The digits 1 through 4 are aliases of those words,
not extra classes. No simpler attribute vocabulary is defined here.
"""

from __future__ import annotations

COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
COUNT_WORDS: tuple[str, ...] = ("one", "two", "three", "four")
COUNT_DIGIT_ALIASES: dict[str, str] = {"1": "one", "2": "two", "3": "three", "4": "four"}

TASK_FAMILIES: tuple[str, ...] = ("attribute", "counting", "spatial")


def task_vocabulary(task_family: str) -> tuple[str, ...]:
    """Canonical class names, in the order used for a name tie."""
    if task_family in {"attribute", "spatial"}:
        return COLORS
    if task_family == "counting":
        return COUNT_WORDS
    raise ValueError(f"unknown task family: {task_family}")


def alias_to_canonical(task_family: str) -> dict[str, str]:
    """Normalized surface form to the canonical class name."""
    mapping = {word: word for word in task_vocabulary(task_family)}
    if task_family == "counting":
        mapping.update(COUNT_DIGIT_ALIASES)
    return mapping


def aliases_for(task_family: str, canonical: str) -> tuple[str, ...]:
    """Spellings that canonicalize to ``canonical``, canonical word first."""
    if canonical not in task_vocabulary(task_family):
        raise ValueError(f"{canonical!r} is not a {task_family} class")
    if task_family != "counting":
        return (canonical,)
    digits = tuple(digit for digit, word in COUNT_DIGIT_ALIASES.items() if word == canonical)
    return (canonical, *digits)


def spelling_forms(canonical: str) -> tuple[str, ...]:
    """Lowercase, capitalized, and uppercase spellings, without duplicates."""
    capitalized = canonical[:1].upper() + canonical[1:]
    upper = canonical.upper()
    forms = [canonical]
    if capitalized not in forms:
        forms.append(capitalized)
    if upper not in forms:
        forms.append(upper)
    return tuple(forms)
