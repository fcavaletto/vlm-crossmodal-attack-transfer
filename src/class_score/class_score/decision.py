"""Closed-vocabulary decision from a class loss.

The chosen class is the lowest loss. Equal losses break by class name. A valid
free-generation parse disagrees with that choice only when the parsed class is
worse by more than 1e-6. An invalid generation is not a disagreement.
"""

from __future__ import annotations

from typing import Mapping

from .protocol import CLASS_ARGMIN_IS_PRIMARY, CLASS_DECISION_ROLE, DISAGREEMENT_TOLERANCE


def argmin_class(scores: Mapping[str, float]) -> str:
    """Lowest loss, then the class name."""
    if not scores:
        raise ValueError("scores are empty")
    return min(scores, key=lambda name: (float(scores[name]), name))


def is_disagreement(
    parsed_answer: str | None,
    invalid: bool,
    scores: Mapping[str, float],
    *,
    tolerance: float = DISAGREEMENT_TOLERANCE,
) -> bool:
    """True when a valid parsed class is worse than the argmin by more than ``tolerance``."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    chosen = argmin_class(scores)
    if invalid or parsed_answer is None:
        return False
    if parsed_answer not in scores:
        raise ValueError(f"parsed answer {parsed_answer!r} was not scored")
    return float(scores[parsed_answer]) > float(scores[chosen]) + tolerance


def class_decision(scores: Mapping[str, float], answer: str) -> dict[str, object]:
    """Readout for one row. This is not the official primary outcome."""
    if answer not in scores:
        raise ValueError(f"answer {answer!r} was not scored")
    chosen = argmin_class(scores)
    return {
        "metric": "class_argmin",
        "role": CLASS_DECISION_ROLE,
        "is_primary_outcome": CLASS_ARGMIN_IS_PRIMARY,
        "chosen": chosen,
        "correct": chosen == answer,
        "loss": float(scores[chosen]),
    }
