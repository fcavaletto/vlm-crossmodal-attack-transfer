"""Argmin of summed NLL, with a deterministic color-name tie break.

The sort key is the numeric score, then the color name. ``blue`` precedes
``green``, ``red``, and ``yellow``. Dict insertion order is not the tie break.
The name order is not the order of the color tuple.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .colors import COLORS, require_color_keys


def as_float(value: object) -> float:
    """Finite Python float for a score. Does not keep a gradient."""
    if isinstance(value, bool):
        raise TypeError("score must be a float or a scalar tensor")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("score must be a scalar")
        number = float(value.detach().to(dtype=torch.float64).reshape(-1)[0].cpu())
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise TypeError(f"score must be a float or a scalar tensor, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError("score is not finite")
    return number


def rank_colors(scores: Mapping[str, object]) -> tuple[str, ...]:
    """Lowest summed NLL first. Equal scores break by color name."""
    require_color_keys(scores, what="scores")
    keyed = {color: as_float(scores[color]) for color in COLORS}
    return tuple(sorted(COLORS, key=lambda color: (keyed[color], color)))


def choose_color(scores: Mapping[str, object]) -> str:
    """The primary closed-vocabulary decision."""
    return rank_colors(scores)[0]
