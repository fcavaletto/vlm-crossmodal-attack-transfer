"""The four closed-vocabulary colors.

Counting words are not classes in this protocol. Case variants are not extra
classes. The tie break sorts these names, which is not this tuple's order.
"""

from __future__ import annotations

from collections.abc import Mapping

COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")


def require_color(color: str) -> str:
    """Return ``color`` when it is one of the four names."""
    if color not in COLORS:
        raise ValueError(f"{color!r} is not one of {COLORS}")
    return color


def require_color_keys(mapping: Mapping[str, object], *, what: str) -> None:
    """Require exactly the four color names and no other key."""
    if set(mapping) != set(COLORS):
        raise ValueError(f"{what} must be exactly {COLORS}, got {tuple(mapping)}")
