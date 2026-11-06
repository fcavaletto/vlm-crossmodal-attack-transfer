"""Closed vocabulary for the referent-color development sample.

Color RGB triples are the Checkpoint 1 display palette. Large radius is 60,
the Checkpoint 1.6 radius, on the same 128 by 192 cell. Small radius is 28,
so a small shape can sit in the column next to a large one and still leave
an 8-pixel gap. The blank band between the rows stays empty.
"""

from __future__ import annotations

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SIZES: tuple[str, ...] = ("large", "small")
ROW_NAMES: tuple[str, ...] = ("top", "bottom")
ROW_INDEX: dict[str, int] = {"top": 0, "bottom": 1}

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (213, 53, 65),
    "blue": (48, 109, 207),
    "green": (49, 153, 97),
    "yellow": (231, 179, 39),
}
BACKGROUND: tuple[int, int, int] = (236, 236, 236)
OUTLINE: tuple[int, int, int] = (32, 32, 32)

RADIUS: dict[str, int] = {"large": 60, "small": 28}
