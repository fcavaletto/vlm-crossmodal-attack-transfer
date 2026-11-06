"""Closed vocabulary for the row-color development sample.

Color RGB triples are the Checkpoint 1 display palette, so the four color
names still refer to those hues. Every object is large. A small shape is not
part of this task. The radius is 60 pixels, larger than the Checkpoint 1
radius of 46, on the same 128 by 192 cell.
"""

from __future__ import annotations

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SIZES: tuple[str, ...] = ("large",)
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

RADIUS: dict[str, int] = {"large": 60}
