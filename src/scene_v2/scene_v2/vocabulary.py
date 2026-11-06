"""Closed vocabularies for the revised development sample.

Color RGB triples are the same display palette as the Checkpoint 1 generator,
so the color names still refer to those hues. Radii are unchanged: a large
shape is visibly larger than a small one (46 versus 22).
"""

from __future__ import annotations

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SIZES: tuple[str, ...] = ("small", "large")

COUNT_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four"}
WORD_TO_COUNT: dict[str, int] = {word: count for count, word in COUNT_WORDS.items()}

PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (213, 53, 65),
    "blue": (48, 109, 207),
    "green": (49, 153, 97),
    "yellow": (231, 179, 39),
}
BACKGROUND: tuple[int, int, int] = (236, 236, 236)
OUTLINE: tuple[int, int, int] = (32, 32, 32)

RADIUS: dict[str, int] = {"large": 46, "small": 22}
