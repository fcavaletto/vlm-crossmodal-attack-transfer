"""Closed vocabularies for the development sample.

Color RGB triples match Study 1's display palette so the color names refer to
the same hues. That reuse is cosmetic. It does not import Study 1's single-object
task, splits, or context wording.
"""

from __future__ import annotations

SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow")
SIZES: tuple[str, ...] = ("small", "large")

# Counting answers used in this sample. "one" and "four" exist so a false
# count still has a word, and so leakage tests can ban those words too.
COUNT_WORDS: dict[int, str] = {1: "one", 2: "two", 3: "three", 4: "four"}
WORD_TO_COUNT: dict[str, int] = {word: count for count, word in COUNT_WORDS.items()}

# Display colors. Not a claim about perceptual uniformity.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (213, 53, 65),
    "blue": (48, 109, 207),
    "green": (49, 153, 97),
    "yellow": (231, 179, 39),
}
BACKGROUND: tuple[int, int, int] = (236, 236, 236)
OUTLINE: tuple[int, int, int] = (32, 32, 32)

RADIUS: dict[str, int] = {"large": 46, "small": 22}
