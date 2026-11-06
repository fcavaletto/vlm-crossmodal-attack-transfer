"""Ground truth read off the scene graph.

The answer is the color of the single object that has the named shape and
sits in the named row. Top is row 0. Bottom is row 1. The function returns
``None`` when the question does not parse or the shape-and-row pair is not
unique. It does not read a context string.
"""

from __future__ import annotations

import re
from typing import Protocol

from .vocabulary import ROW_INDEX

QUESTION_RE = re.compile(
    r"^What color is the (circle|square|triangle) in the (top|bottom) row\?$"
)


class _Object(Protocol):
    shape: str
    color: str
    row: int


def row_color_answer(question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    """Color of the unique named shape in the named row, or ``None``."""
    match = QUESTION_RE.fullmatch(question)
    if match is None:
        return None
    shape, row_name = match.group(1), match.group(2)
    row = ROW_INDEX[row_name]
    hits = [obj for obj in objects if obj.shape == shape and obj.row == row]
    if len(hits) != 1:
        return None
    return hits[0].color


def answer_for(task_family: str, question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    """Dispatch on the only family this package grades."""
    if task_family != "row_color":
        return None
    return row_color_answer(question, objects)
