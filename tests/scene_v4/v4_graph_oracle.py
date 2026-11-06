"""Independent answer reader.

This file does not import ``scene_v4``. It reads the question and the object
list only. Context strings are not an input.
"""

from __future__ import annotations

import re

_QUESTION = re.compile(
    r"^What color is the (large|small) (circle|square|triangle) in the (top|bottom) row\?$"
)
_ROW = {"top": 0, "bottom": 1}


def graph_answer(question: str, objects: list[dict]) -> str | None:
    """Color of the one named size and shape in the named row, else ``None``."""
    match = _QUESTION.fullmatch(question)
    if match is None:
        return None
    size = match.group(1)
    shape = match.group(2)
    row = _ROW[match.group(3)]
    hits = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size
        and obj["category"] == shape
        and obj["position"]["row"] == row
    ]
    if len(hits) != 1:
        return None
    return hits[0]["attributes"]["color"]
