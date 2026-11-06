"""Ground truth read off the scene graph.

``answer_for`` returns the canonical word, or ``None`` when the scene does not
determine exactly one answer. Spatial scenes abstain when a second reading of
"left" names a different object: an extra object on the same row, another
object in the neighbor's column, or a left-half object that is not at least
twice as far as the adjacent cell.
"""

from __future__ import annotations

import re
from typing import Protocol

from .geometry import MIN_LEFT_RATIO, center, distance
from .vocabulary import COUNT_WORDS, WORD_TO_COUNT


class _Object(Protocol):
    object_id: str
    shape: str
    color: str
    size: str
    col: int
    row: int


ATTRIBUTE_RE = re.compile(r"^What color is the (large|small) (circle|square|triangle)\?$")
COUNTING_RE = re.compile(r"^How many (red|blue|green|yellow) objects are there\?$")
SPATIAL_RE = re.compile(
    r"^What color is the object immediately left of the "
    r"(large|small) (circle|square|triangle) in the same row\?$"
)


def attribute_answer(question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    match = ATTRIBUTE_RE.fullmatch(question)
    if match is None:
        return None
    size, shape = match.group(1), match.group(2)
    hits = [obj for obj in objects if obj.size == size and obj.shape == shape]
    if len(hits) != 1:
        return None
    return hits[0].color


def counting_answer(question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    match = COUNTING_RE.fullmatch(question)
    if match is None:
        return None
    color = match.group(1)
    count = sum(1 for obj in objects if obj.color == color)
    return COUNT_WORDS.get(count)


def _left_ranked(
    anchor: _Object, objects: list[_Object] | tuple[_Object, ...]
) -> list[tuple[float, _Object]]:
    ax, ay = center(anchor.col, anchor.row)
    ranked: list[tuple[float, _Object]] = []
    for obj in objects:
        if obj.col == anchor.col and obj.row == anchor.row:
            continue
        ox, oy = center(obj.col, obj.row)
        if ox < ax:
            ranked.append((distance(ax, ay, ox, oy), obj))
    ranked.sort(key=lambda item: (item[0], item[1].col, item[1].row, item[1].object_id))
    return ranked


def spatial_answer(question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    """Color of the adjacent same-row object to the left, when that reading is unique.

    Returns ``None`` if the anchor is not unique, the adjacent cell is not a
    single object, anything else sits further left on that row, anything else
    sits in the adjacent column, the nearest left-half object is not that
    neighbor, or another left-half object is closer than twice the neighbor
    distance.
    """
    match = SPATIAL_RE.fullmatch(question)
    if match is None:
        return None
    size, shape = match.group(1), match.group(2)
    anchors = [obj for obj in objects if obj.size == size and obj.shape == shape]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    adjacent = [obj for obj in objects if obj.row == anchor.row and obj.col == anchor.col - 1]
    further = [obj for obj in objects if obj.row == anchor.row and obj.col < anchor.col - 1]
    mates = [obj for obj in objects if obj.col == anchor.col - 1 and obj.row != anchor.row]
    if len(adjacent) != 1 or further or mates:
        return None
    ranked = _left_ranked(anchor, objects)
    if not ranked:
        return None
    neighbor = adjacent[0]
    if (ranked[0][1].col, ranked[0][1].row) != (neighbor.col, neighbor.row):
        return None
    neighbor_distance = ranked[0][0]
    if any(dist < MIN_LEFT_RATIO * neighbor_distance for dist, _obj in ranked[1:]):
        return None
    return neighbor.color


def answer_for(
    task_family: str,
    question: str,
    objects: list[_Object] | tuple[_Object, ...],
) -> str | None:
    if task_family == "attribute":
        return attribute_answer(question, objects)
    if task_family == "counting":
        return counting_answer(question, objects)
    if task_family == "spatial":
        return spatial_answer(question, objects)
    raise ValueError(f"unknown task family: {task_family}")


def count_word(count: int) -> str:
    try:
        return COUNT_WORDS[count]
    except KeyError as exc:
        raise ValueError(f"no count word for {count}") from exc


def numeric_answer(answer: str) -> int:
    try:
        return WORD_TO_COUNT[answer]
    except KeyError as exc:
        raise ValueError(f"not a count word: {answer}") from exc
