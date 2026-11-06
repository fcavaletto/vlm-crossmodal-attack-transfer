"""Ground truth read off the scene graph.

The question templates below are the only ones this checkpoint emits.
``answer_for`` returns the canonical answer word, or ``None`` when the scene
does not determine exactly one answer.
"""

from __future__ import annotations

import re
from typing import Protocol

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
    r"^What color is the object immediately left of the (large|small) (circle|square|triangle)\?$"
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


def spatial_answer(question: str, objects: list[_Object] | tuple[_Object, ...]) -> str | None:
    """Color of the unique object one column toward the left edge, same row.

    Any other object on that row in a column further left makes the relation
    ambiguous for this checker, so the function returns ``None``.
    """
    match = SPATIAL_RE.fullmatch(question)
    if match is None:
        return None
    size, shape = match.group(1), match.group(2)
    anchors = [obj for obj in objects if obj.size == size and obj.shape == shape]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    adjacent = [
        obj for obj in objects if obj.row == anchor.row and obj.col == anchor.col - 1
    ]
    further = [obj for obj in objects if obj.row == anchor.row and obj.col < anchor.col - 1]
    if len(adjacent) != 1 or further:
        return None
    return adjacent[0].color


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


def queried_color(question: str) -> str | None:
    match = COUNTING_RE.fullmatch(question)
    if match is None:
        return None
    return match.group(1)


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
