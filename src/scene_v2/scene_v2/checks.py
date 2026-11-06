"""Ambiguity flags. An empty list means every check below passed.

The generator refuses to write a scene whose flag list is not empty. Tests
call this function on broken records, so a silent ``return []`` fails those
tests.

Spatial flags separate two failures that Checkpoint 1 tied together. The
adjacent cell can be filled and still be a bad item if another object shares
that column or sits too close on the left.
"""

from __future__ import annotations

from typing import Any

from .geometry import (
    CANVAS_H,
    CANVAS_W,
    MIN_LEFT_RATIO,
    bounding_box,
    boxes_too_close,
    center,
    distance,
)
from .oracle import (
    ATTRIBUTE_RE,
    COUNTING_RE,
    SPATIAL_RE,
    attribute_answer,
    counting_answer,
    spatial_answer,
)
from .vocabulary import COLORS, COUNT_WORDS, SHAPES, SIZES


def _as_obj(raw: dict[str, Any]) -> Any:
    """Adapt a manifest object dict to the attribute names the oracle reads."""

    class _View:
        object_id = raw["object_id"]
        shape = raw["category"]
        color = raw["attributes"]["color"]
        size = raw["attributes"]["size"]
        col = raw["position"]["col"]
        row = raw["position"]["row"]

    return _View()


def ambiguity_flags(record: dict[str, Any]) -> list[str]:
    """Return sorted flag names for one manifest record."""
    flags: list[str] = []
    objects = record.get("objects") or []
    if not objects:
        flags.append("empty_scene")
        return flags

    ids = [obj["object_id"] for obj in objects]
    if len(ids) != len(set(ids)):
        flags.append("duplicate_object_id")

    cells: list[tuple[int, int]] = []
    views = []
    for raw in objects:
        category = raw["category"]
        color = raw["attributes"]["color"]
        size = raw["attributes"]["size"]
        col = raw["position"]["col"]
        row = raw["position"]["row"]
        if category not in SHAPES or color not in COLORS or size not in SIZES:
            flags.append("unknown_attribute")
        cx = raw["position"]["cx"]
        cy = raw["position"]["cy"]
        try:
            expected = center(col, row)
        except ValueError:
            flags.append("object_outside_canvas")
            continue
        if (cx, cy) != expected:
            flags.append("position_not_derived_from_cell")
        box = bounding_box(col, row, size) if size in SIZES else None
        if box is not None:
            x0, y0, x1, y1 = box
            if x0 < 0 or y0 < 0 or x1 >= CANVAS_W or y1 >= CANVAS_H:
                flags.append("object_outside_canvas")
        cells.append((col, row))
        views.append(_as_obj(raw))

    if len(cells) != len(set(cells)):
        flags.append("shared_cell")

    boxes = []
    for raw in objects:
        size = raw["attributes"]["size"]
        col = raw["position"]["col"]
        row = raw["position"]["row"]
        if size not in SIZES:
            continue
        try:
            boxes.append(bounding_box(col, row, size))
        except ValueError:
            continue
    for index, box_a in enumerate(boxes):
        for box_b in boxes[index + 1 :]:
            if boxes_too_close(box_a, box_b):
                flags.append("overlapping_boxes")
                break
        else:
            continue
        break

    question = record["question"]
    task = record["task_family"]
    resolved: str | None
    if task == "attribute":
        if ATTRIBUTE_RE.fullmatch(question) is None:
            flags.append("unparsed_question")
            resolved = None
        else:
            resolved = attribute_answer(question, views)
            if resolved is None:
                flags.append("non_unique_referent")
    elif task == "counting":
        if COUNTING_RE.fullmatch(question) is None:
            flags.append("unparsed_question")
            resolved = None
        else:
            color = COUNTING_RE.fullmatch(question).group(1)  # type: ignore[union-attr]
            count = sum(1 for obj in views if obj.color == color)
            if count not in COUNT_WORDS:
                flags.append("count_out_of_vocabulary")
                resolved = None
            else:
                resolved = counting_answer(question, views)
    elif task == "spatial":
        resolved = _spatial_flags(question, objects, flags)
    else:
        flags.append("unparsed_question")
        resolved = None

    if resolved != record.get("answer"):
        flags.append("answer_disagrees_with_scene")

    return sorted(set(flags))


def _spatial_flags(question: str, objects: list[dict[str, Any]], flags: list[str]) -> str | None:
    if SPATIAL_RE.fullmatch(question) is None:
        flags.append("unparsed_question")
        return None
    match = SPATIAL_RE.fullmatch(question)
    assert match is not None
    size, shape = match.group(1), match.group(2)
    anchors = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size and obj["category"] == shape
    ]
    if len(anchors) != 1:
        flags.append("non_unique_referent")
        return None
    anchor = anchors[0]
    acol = anchor["position"]["col"]
    arow = anchor["position"]["row"]
    ax = anchor["position"]["cx"]
    ay = anchor["position"]["cy"]
    adjacent = [
        obj
        for obj in objects
        if obj["position"]["row"] == arow and obj["position"]["col"] == acol - 1
    ]
    further = [
        obj
        for obj in objects
        if obj["position"]["row"] == arow and obj["position"]["col"] < acol - 1
    ]
    mates = [
        obj
        for obj in objects
        if obj["position"]["col"] == acol - 1 and obj["position"]["row"] != arow
    ]
    if len(adjacent) == 0:
        flags.append("missing_relation_target")
    if len(adjacent) > 1:
        flags.append("non_unique_referent")
    if further:
        flags.append("extra_object_on_relation_ray")
    if mates:
        flags.append("column_mate_beside_relation")

    left: list[tuple[float, dict[str, Any]]] = []
    for obj in objects:
        if obj["object_id"] == anchor["object_id"]:
            continue
        ox = obj["position"]["cx"]
        oy = obj["position"]["cy"]
        if ox < ax:
            left.append((distance(ax, ay, ox, oy), obj))
    left.sort(key=lambda item: (item[0], item[1]["object_id"]))
    if len(adjacent) == 1 and left:
        neighbor = adjacent[0]
        nearest = left[0][1]
        if nearest["object_id"] != neighbor["object_id"]:
            flags.append("nearest_left_disagrees")
        elif any(dist < MIN_LEFT_RATIO * left[0][0] for dist, _obj in left[1:]):
            flags.append("close_left_competitor")
    elif len(adjacent) == 1 and not left:
        flags.append("nearest_left_disagrees")

    views = [_as_obj(obj) for obj in objects]
    return spatial_answer(question, views)
