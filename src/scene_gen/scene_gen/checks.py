"""Ambiguity flags. An empty list means every check below passed.

The generator refuses to write a scene whose flag list is not empty. Tests
still call this function on broken records, so a silent ``return []`` fails
those tests.
"""

from __future__ import annotations

from typing import Any

from .geometry import CANVAS_H, CANVAS_W, bounding_box, boxes_too_close, center
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
        if SPATIAL_RE.fullmatch(question) is None:
            flags.append("unparsed_question")
            resolved = None
        else:
            size_shape = SPATIAL_RE.fullmatch(question)
            assert size_shape is not None
            size, shape = size_shape.group(1), size_shape.group(2)
            anchors = [obj for obj in views if obj.size == size and obj.shape == shape]
            if len(anchors) != 1:
                flags.append("non_unique_referent")
                resolved = None
            else:
                anchor = anchors[0]
                adjacent = [
                    obj
                    for obj in views
                    if obj.row == anchor.row and obj.col == anchor.col - 1
                ]
                further = [
                    obj for obj in views if obj.row == anchor.row and obj.col < anchor.col - 1
                ]
                if len(adjacent) == 0:
                    flags.append("missing_relation_target")
                if further:
                    flags.append("extra_object_on_relation_ray")
                resolved = spatial_answer(question, views)
    else:
        flags.append("unparsed_question")
        resolved = None

    if resolved != record.get("answer"):
        flags.append("answer_disagrees_with_scene")

    return sorted(set(flags))
