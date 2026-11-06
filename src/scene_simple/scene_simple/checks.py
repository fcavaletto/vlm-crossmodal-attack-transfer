"""Ambiguity flags. An empty list means every check below passed.

The generator refuses to write a scene whose flag list is not empty. Tests
call this function on broken records, so a silent ``return []`` fails those
tests.

The row-color checks are about the question that is actually graded. There
is no "immediately left" relation, so there is no column-mate flag. A same
shape in the other row is required, and it must be a different color, so the
row word does work. Another shape in the named row is required, and it must
be a different color, so the shape word does work. The two rows must hold
the same number of objects, so occupancy cannot stand in for the row word.
"""

from __future__ import annotations

from typing import Any

from .geometry import (
    CANVAS_H,
    CANVAS_W,
    MIN_SAME_ROW_COL_GAP,
    bounding_box,
    boxes_too_close,
    center,
)
from .oracle import QUESTION_RE
from .vocabulary import COLORS, ROW_INDEX, SHAPES, SIZES


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
        if size in SIZES:
            box = bounding_box(col, row, size)
            x0, y0, x1, y1 = box
            if x0 < 0 or y0 < 0 or x1 >= CANVAS_W or y1 >= CANVAS_H:
                flags.append("object_outside_canvas")
        cells.append((col, row))

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

    by_row: dict[int, list[int]] = {0: [], 1: []}
    for raw in objects:
        row = raw["position"]["row"]
        if row in by_row:
            by_row[row].append(raw["position"]["col"])
    if len(by_row[0]) != len(by_row[1]):
        flags.append("row_occupancy_names_the_query")
    for cols in by_row.values():
        for index, col_a in enumerate(cols):
            for col_b in cols[index + 1 :]:
                if abs(col_a - col_b) < MIN_SAME_ROW_COL_GAP:
                    flags.append("cramped_same_row")

    question = str(record.get("question", ""))
    if record.get("task_family") != "row_color" or QUESTION_RE.fullmatch(question) is None:
        flags.append("unparsed_question")
        resolved = None
    else:
        resolved = _relation_flags(question, objects, flags)

    if resolved != record.get("answer"):
        flags.append("answer_disagrees_with_scene")
    return sorted(set(flags))


def _relation_flags(question: str, objects: list[dict[str, Any]], flags: list[str]) -> str | None:
    match = QUESTION_RE.fullmatch(question)
    assert match is not None
    shape, row_name = match.group(1), match.group(2)
    row = ROW_INDEX[row_name]
    hits = [
        obj
        for obj in objects
        if obj["category"] == shape and obj["position"]["row"] == row
    ]
    if len(hits) != 1:
        flags.append("non_unique_referent")
    other_row = 1 - row
    shape_decoys = [
        obj
        for obj in objects
        if obj["category"] == shape and obj["position"]["row"] == other_row
    ]
    if len(shape_decoys) != 1:
        flags.append("missing_shape_decoy")
    elif len(hits) == 1 and shape_decoys[0]["attributes"]["color"] == hits[0]["attributes"]["color"]:
        flags.append("shape_decoy_matches_answer")
    row_decoys = [
        obj
        for obj in objects
        if obj["position"]["row"] == row and obj["category"] != shape
    ]
    if not row_decoys:
        flags.append("missing_row_decoy")
    elif len(hits) == 1 and any(
        obj["attributes"]["color"] == hits[0]["attributes"]["color"] for obj in row_decoys
    ):
        flags.append("row_decoy_matches_answer")
    if len(hits) != 1:
        return None
    return hits[0]["attributes"]["color"]
