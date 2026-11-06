"""Ambiguity flags. An empty list means every check below passed.

The generator refuses to write a scene whose flag list is not empty. Tests
call this function on broken records, so a silent ``return []`` fails those
tests.

The referent is the unique object with the named size, shape, and row. The
same size and shape must also sit in the other row, in another color, or the
row word does no work. The same row must contain another size of that shape,
in another color, or the size word does no work. The same row and size must
contain another shape, in another color, or the shape word does no work.
The two rows hold the same number of objects. A color that is strictly more
common than every other color is not the answer, and neither is a color that
appears once when it is the only such color.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .geometry import (
    CANVAS_H,
    CANVAS_W,
    MIN_LARGE_COL_GAP,
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

    by_row: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for raw in objects:
        row = raw["position"]["row"]
        if row in by_row:
            by_row[row].append(raw)
    if len(by_row[0]) != len(by_row[1]):
        flags.append("unequal_row_occupancy")
    for members in by_row.values():
        large_cols = [
            obj["position"]["col"]
            for obj in members
            if obj["attributes"]["size"] == "large"
        ]
        for index, col_a in enumerate(large_cols):
            for col_b in large_cols[index + 1 :]:
                if abs(col_a - col_b) < MIN_LARGE_COL_GAP:
                    flags.append("adjacent_larges")

    question = str(record.get("question", ""))
    if record.get("task_family") != "referent_color" or QUESTION_RE.fullmatch(question) is None:
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
    size, shape, row_name = match.group(1), match.group(2), match.group(3)
    row = ROW_INDEX[row_name]
    hits = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size
        and obj["category"] == shape
        and obj["position"]["row"] == row
    ]
    if len(hits) != 1:
        flags.append("non_unique_referent")
        referent = None
    else:
        referent = hits[0]["attributes"]["color"]

    other_row = 1 - row
    cross = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size
        and obj["category"] == shape
        and obj["position"]["row"] == other_row
    ]
    if len(cross) == 0:
        flags.append("missing_cross_row_decoy")
    elif len(cross) != 1:
        flags.append("non_unique_cross_row_decoy")
    elif referent is not None and cross[0]["attributes"]["color"] == referent:
        flags.append("cross_row_decoy_matches_answer")

    size_decoys = [
        obj
        for obj in objects
        if obj["category"] == shape
        and obj["position"]["row"] == row
        and obj["attributes"]["size"] != size
    ]
    if not size_decoys:
        flags.append("missing_size_decoy")
    elif referent is not None and all(obj["attributes"]["color"] == referent for obj in size_decoys):
        flags.append("size_decoy_matches_answer")

    row_decoys = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size
        and obj["position"]["row"] == row
        and obj["category"] != shape
    ]
    if not row_decoys:
        flags.append("missing_row_decoy")
    elif referent is not None and all(obj["attributes"]["color"] == referent for obj in row_decoys):
        flags.append("row_decoy_matches_answer")

    if referent is not None:
        _frequency_flags(objects, referent, flags)
    if len(hits) != 1:
        return None
    return referent


def _frequency_flags(objects: list[dict[str, Any]], referent: str, flags: list[str]) -> None:
    counts = Counter(obj["attributes"]["color"] for obj in objects)
    if not counts:
        return
    top = max(counts.values())
    modes = [color for color, count in counts.items() if count == top]
    if len(modes) == 1 and modes[0] == referent:
        flags.append("mode_color_is_answer")
    singles = [color for color, count in counts.items() if count == 1]
    if len(singles) == 1 and singles[0] == referent:
        flags.append("singleton_color_is_answer")
