"""Matched pairs: same question, one local two-object recolor, two answers."""

from __future__ import annotations

import re
from collections import defaultdict

from PIL import Image

from scene_v4.vocabulary import RADIUS
from v4_graph_oracle import graph_answer

# Typed from the design table. Not imported from the generator.
_PAIRS = (
    ("red", "blue"),
    ("red", "green"),
    ("red", "blue"),
    ("red", "blue"),
    ("red", "green"),
    ("red", "blue"),
    ("blue", "red"),
    ("green", "red"),
    ("blue", "red"),
    ("blue", "red"),
    ("green", "red"),
    ("blue", "red"),
    ("blue", "green"),
    ("green", "blue"),
    ("blue", "yellow"),
    ("blue", "green"),
    ("green", "blue"),
    ("blue", "yellow"),
    ("green", "blue"),
    ("blue", "green"),
    ("yellow", "blue"),
    ("green", "blue"),
    ("blue", "green"),
    ("yellow", "blue"),
    ("green", "yellow"),
    ("blue", "yellow"),
    ("yellow", "green"),
    ("green", "yellow"),
    ("blue", "yellow"),
    ("yellow", "green"),
    ("yellow", "green"),
    ("yellow", "blue"),
    ("green", "yellow"),
    ("yellow", "green"),
    ("yellow", "blue"),
    ("green", "yellow"),
    ("yellow", "red"),
    ("yellow", "red"),
    ("green", "red"),
    ("yellow", "red"),
    ("yellow", "red"),
    ("green", "red"),
    ("red", "yellow"),
    ("red", "yellow"),
    ("red", "green"),
    ("red", "yellow"),
    ("red", "yellow"),
    ("red", "green"),
)


def _expected() -> dict[str, str]:
    table = {}
    for index, (arm_a, arm_b) in enumerate(_PAIRS):
        table[f"dev4-ref-{index:02d}-a"] = arm_a
        table[f"dev4-ref-{index:02d}-b"] = arm_b
    return table


EXPECTED_ANSWERS = _expected()
_QUESTION = re.compile(
    r"^What color is the (large|small) (circle|square|triangle) in the (top|bottom) row\?$"
)
_ROW = {"top": 0, "bottom": 1}


def test_hand_table_and_independent_reader_agree(records: list[dict]) -> None:
    found = {record["scene_id"]: record["answer"] for record in records}
    assert found == EXPECTED_ANSWERS
    assert len(EXPECTED_ANSWERS) == 96
    for record in records:
        assert graph_answer(record["question"], record["objects"]) == record["answer"]
        assert "contexts" not in graph_answer.__code__.co_varnames


def _by_id(record: dict) -> dict[str, dict]:
    return {obj["object_id"]: obj for obj in record["objects"]}


def _groups(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["counterfactual_group_id"]].append(record)
    for members in grouped.values():
        members.sort(key=lambda record: record["scene_id"])
    return grouped


def _nonrevealing(record: dict) -> tuple[str, ...]:
    return (
        record["question"],
        record["contexts"]["neutral"]["text"],
        record["contexts"]["helpful_nonrevealing"]["text"],
        record["contexts"]["contradictory"]["text"],
    )


def test_pairs_share_text_and_differ_in_answer(records: list[dict]) -> None:
    grouped = _groups(records)
    assert len(grouped) == 48
    for members in grouped.values():
        left, right = members
        assert left["answer"] != right["answer"]
        assert left["task_family"] == right["task_family"] == "referent_color"
        assert _nonrevealing(left) == _nonrevealing(right)
        assert left["contexts"]["answer_revealing"]["text"] != right["contexts"]["answer_revealing"]["text"]


def test_edits_swap_the_referent_with_a_different_shape(records: list[dict]) -> None:
    for members in _groups(records).values():
        left, right = _by_id(members[0]), _by_id(members[1])
        assert list(left) == [f"o{number}" for number in range(8)]
        changed = []
        for object_id in left:
            before, after = left[object_id], right[object_id]
            assert before["position"] == after["position"]
            assert before["category"] == after["category"]
            assert before["attributes"]["size"] == after["attributes"]["size"]
            if before["attributes"]["color"] != after["attributes"]["color"]:
                changed.append(object_id)
        assert changed == ["o0", "o4"]
        assert left["o0"]["attributes"]["color"] == right["o4"]["attributes"]["color"]
        assert left["o4"]["attributes"]["color"] == right["o0"]["attributes"]["color"]
        assert left["o4"]["category"] != left["o0"]["category"]


def test_contradictory_color_is_the_other_row_copy_and_is_wrong(records: list[dict]) -> None:
    """Copying the contradictory sentence is a wrong answer. The picture path still works."""
    for record in records:
        match = _QUESTION.fullmatch(record["question"])
        assert match is not None
        size, shape, row_name = match.group(1), match.group(2), _ROW[match.group(3)]
        stated = record["contexts"]["contradictory"]["text"].rsplit(" ", 1)[-1].rstrip(".")
        assert stated != record["answer"]
        decoys = [
            obj
            for obj in record["objects"]
            if obj["attributes"]["size"] == size
            and obj["category"] == shape
            and obj["position"]["row"] != row_name
        ]
        assert len(decoys) == 1
        assert decoys[0]["attributes"]["color"] == stated
        assert decoys[0]["position"]["col"] != next(
            obj["position"]["col"] for obj in record["objects"] if obj["object_id"] == "o0"
        )
        assert graph_answer(record["question"], record["objects"]) == record["answer"]


def test_counterfactual_pixels_stay_inside_the_edited_boxes(records: list[dict], generated) -> None:
    for members in _groups(records).values():
        image_a = Image.open(generated / members[0]["image_path"])
        image_b = Image.open(generated / members[1]["image_path"])
        assert image_a.size == image_b.size == (512, 384)
        pixels_a = list(image_a.getdata())
        pixels_b = list(image_b.getdata())
        changed = sum(pixel_a != pixel_b for pixel_a, pixel_b in zip(pixels_a, pixels_b, strict=True))
        fraction = changed / len(pixels_a)
        assert 200 <= changed and fraction < 0.25, (members[0]["scene_id"], changed, fraction)
        edited = [
            obj
            for obj in members[0]["objects"]
            if _by_id(members[1])[obj["object_id"]]["attributes"]["color"] != obj["attributes"]["color"]
        ]
        assert [obj["object_id"] for obj in edited] == ["o0", "o4"]
        boxes = []
        for obj in edited:
            pad = RADIUS[obj["attributes"]["size"]] + 4
            cx, cy = obj["position"]["cx"], obj["position"]["cy"]
            boxes.append((cx - pad, cy - pad, cx + pad, cy + pad))
        width = image_a.size[0]
        for index, (pixel_a, pixel_b) in enumerate(zip(pixels_a, pixels_b, strict=True)):
            if pixel_a == pixel_b:
                continue
            px, py = index % width, index // width
            assert any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in boxes), (
                members[0]["scene_id"],
                px,
                py,
            )
