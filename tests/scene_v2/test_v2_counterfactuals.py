"""Matched pairs: same question, one controlled edit, two answers."""

from __future__ import annotations

from collections import defaultdict

from PIL import Image

from v2_graph_oracle import graph_answer

# Typed from the design table. Not imported from the generator.
_ATTRIBUTE = [
    ("red", "blue"),
    ("green", "yellow"),
    ("yellow", "red"),
    ("blue", "green"),
    ("red", "green"),
    ("blue", "yellow"),
    ("green", "red"),
    ("yellow", "blue"),
    ("red", "yellow"),
    ("green", "blue"),
    ("yellow", "green"),
    ("blue", "red"),
]
_COUNTING = [("one", "two")] * 4 + [("two", "three")] * 4 + [("three", "four")] * 4
_SPATIAL = [
    ("red", "blue"),
    ("yellow", "green"),
    ("blue", "yellow"),
    ("red", "green"),
    ("green", "red"),
    ("yellow", "blue"),
    ("blue", "red"),
    ("green", "yellow"),
    ("yellow", "red"),
    ("blue", "green"),
    ("red", "yellow"),
    ("green", "blue"),
]


def _expected() -> dict[str, str]:
    table: dict[str, str] = {}
    for family, pairs in (
        ("attribute", _ATTRIBUTE),
        ("counting", _COUNTING),
        ("spatial", _SPATIAL),
    ):
        for index, (arm_a, arm_b) in enumerate(pairs):
            table[f"dev2-{family}-{index:02d}-a"] = arm_a
            table[f"dev2-{family}-{index:02d}-b"] = arm_b
    return table


EXPECTED_ANSWERS = _expected()


def test_hand_table_and_independent_reader_agree(records: list[dict]) -> None:
    found = {record["scene_id"]: record["answer"] for record in records}
    assert found == EXPECTED_ANSWERS
    assert len(EXPECTED_ANSWERS) == 72
    for record in records:
        assert graph_answer(record) == record["answer"]


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
    assert len(grouped) == 36
    for members in grouped.values():
        left, right = members
        assert left["answer"] != right["answer"]
        assert left["task_family"] == right["task_family"]
        assert _nonrevealing(left) == _nonrevealing(right)
        assert left["contexts"]["answer_revealing"]["text"] != right["contexts"]["answer_revealing"]["text"]


def test_edits_recolor_and_do_not_move(records: list[dict]) -> None:
    for members in _groups(records).values():
        left, right = _by_id(members[0]), _by_id(members[1])
        assert left.keys() == right.keys()
        changed = []
        for object_id in left:
            before, after = left[object_id], right[object_id]
            assert before["position"] == after["position"]
            assert before["category"] == after["category"]
            assert before["attributes"]["size"] == after["attributes"]["size"]
            if before["attributes"]["color"] != after["attributes"]["color"]:
                changed.append(object_id)
        family = members[0]["task_family"]
        if family == "attribute":
            assert changed == ["o0"]
        elif family == "counting":
            assert len(changed) == 1
        else:
            assert changed == ["o0", "o1"]
            assert right["o0"]["attributes"]["color"] == left["o1"]["attributes"]["color"]
            assert right["o1"]["attributes"]["color"] == left["o0"]["attributes"]["color"]


def _radius(size: str) -> int:
    return {"small": 22, "large": 46}[size]


def test_counterfactual_pixels_stay_inside_the_edited_boxes(records: list[dict], generated) -> None:
    for members in _groups(records).values():
        image_a = Image.open(generated / members[0]["image_path"])
        image_b = Image.open(generated / members[1]["image_path"])
        pixels_a = list(image_a.getdata())
        pixels_b = list(image_b.getdata())
        changed = sum(pixel_a != pixel_b for pixel_a, pixel_b in zip(pixels_a, pixels_b))
        fraction = changed / len(pixels_a)
        assert 200 <= changed and fraction < 0.25, (members[0]["scene_id"], changed, fraction)
        by_a = _by_id(members[0])
        by_b = _by_id(members[1])
        edited = [
            object_id
            for object_id in by_a
            if by_a[object_id]["attributes"]["color"] != by_b[object_id]["attributes"]["color"]
        ]
        boxes = []
        for object_id in edited:
            obj = by_a[object_id]
            pad = _radius(obj["attributes"]["size"]) + 4
            cx, cy = obj["position"]["cx"], obj["position"]["cy"]
            boxes.append((cx - pad, cy - pad, cx + pad, cy + pad))
        width = image_a.size[0]
        for index, (pixel_a, pixel_b) in enumerate(zip(pixels_a, pixels_b)):
            if pixel_a == pixel_b:
                continue
            px, py = index % width, index // width
            assert any(x0 <= px <= x1 and y0 <= py <= y1 for x0, y0, x1, y1 in boxes)
