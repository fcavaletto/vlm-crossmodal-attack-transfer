"""Shortcut checks. A passing test is a failed shortcut, computed from the scenes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from PIL import Image

_SPATIAL = re.compile(
    r"^What color is the object immediately left of the (large|small) (circle|square|triangle)\?$"
)


def _groups(records: list[dict], family: str | None = None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if family is not None and record["task_family"] != family:
            continue
        grouped[record["counterfactual_group_id"]].append(record)
    for members in grouped.values():
        members.sort(key=lambda record: record["scene_id"])
    return grouped


def _nonrevealing_bundle(record: dict) -> tuple[str, ...]:
    return (
        record["question"],
        record["contexts"]["neutral"]["text"],
        record["contexts"]["helpful_nonrevealing"]["text"],
        record["contexts"]["contradictory"]["text"],
    )


def test_shared_text_does_not_identify_the_arm(records: list[dict]) -> None:
    for members in _groups(records).values():
        assert len(members) == 2
        left, right = members
        assert left["answer"] != right["answer"]
        assert _nonrevealing_bundle(left) == _nonrevealing_bundle(right)
        assert (
            left["contexts"]["answer_revealing"]["text"]
            != right["contexts"]["answer_revealing"]["text"]
        )


def test_each_nonrevealing_string_occurs_with_at_least_two_answers(records: list[dict]) -> None:
    for key in ("neutral", "helpful_nonrevealing", "contradictory"):
        by_text: dict[str, set[str]] = defaultdict(set)
        for record in records:
            by_text[record["contexts"][key]["text"]].add(record["answer"])
        for text, answers in by_text.items():
            assert len(answers) >= 2, (key, text, answers)


def test_attribute_target_color_is_not_unique(records: list[dict]) -> None:
    for record in records:
        if record["task_family"] != "attribute":
            continue
        count = sum(
            1 for obj in record["objects"] if obj["attributes"]["color"] == record["answer"]
        )
        assert count >= 2, record["scene_id"]


def test_no_fixed_cell_predicts_attribute_answers(records: list[dict]) -> None:
    attribute = [record for record in records if record["task_family"] == "attribute"]
    for col in range(4):
        for row in range(2):
            occupied = 0
            matches = 0
            for record in attribute:
                here = [
                    obj
                    for obj in record["objects"]
                    if obj["position"]["col"] == col and obj["position"]["row"] == row
                ]
                if not here:
                    continue
                occupied += 1
                if here[0]["attributes"]["color"] == record["answer"]:
                    matches += 1
            if occupied >= 2:
                assert matches < occupied, (col, row, matches, occupied)


def test_counting_histograms_do_not_encode_the_answer(records: list[dict]) -> None:
    counting = [record for record in records if record["task_family"] == "counting"]
    numeric = [{"two": 2, "three": 3}[record["answer"]] for record in counting]

    def color_hist(record: dict) -> tuple[int, ...]:
        counts = Counter(obj["attributes"]["color"] for obj in record["objects"])
        return tuple(sorted(counts.values()))

    for members in _groups(counting).values():
        assert color_hist(members[0]) == color_hist(members[1]) == (2, 3, 3)
        queried = re.fullmatch(
            r"How many (red|blue|green|yellow) objects are there\?",
            members[0]["question"],
        )
        assert queried is not None
        color = queried.group(1)

        def count_color(record: dict, name: str = color) -> int:
            return sum(1 for obj in record["objects"] if obj["attributes"]["color"] == name)

        assert count_color(members[0]) == 2
        assert count_color(members[1]) == 3
        changed = []
        for name in ("red", "blue", "green", "yellow"):
            before = count_color(members[0], name)
            after = count_color(members[1], name)
            if before != after:
                changed.append((name, after - before))
        assert len(changed) == 2
        assert dict(changed)[color] == 1
        assert set(dict(changed).values()) == {1, -1}

    stats: dict[str, list[int]] = {
        "total": [len(record["objects"]) for record in counting],
        "max": [],
        "second": [],
        "min": [],
    }
    for color in ("red", "blue", "green", "yellow"):
        stats[color] = [
            sum(1 for obj in record["objects"] if obj["attributes"]["color"] == color)
            for record in counting
        ]
    for shape in ("circle", "square", "triangle"):
        stats[shape] = [
            sum(1 for obj in record["objects"] if obj["category"] == shape) for record in counting
        ]
    for size in ("small", "large"):
        stats[size] = [
            sum(1 for obj in record["objects"] if obj["attributes"]["size"] == size)
            for record in counting
        ]
    for record in counting:
        values = sorted(
            Counter(obj["attributes"]["color"] for obj in record["objects"]).values(),
            reverse=True,
        )
        stats["max"].append(values[0])
        stats["second"].append(values[1])
        stats["min"].append(values[-1])
    for name, values in stats.items():
        assert values != numeric, name


def test_spatial_relation_is_not_the_leftmost_object(records: list[dict]) -> None:
    spatial = [record for record in records if record["task_family"] == "spatial"]
    anchor_cols = set()
    anchor_rows = set()
    for record in spatial:
        objects = record["objects"]
        min_col = min(obj["position"]["col"] for obj in objects)
        leftmost = [obj for obj in objects if obj["position"]["col"] == min_col]
        assert len(leftmost) == 1
        assert leftmost[0]["attributes"]["color"] != record["answer"]
        match = _SPATIAL.fullmatch(record["question"])
        assert match is not None
        size, shape = match.group(1), match.group(2)
        anchors = [
            obj
            for obj in objects
            if obj["attributes"]["size"] == size and obj["category"] == shape
        ]
        assert len(anchors) == 1
        anchor = anchors[0]
        anchor_cols.add(anchor["position"]["col"])
        anchor_rows.add(anchor["position"]["row"])
        further = [
            obj
            for obj in objects
            if obj["position"]["row"] == anchor["position"]["row"]
            and obj["position"]["col"] < anchor["position"]["col"] - 1
        ]
        assert further == []
        same_column = [
            obj for obj in objects if obj["position"]["col"] == anchor["position"]["col"] - 1
        ]
        assert len(same_column) >= 2
        others = [
            obj for obj in same_column if obj["position"]["row"] != anchor["position"]["row"]
        ]
        assert others
        assert all(obj["attributes"]["color"] != record["answer"] for obj in others)
        singles = [
            color
            for color, count in Counter(
                obj["attributes"]["color"] for obj in objects
            ).items()
            if count == 1
        ]
        assert set(singles) != {record["answer"]}
        assert len(singles) != 1
    assert anchor_cols == {2, 3}
    assert anchor_rows == {0, 1}


def test_counterfactual_images_differ_in_a_small_region(records: list[dict], generated) -> None:
    for members in _groups(records).values():
        left = Image.open(generated / members[0]["image_path"])
        right = Image.open(generated / members[1]["image_path"])
        pixels_a = list(left.getdata())
        pixels_b = list(right.getdata())
        changed = sum(pixel_a != pixel_b for pixel_a, pixel_b in zip(pixels_a, pixels_b))
        fraction = changed / len(pixels_a)
        # The smallest edit recolors one small triangle (a few hundred pixels).
        # A full redraw would move a large fraction of the canvas.
        assert 200 <= changed and fraction < 0.25, (members[0]["scene_id"], changed, fraction)
