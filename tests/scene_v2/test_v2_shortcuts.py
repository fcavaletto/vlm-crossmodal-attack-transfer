"""Shortcut checks. A passing test is a failed shortcut, computed from the scenes."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_COUNT = {"one": 1, "two": 2, "three": 3, "four": 4}
_SPATIAL = re.compile(
    r"^What color is the object immediately left of the "
    r"(large|small) (circle|square|triangle) in the same row\?$"
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


def test_each_nonrevealing_string_occurs_with_at_least_two_answers(records: list[dict]) -> None:
    for key in ("neutral", "helpful_nonrevealing", "contradictory"):
        by_text: dict[str, set[str]] = defaultdict(set)
        for record in records:
            by_text[record["contexts"][key]["text"]].add(record["answer"])
        for text, answers in by_text.items():
            assert len(answers) >= 2, (key, text, answers)


def test_attribute_words_and_color_are_not_sufficient_alone(records: list[dict]) -> None:
    for record in records:
        if record["task_family"] != "attribute":
            continue
        match = re.fullmatch(r"What color is the (large|small) (circle|square|triangle)\?", record["question"])
        assert match is not None
        size, shape = match.group(1), match.group(2)
        assert sum(1 for obj in record["objects"] if obj["attributes"]["color"] == record["answer"]) >= 2
        assert sum(1 for obj in record["objects"] if obj["category"] == shape) >= 2
        assert sum(1 for obj in record["objects"] if obj["attributes"]["size"] == size) >= 2
        assert (
            sum(
                1
                for obj in record["objects"]
                if obj["category"] == shape and obj["attributes"]["size"] == size
            )
            == 1
        )


def test_no_fixed_cell_predicts_attribute_answers(records: list[dict]) -> None:
    attribute = [record for record in records if record["task_family"] == "attribute"]
    for col in range(4):
        for row in range(3):
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


def test_counting_statistics_do_not_encode_the_answer(records: list[dict]) -> None:
    counting = [record for record in records if record["task_family"] == "counting"]
    numeric = [_COUNT[record["answer"]] for record in counting]
    assert set(numeric) == {1, 2, 3, 4}

    def count_color(record: dict, name: str) -> int:
        return sum(1 for obj in record["objects"] if obj["attributes"]["color"] == name)

    histograms = []
    for members in _groups(counting).values():
        def histogram(record: dict) -> tuple[int, ...]:
            return tuple(sorted(Counter(obj["attributes"]["color"] for obj in record["objects"]).values()))

        assert histogram(members[0]) == histogram(members[1])
        histograms.append(histogram(members[0]))
        query = members[0]["question"].split()[2]
        assert count_color(members[0], query) + 1 == count_color(members[1], query)
    assert len(set(histograms)) > 1

    stats: dict[str, list[int]] = {
        "total": [len(record["objects"]) for record in counting],
        "max": [],
        "second": [],
        "min": [],
    }
    for color in ("red", "blue", "green", "yellow"):
        stats[color] = [count_color(record, color) for record in counting]
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
        values = sorted(Counter(obj["attributes"]["color"] for obj in record["objects"]).values(), reverse=True)
        stats["max"].append(values[0])
        stats["second"].append(values[1])
        stats["min"].append(values[-1])
    for name, values in stats.items():
        assert values != numeric, name
        if name in {"total", "circle", "square", "triangle", "small", "large"}:
            assert len(set(values)) > 1, name


def test_spatial_shortcuts_disagree_with_the_relation(records: list[dict]) -> None:
    spatial = [record for record in records if record["task_family"] == "spatial"]
    anchor_cols = set()
    anchor_rows = set()
    ratios = []
    for record in spatial:
        objects = record["objects"]
        leftmost = [obj for obj in objects if obj["position"]["col"] == 0]
        assert len(leftmost) == 1
        assert leftmost[0]["attributes"]["color"] != record["answer"]
        counts = Counter(obj["attributes"]["color"] for obj in objects)
        assert all(count >= 2 for count in counts.values())
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
        mates = [
            obj
            for obj in objects
            if obj["position"]["col"] == anchor["position"]["col"] - 1
            and obj["position"]["row"] != anchor["position"]["row"]
        ]
        assert mates == []
        ax, ay = anchor["position"]["cx"], anchor["position"]["cy"]
        left = []
        for obj in objects:
            if obj["object_id"] == anchor["object_id"]:
                continue
            ox, oy = obj["position"]["cx"], obj["position"]["cy"]
            if ox < ax:
                left.append((math.hypot(ax - ox, ay - oy), obj))
        left.sort(key=lambda item: item[0])
        assert left[0][1]["attributes"]["color"] == record["answer"]
        ratio = left[1][0] / left[0][0]
        assert ratio >= 2
        ratios.append(ratio)
    assert anchor_cols == {2, 3}
    assert anchor_rows == {0, 1, 2}
    assert min(ratios) >= 2
