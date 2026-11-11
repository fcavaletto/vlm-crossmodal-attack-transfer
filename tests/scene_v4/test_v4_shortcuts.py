"""Shortcut checks. A passing test is a failed shortcut, read off the scenes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from scene_v4.vocabulary import BACKGROUND

_QUESTION = re.compile(
    r"^What color is the (large|small) (circle|square|triangle) in the (top|bottom) row\?$"
)
_ROW = {"top": 0, "bottom": 1}


def _parsed(record: dict) -> tuple[str, str, int]:
    match = _QUESTION.fullmatch(record["question"])
    assert match is not None
    return match.group(1), match.group(2), _ROW[match.group(3)]


def _groups(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
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
            assert len(answers) >= 2, (key, answers)
    by_question: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_question[record["question"]].add(record["answer"])
    for question, answers in by_question.items():
        assert answers == {"red", "blue", "green", "yellow"}, (question, answers)


def test_size_shape_and_row_are_each_necessary(records: list[dict]) -> None:
    for record in records:
        size, shape, row = _parsed(record)
        answer = record["answer"]
        objects = record["objects"]
        full = [
            obj
            for obj in objects
            if obj["attributes"]["size"] == size
            and obj["category"] == shape
            and obj["position"]["row"] == row
        ]
        assert len(full) == 1
        assert full[0]["attributes"]["color"] == answer
        without_size = [
            obj for obj in objects if obj["category"] == shape and obj["position"]["row"] == row
        ]
        without_shape = [
            obj
            for obj in objects
            if obj["attributes"]["size"] == size and obj["position"]["row"] == row
        ]
        without_row = [
            obj for obj in objects if obj["attributes"]["size"] == size and obj["category"] == shape
        ]
        for broadened in (without_size, without_shape, without_row):
            assert len(broadened) > 1
            assert {obj["attributes"]["color"] for obj in broadened} != {answer}


def test_no_fixed_cell_predicts_the_answer(records: list[dict]) -> None:
    for col in range(4):
        for row in range(2):
            occupied = 0
            matches = 0
            for record in records:
                here = [
                    obj
                    for obj in record["objects"]
                    if obj["position"]["col"] == col and obj["position"]["row"] == row
                ]
                assert len(here) == 1
                occupied += 1
                if here[0]["attributes"]["color"] == record["answer"]:
                    matches += 1
            assert matches < occupied, (col, row, matches, occupied)


def test_mode_and_singleton_are_never_the_answer(records: list[dict]) -> None:
    """Both arms show counts 3, 2, 2, 1. The answer is one of the two doubles."""
    for record in records:
        counts = Counter(obj["attributes"]["color"] for obj in record["objects"])
        assert sorted(counts.values()) == [1, 2, 2, 3]
        assert counts[record["answer"]] == 2
        mode = [color for color, count in counts.items() if count == 3]
        single = [color for color, count in counts.items() if count == 1]
        assert mode != [record["answer"]]
        assert single != [record["answer"]]
        doubles = {color for color, count in counts.items() if count == 2}
        assert record["answer"] in doubles
        assert len(doubles) == 2


def test_arm_invariant_counts_cannot_name_both_answers(records: list[dict]) -> None:
    for members in _groups(records).values():
        left, right = members
        assert left["answer"] != right["answer"]

        def profile(record: dict) -> tuple[object, ...]:
            colors = Counter(obj["attributes"]["color"] for obj in record["objects"])
            shapes = Counter(obj["category"] for obj in record["objects"])
            sizes = Counter(obj["attributes"]["size"] for obj in record["objects"])
            return (
                len(record["objects"]),
                tuple(sorted(colors.items())),
                tuple(sorted(shapes.items())),
                tuple(sorted(sizes.items())),
            )

        assert profile(left) == profile(right)


def test_object_count_does_not_vary_with_the_answer(records: list[dict]) -> None:
    assert Counter(len(record["objects"]) for record in records) == {8: 96}
    assert set(record["answer"] for record in records) == {"red", "blue", "green", "yellow"}


def test_referent_is_not_stacked_on_its_copy(records: list[dict]) -> None:
    for record in records:
        _size, _shape, _row = _parsed(record)
        target = next(obj for obj in record["objects"] if obj["object_id"] == "o0")
        foil = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
        assert target["position"]["row"] != foil["position"]["row"]
        assert target["position"]["col"] != foil["position"]["col"]


def test_leftmost_object_in_the_named_row_is_not_always_the_answer(records: list[dict]) -> None:
    matches = 0
    for record in records:
        _size, _shape, row = _parsed(record)
        same_row = [obj for obj in record["objects"] if obj["position"]["row"] == row]
        leftmost = min(same_row, key=lambda obj: obj["position"]["col"])
        if leftmost["attributes"]["color"] == record["answer"]:
            matches += 1
    assert matches < len(records)


def test_rows_are_separated_by_an_empty_band(records: list[dict], generated) -> None:
    from PIL import Image

    for record in records:
        image = Image.open(generated / record["image_path"])
        for y in range(164, 224):
            for x in range(image.size[0]):
                assert image.getpixel((x, y)) == BACKGROUND, (record["scene_id"], x, y)
        assert image.getpixel((0, 0)) == BACKGROUND
        assert image.getpixel((511, 383)) == BACKGROUND
