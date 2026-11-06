"""Shortcut checks. A passing test is a failed shortcut, read off the scenes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

_QUESTION = re.compile(r"^What color is the (circle|square|triangle) in the (top|bottom) row\?$")
_ROW = {"top": 0, "bottom": 1}


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
        assert len(answers) >= 2, (question, answers)


def test_shape_and_row_are_each_necessary(records: list[dict]) -> None:
    for record in records:
        match = _QUESTION.fullmatch(record["question"])
        assert match is not None
        shape, row = match.group(1), _ROW[match.group(2)]
        same_shape = [obj for obj in record["objects"] if obj["category"] == shape]
        assert len(same_shape) == 2
        assert {obj["attributes"]["color"] for obj in same_shape} != {record["answer"]}
        same_row = [obj for obj in record["objects"] if obj["position"]["row"] == row]
        assert len(same_row) == 2
        assert {obj["attributes"]["color"] for obj in same_row} != {record["answer"]}
        hits = [
            obj
            for obj in record["objects"]
            if obj["category"] == shape and obj["position"]["row"] == row
        ]
        assert len(hits) == 1
        assert hits[0]["attributes"]["color"] == record["answer"]
        shapes = {obj["category"] for obj in record["objects"]}
        assert shapes == {"circle", "square", "triangle"}


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
                if not here:
                    continue
                occupied += 1
                if here[0]["attributes"]["color"] == record["answer"]:
                    matches += 1
            if occupied >= 2:
                assert matches < occupied, (col, row, matches, occupied)


def test_each_color_appears_once_so_frequency_does_not_name_it(records: list[dict]) -> None:
    """Both arms show red, blue, green, and yellow once. A count cannot pick the answer."""
    for record in records:
        counts = Counter(obj["attributes"]["color"] for obj in record["objects"])
        assert counts == {"red": 1, "blue": 1, "green": 1, "yellow": 1}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["counterfactual_group_id"]].append(record)
    for members in grouped.values():
        inventories = [
            Counter(obj["attributes"]["color"] for obj in record["objects"]) for record in members
        ]
        assert inventories[0] == inventories[1]


def test_rows_are_separated_by_an_empty_band(records: list[dict], generated) -> None:
    from PIL import Image

    from scene_simple.vocabulary import BACKGROUND

    for record in records:
        image = Image.open(generated / record["image_path"])
        for y in range(164, 224):
            for x in range(image.size[0]):
                assert image.getpixel((x, y)) == BACKGROUND, (record["scene_id"], x, y)
        assert image.getpixel((0, 0)) == BACKGROUND
        assert image.getpixel((511, 383)) == BACKGROUND
