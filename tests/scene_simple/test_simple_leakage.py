"""Answer leakage. Non-revealing text must fail a substring test."""

from __future__ import annotations

import re

from scene_gen.contexts import (
    ATTRIBUTE_HELPFUL,
    COUNTING_HELPFUL,
    NEUTRAL_TEXT as V1_NEUTRAL,
    SPATIAL_HELPFUL,
)
from scene_gen.normalize import normalize_for_leakage as v1_normalize

from scene_simple.contexts import HELPFUL_TEXT, NEUTRAL_TEXT
from scene_simple.normalize import normalize_for_leakage

NON_REVEALING = ("neutral", "helpful_nonrevealing", "contradictory")
LABEL_WORDS = ("red", "blue", "green", "yellow", "one", "two", "three", "four")
COLOR_WORDS = ("red", "blue", "green", "yellow")


def test_normalization_matches_the_checkpoint1_function() -> None:
    samples = [
        "  The LARGE circle is RED!!! ",
        "none",
        "item 2",
        NEUTRAL_TEXT,
        HELPFUL_TEXT,
        "The circle in the top row is green.",
    ]
    for sample in samples:
        assert normalize_for_leakage(sample) == v1_normalize(sample)
    assert "one" in normalize_for_leakage("none")


def test_neutral_is_the_checkpoint1_sentence_and_helpful_is_not() -> None:
    assert NEUTRAL_TEXT == V1_NEUTRAL
    assert HELPFUL_TEXT not in {ATTRIBUTE_HELPFUL, COUNTING_HELPFUL, SPATIAL_HELPFUL}
    assert "Identify" not in HELPFUL_TEXT
    folded = HELPFUL_TEXT.casefold()
    for banned in ("top", "bottom", "left", "right", "large", "small", "count", "size"):
        assert banned not in folded
    for word in LABEL_WORDS:
        assert word not in normalize_for_leakage(HELPFUL_TEXT)


def test_records_use_the_row_color_strings(records: list[dict]) -> None:
    for record in records:
        assert record["contexts"]["neutral"]["text"] == NEUTRAL_TEXT
        assert record["contexts"]["helpful_nonrevealing"]["text"] == HELPFUL_TEXT


def test_nonrevealing_channels_and_question_omit_the_answer(records: list[dict]) -> None:
    for record in records:
        aliases = [normalize_for_leakage(alias) for alias in record["answer_aliases"]]
        assert aliases == [normalize_for_leakage(record["answer"])]
        pieces = [record["question"], *(record["contexts"][key]["text"] for key in NON_REVEALING)]
        for piece in pieces:
            norm = normalize_for_leakage(piece)
            for alias in aliases:
                assert alias not in norm, (record["scene_id"], alias, piece)
        for key in NON_REVEALING:
            joined = record["contexts"][key]["text"] + "\n" + record["question"]
            joined_norm = normalize_for_leakage(joined)
            for alias in aliases:
                assert alias not in joined_norm, (record["scene_id"], key)


def test_neutral_and_helpful_contain_no_label_vocabulary(records: list[dict]) -> None:
    for record in records:
        for key in ("neutral", "helpful_nonrevealing"):
            norm = normalize_for_leakage(record["contexts"][key]["text"])
            for word in LABEL_WORDS:
                assert word not in norm, (record["scene_id"], key, word)
            assert not re.search(r"\d", norm)


def test_question_names_the_relation_and_not_a_color(records: list[dict]) -> None:
    pattern = re.compile(r"^What color is the (circle|square|triangle) in the (top|bottom) row\?$")
    for record in records:
        match = pattern.fullmatch(record["question"])
        assert match is not None, record["question"]
        norm = normalize_for_leakage(record["question"])
        for color in COLOR_WORDS:
            assert color not in norm
        assert "left" not in record["question"]
        assert "how many" not in record["question"].casefold()


def test_only_answer_revealing_is_marked_and_it_contains_the_answer(records: list[dict]) -> None:
    for record in records:
        flags = {key: value["reveals_answer"] for key, value in record["contexts"].items()}
        assert flags == {
            "neutral": False,
            "helpful_nonrevealing": False,
            "contradictory": False,
            "answer_revealing": True,
        }
        revealing = normalize_for_leakage(record["contexts"]["answer_revealing"]["text"])
        assert normalize_for_leakage(record["answer"]) in revealing
        contradictory = normalize_for_leakage(record["contexts"]["contradictory"]["text"])
        assert normalize_for_leakage(record["answer"]) not in contradictory


def test_contradictory_text_states_a_false_color(records: list[dict]) -> None:
    pattern = re.compile(
        r"^The (circle|square|triangle) in the (top|bottom) row is (red|blue|green|yellow)\.$"
    )
    for record in records:
        text = record["contexts"]["contradictory"]["text"]
        match = pattern.fullmatch(text)
        assert match is not None, text
        assert match.group(3) != record["answer"]
        revealing = record["contexts"]["answer_revealing"]["text"]
        assert revealing.endswith(f" is {record['answer']}.")
        assert revealing != text
