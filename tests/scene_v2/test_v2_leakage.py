"""Answer leakage. The helpful context must fail a substring test, not a promise."""

from __future__ import annotations

import re

from scene_gen.contexts import (
    ATTRIBUTE_HELPFUL as V1_ATTRIBUTE_HELPFUL,
)
from scene_gen.contexts import (
    COUNTING_HELPFUL as V1_COUNTING_HELPFUL,
)
from scene_gen.contexts import (
    NEUTRAL_TEXT as V1_NEUTRAL_TEXT,
)
from scene_gen.contexts import (
    SPATIAL_HELPFUL as V1_SPATIAL_HELPFUL,
)
from scene_gen.normalize import normalize_for_leakage as v1_normalize

from scene_v2.contexts import (
    ATTRIBUTE_HELPFUL,
    COUNTING_HELPFUL,
    NEUTRAL_TEXT,
    SPATIAL_HELPFUL,
)
from scene_v2.normalize import normalize_for_leakage

NON_REVEALING = ("neutral", "helpful_nonrevealing", "contradictory")
LABEL_WORDS = ("red", "blue", "green", "yellow", "one", "two", "three", "four")
COUNT_WORDS = ("one", "two", "three", "four")


def test_normalization_matches_the_checkpoint1_function() -> None:
    samples = [
        "  The LARGE circle is RED!!! ",
        "none",
        "item 2",
        NEUTRAL_TEXT,
        ATTRIBUTE_HELPFUL,
        COUNTING_HELPFUL,
        SPATIAL_HELPFUL,
        "There is one red object.",
        "There are four blue objects.",
    ]
    for sample in samples:
        assert normalize_for_leakage(sample) == v1_normalize(sample)
    assert "one" in normalize_for_leakage("none")
    assert "2" in normalize_for_leakage("item 2")


def test_prompt_strings_are_the_checkpoint1_strings() -> None:
    assert NEUTRAL_TEXT == V1_NEUTRAL_TEXT
    assert ATTRIBUTE_HELPFUL == V1_ATTRIBUTE_HELPFUL
    assert COUNTING_HELPFUL == V1_COUNTING_HELPFUL
    assert SPATIAL_HELPFUL == V1_SPATIAL_HELPFUL


def test_records_use_those_strings(records: list[dict]) -> None:
    helpful = {
        "attribute": ATTRIBUTE_HELPFUL,
        "counting": COUNTING_HELPFUL,
        "spatial": SPATIAL_HELPFUL,
    }
    for record in records:
        assert record["contexts"]["neutral"]["text"] == NEUTRAL_TEXT
        assert record["contexts"]["helpful_nonrevealing"]["text"] == helpful[record["task_family"]]


def test_nonrevealing_channels_and_question_omit_aliases(records: list[dict]) -> None:
    for record in records:
        aliases = [normalize_for_leakage(alias) for alias in record["answer_aliases"]]
        pieces = [record["question"], *(record["contexts"][key]["text"] for key in NON_REVEALING)]
        for piece in pieces:
            norm = normalize_for_leakage(piece)
            for alias in aliases:
                assert alias not in norm, (record["scene_id"], alias, piece)
        joined = record["contexts"]["helpful_nonrevealing"]["text"] + "\n" + record["question"]
        joined_norm = normalize_for_leakage(joined)
        for alias in aliases:
            assert alias not in joined_norm


def test_neutral_and_helpful_contain_no_label_vocabulary(records: list[dict]) -> None:
    for record in records:
        for key in ("neutral", "helpful_nonrevealing"):
            norm = normalize_for_leakage(record["contexts"][key]["text"])
            for word in LABEL_WORDS:
                assert word not in norm, (record["scene_id"], key, word, norm)
            assert not re.search(r"\d", norm)


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


def test_contradictory_text_states_a_different_label(records: list[dict]) -> None:
    color = re.compile(r"\b(red|blue|green|yellow)\b")
    for record in records:
        text = record["contexts"]["contradictory"]["text"]
        if record["task_family"] == "counting":
            if text.startswith("There is one "):
                assert text.endswith(" object.")
                stated = "one"
            else:
                match = re.fullmatch(r"There are (two|three|four) (red|blue|green|yellow) objects\.", text)
                assert match is not None, text
                stated = match.group(1)
            assert stated != record["answer"]
            assert stated not in record["answer_aliases"]
        else:
            stated_colors = color.findall(text.casefold())
            assert stated_colors
            assert record["answer"] not in stated_colors
            if record["task_family"] == "spatial":
                assert "in the same row" in text


def test_spatial_question_states_the_row(records: list[dict]) -> None:
    for record in records:
        if record["task_family"] == "spatial":
            assert record["question"].endswith("in the same row?")
        else:
            assert "same row" not in record["question"]
