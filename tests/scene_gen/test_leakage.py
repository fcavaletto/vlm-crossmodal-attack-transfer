"""Answer leakage. The helpful context must fail a substring test, not a promise."""

from __future__ import annotations

import re

from scene_gen.normalize import normalize_for_leakage

NON_REVEALING = ("neutral", "helpful_nonrevealing", "contradictory")
LABEL_WORDS = ("red", "blue", "green", "yellow", "one", "two", "three", "four")


def test_normalization_is_casefold_and_literal_substring() -> None:
    assert normalize_for_leakage("  The LARGE circle is RED!!! ") == "the large circle is red"
    assert "red" in normalize_for_leakage("The large circle is red.")
    # The contract is a substring test. "one" occurs inside "none".
    assert "one" in normalize_for_leakage("none")
    assert "2" in normalize_for_leakage("item 2")


def test_helpful_text_does_not_contain_the_answer_or_aliases(records: list[dict]) -> None:
    for record in records:
        raw = record["contexts"]["helpful_nonrevealing"]["text"].casefold()
        norm = normalize_for_leakage(record["contexts"]["helpful_nonrevealing"]["text"])
        assert record["contexts"]["helpful_nonrevealing"]["reveals_answer"] is False
        for alias in record["answer_aliases"]:
            assert alias.casefold() not in raw
            assert normalize_for_leakage(alias) not in norm


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


def test_contradictory_text_states_a_different_label(records: list[dict]) -> None:
    color = re.compile(r"\b(red|blue|green|yellow)\b")
    for record in records:
        text = record["contexts"]["contradictory"]["text"].casefold()
        if record["task_family"] == "counting":
            assert re.search(r"\bfour\b", text)
            assert record["answer"] != "four"
            assert record["answer"] not in text
        else:
            stated = color.findall(text)
            assert stated, text
            assert record["answer"] not in stated
