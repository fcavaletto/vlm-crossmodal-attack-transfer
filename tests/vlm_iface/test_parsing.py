"""D4 parser. Invalid is a vocabulary judgment, not a factual one."""

from __future__ import annotations

import pytest

from vlm_iface.parsing import parse_generation


def test_color_with_case_quotes_and_terminal_punctuation_is_canonical():
    parsed = parse_generation('  "Yellow!"  ', truncated=False, task_family="attribute")
    assert parsed.invalid is False
    assert parsed.parsed_answer == "yellow"
    assert parsed.as_contract()["text"] == '"Yellow!"'


def test_answer_label_is_not_stripped_and_is_invalid():
    parsed = parse_generation("Answer: blue", truncated=False, task_family="attribute")
    assert parsed.invalid is True
    assert parsed.parsed_answer is None
    assert "not_exact" in parsed.reasons


def test_digit_alias_is_the_canonical_count_word_and_is_not_a_color():
    counted = parse_generation("3", truncated=False, task_family="counting")
    assert counted.invalid is False
    assert counted.parsed_answer == "three"
    color = parse_generation("3", truncated=False, task_family="attribute")
    assert color.invalid is True
    assert "not_in_vocabulary" in color.reasons


def test_wrong_recognized_color_and_count_stay_valid_parses():
    assert parse_generation("green", truncated=False, task_family="spatial").parsed_answer == "green"
    assert parse_generation("four", truncated=False, task_family="counting").parsed_answer == "four"


def test_option_lists_and_multiple_labels_are_invalid():
    echo = parse_generation("red or blue", truncated=False, task_family="attribute")
    assert echo.invalid is True
    assert "multiple_labels" in echo.reasons
    listed = parse_generation("a) red b) blue", truncated=False, task_family="spatial")
    assert "multiple_labels" in listed.reasons
    phrase = parse_generation("left of", truncated=False, task_family="spatial")
    assert phrase.invalid is True
    assert phrase.parsed_answer is None
    assert "not_in_vocabulary" in phrase.reasons


def test_empty_multiline_refusal_and_extra_words_are_invalid():
    assert parse_generation("   ", truncated=False, task_family="attribute").reasons == ("empty",)
    assert parse_generation("red\nbecause it looks red", truncated=False, task_family="attribute").reasons == (
        "multiline",
    )
    refusal = parse_generation("I don't know", truncated=False, task_family="counting")
    assert refusal.invalid is True
    assert refusal.parsed_answer is None
    assert "refusal" in refusal.reasons
    extra = parse_generation("the large red cube is left of the blue sphere", truncated=False, task_family="attribute")
    assert extra.invalid is True
    assert "multiple_labels" in extra.reasons


def test_truncation_marks_invalid_even_when_the_text_is_a_vocabulary_item():
    parsed = parse_generation("red", truncated=True, task_family="attribute")
    assert parsed.invalid is True
    assert parsed.parsed_answer is None
    assert "truncated" in parsed.reasons


def test_punctuation_only_is_invalid():
    parsed = parse_generation("...", truncated=False, task_family="attribute")
    assert parsed.invalid is True
    assert parsed.parsed_answer is None
    assert "empty" in parsed.reasons
    stars = parse_generation("***", truncated=False, task_family="counting")
    assert "no_alphanumeric" in stars.reasons


def test_unknown_task_family_is_rejected():
    with pytest.raises(ValueError, match="task family"):
        parse_generation("red", truncated=False, task_family="caption")
