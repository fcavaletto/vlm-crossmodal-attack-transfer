"""Parser cases. Invalid is a format judgment, not a factual one."""

from __future__ import annotations

from vlm_iface.parsing import parse_generation


def test_word_with_punctuation_is_a_short_answer():
    parsed = parse_generation("Red.", truncated=False)
    assert parsed.invalid is False
    assert parsed.parsed_answer == "red"
    assert parsed.as_contract() == {"text": "Red.", "parsed_answer": "red", "invalid": False}


def test_answer_prefix_is_removed():
    parsed = parse_generation("Answer: blue", truncated=False)
    assert parsed.parsed_answer == "blue"
    assert parsed.invalid is False


def test_count_and_short_spatial_phrase_are_valid():
    assert parse_generation("3", truncated=False).parsed_answer == "3"
    assert parse_generation("left of", truncated=False).parsed_answer == "left of"


def test_empty_multiline_refusal_and_length_are_invalid():
    assert parse_generation("   ", truncated=False).reasons == ("empty",)
    assert parse_generation("red\nbecause it looks red", truncated=False).reasons == ("multiline",)
    refusal = parse_generation("I don't know", truncated=False)
    assert refusal.invalid is True
    assert refusal.parsed_answer is None
    assert "refusal" in refusal.reasons
    long = parse_generation("the large red cube is left of the blue sphere", truncated=False)
    assert long.reasons == ("too_long",)
    assert long.parsed_answer is None


def test_truncation_marks_invalid_even_when_the_text_looks_usable():
    parsed = parse_generation("red", truncated=True)
    assert parsed.invalid is True
    assert parsed.parsed_answer is None
    assert "truncated" in parsed.reasons


def test_punctuation_only_is_invalid():
    parsed = parse_generation("...", truncated=False)
    assert parsed.invalid is True
    assert parsed.parsed_answer is None
    assert "empty" in parsed.reasons
    stars = parse_generation("***", truncated=False)
    assert "no_alphanumeric" in stars.reasons
