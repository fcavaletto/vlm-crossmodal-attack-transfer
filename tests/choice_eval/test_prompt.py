"""Shared prompt. Question and context stay separate. No weights are loaded."""

from __future__ import annotations

import inspect

import pytest

from choice_eval.prompt import INSTRUCTION, build_prompt, parse_user_text, scoring_inputs

QUESTION = "What color is the circle in the top row?"
CONTEXTS = {
    "neutral": "Several simple objects appear on a plain background.",
    "helpful_nonrevealing": "Use the row named in the question and the shape named in the question.",
    "contradictory": "The circle in the top row is green.",
    "answer_revealing": "The circle in the top row is red.",
}


def test_instruction_asks_for_each_color_once():
    assert INSTRUCTION == "Reply with one of these colors: red, blue, green, yellow."
    for color in ("red", "blue", "green", "yellow"):
        assert INSTRUCTION.count(color) == 1
    assert "{" not in INSTRUCTION


def test_question_and_context_round_trip_as_separate_lines():
    prompt = build_prompt(CONTEXTS["neutral"], QUESTION)
    assert prompt.context == CONTEXTS["neutral"]
    assert prompt.question == QUESTION
    assert prompt.instruction == INSTRUCTION
    assert prompt.user_text().split("\n") == [prompt.context, prompt.question, prompt.instruction]
    assert parse_user_text(prompt.user_text()) == prompt
    assert "Context:" not in prompt.user_text()
    assert "Question:" not in prompt.user_text()


def test_instruction_does_not_depend_on_context_or_a_label():
    assert "label" not in inspect.signature(scoring_inputs).parameters
    assert "answer" not in inspect.signature(build_prompt).parameters
    rendered = []
    continuations = []
    for context in CONTEXTS.values():
        row = scoring_inputs(context, QUESTION, model_key="smolvlm2", answer_prefix="Assistant:")
        rendered.append(row["user_text"])
        continuations.append(row["continuations"])
        assert row["question"] == QUESTION
        assert row["instruction"] == INSTRUCTION
        assert row["context"] == context
    assert len(set(rendered)) == 4
    assert all(item == continuations[0] for item in continuations)
    assert continuations[0] == {
        "red": " red",
        "blue": " blue",
        "green": " green",
        "yellow": " yellow",
    }


def test_both_models_use_the_same_prefix_rule():
    smol = scoring_inputs(CONTEXTS["neutral"], QUESTION, model_key="smolvlm2", answer_prefix="Assistant:")
    qwen = scoring_inputs(
        CONTEXTS["contradictory"], QUESTION, model_key="qwen3-vl-2b", answer_prefix="assistant\n"
    )
    assert smol["continuations"]["red"] == " red"
    assert qwen["continuations"]["red"] == "red"
    assert qwen["prefix_ends_with_whitespace"] is True
    assert smol["prefix_ends_with_whitespace"] is False
    assert "green" in qwen["context"]
    assert qwen["continuations"] == {color: color for color in ("red", "blue", "green", "yellow")}


def test_changed_instruction_is_rejected():
    text = build_prompt(CONTEXTS["neutral"], QUESTION).user_text()
    with pytest.raises(ValueError, match="instruction"):
        parse_user_text(text.replace(INSTRUCTION, "Say the answer."))


@pytest.mark.parametrize(
    ("context", "question", "match"),
    [
        ("", QUESTION, "context"),
        (CONTEXTS["neutral"], "", "question"),
        (" padded", QUESTION, "context"),
        (CONTEXTS["neutral"] + "\nmore", QUESTION, "one line"),
        (INSTRUCTION, QUESTION, "instruction"),
    ],
)
def test_prompt_fields_are_one_line(context, question, match):
    with pytest.raises(ValueError, match=match):
        build_prompt(context, question)


def test_answer_prefix_is_not_a_format_template():
    with pytest.raises(ValueError, match="fixed text"):
        scoring_inputs(CONTEXTS["neutral"], QUESTION, model_key="smolvlm2", answer_prefix="Assistant: {color}")
    with pytest.raises(ValueError, match="unknown model"):
        scoring_inputs(CONTEXTS["neutral"], QUESTION, model_key="other", answer_prefix="Assistant:")
