"""Shared prompt. Question and context stay separate fields.

The rendered user text is three lines: the context, the question, and one
fixed instruction that asks for one of the four colors. The instruction does
not depend on the context or on the scene answer. The answer prefix is the
chat-template generation cue. It is fixed for every color and every context.
A leading space is part of the continuation only when that cue does not
already end in whitespace. That rule is the same function for both models.
"""

from __future__ import annotations

from dataclasses import dataclass

from .colors import COLORS, require_color
from .models import require_model

INSTRUCTION = "Reply with one of these colors: red, blue, green, yellow."


def _one_line(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value == "" or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be one line")
    if INSTRUCTION in value:
        raise ValueError(f"{name} must not already contain the instruction")
    return value


@dataclass(frozen=True)
class ChoicePrompt:
    """Context, question, and the fixed instruction, stored separately."""

    context: str
    question: str
    instruction: str = INSTRUCTION

    def user_text(self) -> str:
        if self.instruction != INSTRUCTION:
            raise ValueError("the instruction is fixed")
        return f"{self.context}\n{self.question}\n{self.instruction}"


def build_prompt(context: str, question: str) -> ChoicePrompt:
    """Build the shared prompt. The answer is not an argument."""
    return ChoicePrompt(context=_one_line("context", context), question=_one_line("question", question))


def parse_user_text(user_text: str) -> ChoicePrompt:
    """Inverse of ``ChoicePrompt.user_text`` for the fixed three-line join."""
    if not isinstance(user_text, str):
        raise TypeError("user_text must be a string")
    parts = user_text.split("\n")
    if len(parts) != 3:
        raise ValueError("user text must be context, question, and instruction")
    context, question, instruction = parts
    if instruction != INSTRUCTION:
        raise ValueError("instruction was changed")
    return build_prompt(context, question)


def prefix_ends_with_whitespace(answer_prefix: str) -> bool:
    """Whether the generation cue already separates the following word."""
    _require_prefix(answer_prefix)
    return answer_prefix[-1].isspace()


def continuation_for(color: str, *, answer_prefix: str) -> str:
    """Candidate text appended after the fixed answer prefix.

    The prefix is not rewritten per color. The returned string is the bare
    color, or one space plus the color, and it does not contain the prefix.
    """
    require_color(color)
    if prefix_ends_with_whitespace(answer_prefix):
        return color
    return " " + color


def scoring_inputs(context: str, question: str, *, model_key: str, answer_prefix: str) -> dict[str, object]:
    """Prompt fields plus the four continuations for one pinned model.

    Continuations depend on the answer prefix, not on the context, the
    question, or a label. Both models call this same function.
    """
    model = require_model(model_key)
    prompt = build_prompt(context, question)
    _require_prefix(answer_prefix)
    continuations = {color: continuation_for(color, answer_prefix=answer_prefix) for color in COLORS}
    return {
        "model_key": model.key,
        "model_id": model.model_id,
        "revision": model.revision,
        "context": prompt.context,
        "question": prompt.question,
        "instruction": prompt.instruction,
        "user_text": prompt.user_text(),
        "answer_prefix": answer_prefix,
        "prefix_ends_with_whitespace": prefix_ends_with_whitespace(answer_prefix),
        "continuations": continuations,
    }


def _require_prefix(answer_prefix: str) -> None:
    if not isinstance(answer_prefix, str) or answer_prefix == "":
        raise ValueError("answer_prefix must be a non-empty string")
    if "{color}" in answer_prefix or "{context}" in answer_prefix or "{question}" in answer_prefix:
        raise ValueError("answer_prefix must be fixed text, not a format template")
