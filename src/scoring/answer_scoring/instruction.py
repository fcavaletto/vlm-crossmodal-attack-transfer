"""One output-format line, appended identically to every context.

Checkpoint 1 decision D2 builds the user message as ``context_text + newline +
question``. That message does not state decision D4's output contract. The line
below states the contract without naming a color, a count, or a digit, and
without depending on the context or the answer. It is a development probe. It
is not a replacement for the stored Checkpoint 1 prompts.
"""

from __future__ import annotations

from scene_gen.normalize import leakage_aliases
from scene_gen.vocabulary import COLORS, WORD_TO_COUNT

from pilot.prompt import user_prompt

# Fixed before any Checkpoint 1.5 model measurement. Not interpolated.
FORMAT_INSTRUCTION = "Reply using only the answer. Do not write a sentence."

_BANNED_ALIASES: tuple[str, ...] = tuple(COLORS) + tuple(WORD_TO_COUNT) + tuple(
    str(digit) for digit in WORD_TO_COUNT.values()
)


def format_instruction_leakage() -> list[str]:
    """Vocabulary aliases that appear inside the format line. Empty is required."""
    return leakage_aliases(FORMAT_INSTRUCTION, list(_BANNED_ALIASES))


def format_user_prompt(context_text: str, question: str) -> str:
    """Return the D2 message plus the same format line for every context."""
    if format_instruction_leakage():
        raise RuntimeError(
            "The format instruction contains a task-vocabulary alias: "
            f"{format_instruction_leakage()}"
        )
    if "\n" in FORMAT_INSTRUCTION or "\r" in FORMAT_INSTRUCTION:
        raise RuntimeError("The format instruction must be one line.")
    if "{" in FORMAT_INSTRUCTION or "}" in FORMAT_INSTRUCTION:
        raise RuntimeError("The format instruction must not be a format string.")
    base = user_prompt(context_text, question)
    prompted = base + "\n" + FORMAT_INSTRUCTION
    if not prompted.endswith("\n" + FORMAT_INSTRUCTION):
        raise RuntimeError("The format line was not appended unchanged.")
    if prompted.count(FORMAT_INSTRUCTION) != 1:
        raise RuntimeError("The format instruction must appear once.")
    return prompted
