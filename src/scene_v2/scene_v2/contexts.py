"""Four textual contexts.

Neutral text and the three helpful sentences are the Checkpoint 1 strings.
They are not rewritten to chase a score. Contradictory and revealing sentences
name the foil or the answer. Within a pair the neutral, helpful, and
contradictory strings match.

The counting foil ``one`` uses a singular sentence. ``There are one red
objects`` is not English, and a broken sentence would be a different control
from the plural line used for two, three, and four.
"""

from __future__ import annotations

from .oracle import ATTRIBUTE_RE, COUNTING_RE, SPATIAL_RE

NEUTRAL_TEXT = "Several simple objects appear on a plain background."

ATTRIBUTE_HELPFUL = (
    "Identify the single object with the size and shape named in the question, "
    "then name its color."
)
COUNTING_HELPFUL = (
    "Count objects of the color named in the question. Size and shape do not matter."
)
SPATIAL_HELPFUL = (
    "Use the object named in the question as the anchor. "
    "Name the color of the adjacent object toward the left edge in the same row."
)

Context = dict[str, dict[str, object]]


def counting_statement(number_word: str, color: str) -> str:
    """A short line that states a count. ``one`` is singular; the others are plural."""
    if number_word == "one":
        return f"There is one {color} object."
    return f"There are {number_word} {color} objects."


def build_contexts(
    *,
    task_family: str,
    question: str,
    answer: str,
    foil: str,
) -> Context:
    if task_family == "attribute":
        match = ATTRIBUTE_RE.fullmatch(question)
        if match is None:
            raise ValueError(f"attribute question not recognized: {question}")
        size, shape = match.group(1), match.group(2)
        helpful = ATTRIBUTE_HELPFUL
        contradictory = f"The {size} {shape} is {foil}."
        revealing = f"The {size} {shape} is {answer}."
    elif task_family == "counting":
        match = COUNTING_RE.fullmatch(question)
        if match is None:
            raise ValueError(f"counting question not recognized: {question}")
        color = match.group(1)
        helpful = COUNTING_HELPFUL
        contradictory = counting_statement(foil, color)
        revealing = counting_statement(answer, color)
    elif task_family == "spatial":
        match = SPATIAL_RE.fullmatch(question)
        if match is None:
            raise ValueError(f"spatial question not recognized: {question}")
        size, shape = match.group(1), match.group(2)
        helpful = SPATIAL_HELPFUL
        contradictory = (
            f"The object immediately left of the {size} {shape} in the same row is {foil}."
        )
        revealing = (
            f"The object immediately left of the {size} {shape} in the same row is {answer}."
        )
    else:
        raise ValueError(f"unknown task family: {task_family}")

    return {
        "neutral": {"text": NEUTRAL_TEXT, "reveals_answer": False},
        "helpful_nonrevealing": {"text": helpful, "reveals_answer": False},
        "contradictory": {"text": contradictory, "reveals_answer": False},
        "answer_revealing": {"text": revealing, "reveals_answer": True},
    }
