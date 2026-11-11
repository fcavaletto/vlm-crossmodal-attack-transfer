"""Four textual contexts. Wording is fixed; only the slot values change.

``helpful_nonrevealing`` tells the reader which rule to apply and does not name
the answer. Within a counterfactual pair the neutral, helpful, and contradictory
strings are identical, so those strings cannot tell the two answers apart.
``answer_revealing`` is a positive control and is allowed to name the answer.
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
        contradictory = f"There are {foil} {color} objects."
        revealing = f"There are {answer} {color} objects."
    elif task_family == "spatial":
        match = SPATIAL_RE.fullmatch(question)
        if match is None:
            raise ValueError(f"spatial question not recognized: {question}")
        size, shape = match.group(1), match.group(2)
        helpful = SPATIAL_HELPFUL
        contradictory = f"The object immediately left of the {size} {shape} is {foil}."
        revealing = f"The object immediately left of the {size} {shape} is {answer}."
    else:
        raise ValueError(f"unknown task family: {task_family}")

    return {
        "neutral": {"text": NEUTRAL_TEXT, "reveals_answer": False},
        "helpful_nonrevealing": {"text": helpful, "reveals_answer": False},
        "contradictory": {"text": contradictory, "reveals_answer": False},
        "answer_revealing": {"text": revealing, "reveals_answer": True},
    }
