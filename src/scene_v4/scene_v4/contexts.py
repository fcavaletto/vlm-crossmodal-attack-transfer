"""Four textual contexts.

Neutral text is the Checkpoint 1 sentence. Helpful text says to use every
object word in the question together. It does not name a row, a size, a
shape, a color, or a count. Contradictory text names the color of the same
size and shape in the other row. That color is the unique singleton and is
false on both arms. Revealing text uses the same frame with the true color.
Only that context has ``reveals_answer``.
"""

from __future__ import annotations

from .oracle import QUESTION_RE

NEUTRAL_TEXT = "Several simple objects appear on a plain background."

HELPFUL_TEXT = (
    "Use the object words in the question together. "
    "An object that matches only part of that description is a different object."
)

Context = dict[str, dict[str, object]]


def build_contexts(*, question: str, answer: str, foil: str) -> Context:
    match = QUESTION_RE.fullmatch(question)
    if match is None:
        raise ValueError(f"referent-color question not recognized: {question}")
    size, shape, row_name = match.group(1), match.group(2), match.group(3)
    if foil == answer:
        raise ValueError("contradictory foil must differ from the answer")
    frame = f"The {size} {shape} in the {row_name} row is {{color}}."
    return {
        "neutral": {"text": NEUTRAL_TEXT, "reveals_answer": False},
        "helpful_nonrevealing": {"text": HELPFUL_TEXT, "reveals_answer": False},
        "contradictory": {"text": frame.format(color=foil), "reveals_answer": False},
        "answer_revealing": {"text": frame.format(color=answer), "reveals_answer": True},
    }
