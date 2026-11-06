"""Four textual contexts.

Neutral text is the Checkpoint 1 sentence. Helpful text is guidance for this
task: use the row and the shape the question already names, and do not take
the same shape in the other row. It does not name a row, a color, a size, or
a count. It does not ask for a sentence.

Contradictory text names the color of the same shape in the other row. That
color is false for the named row on both arms of a pair. Revealing text uses
the same frame with the true color. Only that context has ``reveals_answer``.
"""

from __future__ import annotations

from .oracle import QUESTION_RE

NEUTRAL_TEXT = "Several simple objects appear on a plain background."

HELPFUL_TEXT = (
    "Use the row named in the question and the shape named in the question. "
    "A matching shape in the other row is a different object."
)

Context = dict[str, dict[str, object]]


def build_contexts(*, question: str, answer: str, foil: str) -> Context:
    match = QUESTION_RE.fullmatch(question)
    if match is None:
        raise ValueError(f"row-color question not recognized: {question}")
    shape, row_name = match.group(1), match.group(2)
    if foil == answer:
        raise ValueError("contradictory foil must differ from the answer")
    frame = f"The {shape} in the {row_name} row is {{color}}."
    return {
        "neutral": {"text": NEUTRAL_TEXT, "reveals_answer": False},
        "helpful_nonrevealing": {"text": HELPFUL_TEXT, "reveals_answer": False},
        "contradictory": {"text": frame.format(color=foil), "reveals_answer": False},
        "answer_revealing": {"text": frame.format(color=answer), "reveals_answer": True},
    }
