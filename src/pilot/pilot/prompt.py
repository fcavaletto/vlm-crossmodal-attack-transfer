"""User message for one scene and one context.

Decision D2: the user message is the context text, a newline, and the question.
It does not add a ``Context:`` label, a ``Question:`` label, or an answer label.
The model session wraps that string in the chat template.
"""

from __future__ import annotations


def user_prompt(context_text: str, question: str) -> str:
    """Join one stored context and one stored question."""
    if not isinstance(context_text, str) or not isinstance(question, str):
        raise TypeError("context and question must be strings")
    if context_text == "" or question == "" or context_text != context_text.strip() or question != question.strip():
        raise ValueError("context and question must be non-empty stripped strings")
    if "\n" in context_text or "\r" in context_text or "\n" in question or "\r" in question:
        raise ValueError("context and question must each be one line")
    prompt = context_text + "\n" + question
    if "Context:" in prompt or "Question:" in prompt:
        raise ValueError("the user message must not add Context: or Question: labels")
    return prompt
