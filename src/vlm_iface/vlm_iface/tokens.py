"""Answer-string tokenization for teacher-forced scoring.

SmolVLM's chat template ends at ``Assistant:``. The continuation observed in
Study 1 is a leading-space BPE piece such as ``Ġred``, not the bare piece
``red``. The default here keeps that convention. The feasibility script records
whether the loaded checkpoint still assigns higher probability to the
leading-space ids than to the bare ids.
"""

from __future__ import annotations

import torch


def answer_token_ids(tokenizer, answer_text: str, *, leading_space: bool = True) -> torch.Tensor:
    """Return a 1-D CPU long tensor of token ids for ``answer_text``.

    Special tokens are not added. A leading space is inserted when
    ``leading_space`` is true and the string does not already start with one.
    """
    if not isinstance(answer_text, str):
        raise TypeError("answer_text must be a string")
    if answer_text.strip() == "":
        raise ValueError("answer_text must contain a non-empty answer")
    text = answer_text
    if leading_space and not answer_text.startswith(" "):
        text = " " + answer_text
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError(f"Answer {answer_text!r} produced no tokens")
    return ids[0].to(dtype=torch.long, device="cpu")
