"""Chosen-answer loss for one teacher-forced color sequence.

The prospective primary reduction is the summed negative log-likelihood of the
full candidate sequence. The candidate here is the fixture answer with the
leading-space tokenization already used for this study. The mean over tokens
is not the loss. No alternative is selected by how it scores.
"""

from __future__ import annotations

import torch
from vlm_iface.logprob import sequence_logprob

SKIP_EXTEND = frozenset(
    {"input_ids", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
)
ONES_ON_EXTEND = frozenset({"attention_mask"})


def chosen_answer_nll(logits: torch.Tensor, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Scalar summed NLL of the answer span. One sequence only."""
    log_probability = sequence_logprob(logits, input_ids, prompt_length)
    if log_probability.ndim != 1 or log_probability.shape[0] != 1:
        raise ValueError(f"expected one sequence, got log-probability shape {tuple(log_probability.shape)}")
    return -log_probability.sum()


def extend_with_answer(inputs: dict, answer_ids: torch.Tensor) -> tuple[dict, int]:
    """Append answer token ids and lengthen the matching text masks.

    Vision tensors are left alone. ``attention_mask`` is extended with ones.
    Other sequence-length masks, including ``mm_token_type_ids``, are extended
    with zeros because the answer tokens are text.
    """
    if "input_ids" not in inputs or not torch.is_tensor(inputs["input_ids"]):
        raise ValueError("inputs must contain input_ids")
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    prompt_length = int(inputs["input_ids"].shape[-1])
    if prompt_length < 1:
        raise ValueError("the prompt is empty")
    answer = answer_ids.view(1, -1).to(dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
    if answer.shape[-1] < 1:
        raise ValueError("the answer has no tokens")
    extended = dict(inputs)
    extended["input_ids"] = torch.cat([extended["input_ids"], answer], dim=-1)
    for key, value in list(extended.items()):
        if key in SKIP_EXTEND or not torch.is_tensor(value):
            continue
        if value.ndim == 2 and value.shape[0] == 1 and value.shape[-1] == prompt_length:
            fill = 1 if key in ONES_ON_EXTEND else 0
            extra = torch.full((1, answer.shape[-1]), fill, dtype=value.dtype, device=value.device)
            extended[key] = torch.cat([value, extra], dim=-1)
    return extended, prompt_length
