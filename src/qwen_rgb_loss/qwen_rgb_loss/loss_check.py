"""Same-row checks that the differentiated scalar is the summed labeled-color NLL."""

from __future__ import annotations

from typing import Any

import torch
from choice_eval.likelihood import summed_nll, summed_nll_from_causal_logits
from vlm_iface.logprob import sequence_logprob

from .protocol import SAME_ROW_LOSS_MAX_ABS


def same_row_identity(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_length: int,
    token_id: int,
) -> dict[str, Any]:
    """Compare three formulas on the logit row that predicts the first answer token.

    The returned loss is the live ``choice_eval.summed_nll`` tensor. Backward,
    when the caller takes it, is applied to that tensor. The other two formulas
    are detached copies of the same row plus an unused final logit.
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1, time, vocab]")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, time]")
    if int(input_ids.shape[1]) != prompt_length:
        raise ValueError("prompt_length must be the current input length")
    row = logits[0, -1]
    loss = summed_nll(row.unsqueeze(0), [int(token_id)])
    with torch.no_grad():
        manual = -torch.log_softmax(row.detach().float(), dim=-1)[int(token_id)]
        dummy = torch.zeros_like(logits[:, :1, :])
        extended_logits = torch.cat([logits.detach(), dummy], dim=1)
        answer = torch.tensor([[int(token_id)]], dtype=input_ids.dtype, device=input_ids.device)
        extended_ids = torch.cat([input_ids, answer], dim=1)
        causal = summed_nll_from_causal_logits(extended_logits, extended_ids, prompt_length)
        library = -sequence_logprob(extended_logits, extended_ids, prompt_length)[0]
    live = float(loss.detach().float().cpu())
    gaps = {
        "manual_negative_log_softmax": abs(live - float(manual.cpu())),
        "choice_eval_causal_nll": abs(live - float(causal.detach().cpu())),
        "vlm_iface_sequence_logprob": abs(live - float(library.detach().cpu())),
    }
    max_abs = max(gaps.values())
    return {
        "loss": loss,
        "loss_value": live,
        "gaps": gaps,
        "max_abs": max_abs,
        "agreed": max_abs <= SAME_ROW_LOSS_MAX_ABS,
        "tolerance": SAME_ROW_LOSS_MAX_ABS,
        "token_id": int(token_id),
        "reduction": "summed_nll",
    }
