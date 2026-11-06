"""Teacher-forced negative log-likelihood of a full candidate sequence.

``full_logits[i]`` predicts ``input_ids[i + 1]``. Answer token ``t`` is read
from row ``prompt_length - 1 + t``, not from row ``prompt_length + t``. The
primary reduction is the sum of the token NLLs. The mean is the same sum
divided by the token count and is not the decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def sequence_token_logprobs(predicting_logits: torch.Tensor, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Log-probability of each answer token. ``predicting_logits[t]`` predicts ``token_ids[t]``."""
    if not isinstance(predicting_logits, torch.Tensor) or predicting_logits.ndim != 2:
        raise ValueError("predicting logits must have shape [answer_tokens, vocab]")
    working = _working_logits(predicting_logits)
    if working.shape[1] < 2:
        raise ValueError("vocab size must be at least 2")
    ids = _token_tensor(token_ids, device=working.device)
    if ids.shape[0] != working.shape[0]:
        raise ValueError(
            f"token count {ids.shape[0]} does not match predicting rows {working.shape[0]}"
        )
    if int(ids.min()) < 0 or int(ids.max()) >= working.shape[1]:
        raise ValueError("a token id is outside the logit row")
    log_probs = torch.log_softmax(working, dim=-1)
    return log_probs.gather(1, ids.unsqueeze(1)).squeeze(1)


def summed_nll(predicting_logits: torch.Tensor, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Summed NLL of one full candidate. This is the primary score and the attack loss."""
    total = -sequence_token_logprobs(predicting_logits, token_ids).sum()
    _require_finite(total, "summed NLL")
    return total


def mean_nll(predicting_logits: torch.Tensor, token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Mean token NLL. Diagnostic only: it is not the decision and not the attack loss.

    For one fixed candidate the token count does not depend on the image, so
    the mean and the sum have the same gradient sign. Candidates with different
    lengths can still rank differently. The decision uses the sum.
    """
    token_nll = -sequence_token_logprobs(predicting_logits, token_ids)
    average = token_nll.mean()
    _require_finite(average, "mean NLL")
    return average


def aligned_predicting_logits(
    full_logits: torch.Tensor, input_ids: torch.Tensor | Sequence[int], prompt_length: int
) -> torch.Tensor:
    """Rows of ``full_logits`` that predict the answer span, in answer order.

    The returned tensor has one row per answer token. Row 0 is
    ``full_logits[prompt_length - 1]``. The final logit, which predicts a token
    past the answer, is not included.
    """
    logits, ids = _as_sequence(full_logits, input_ids)
    n_answer = _answer_count(prompt_length, int(ids.shape[0]))
    start = prompt_length - 1
    predicting = logits[start : start + n_answer, :]
    if predicting.shape[0] != n_answer:
        raise RuntimeError(
            f"aligned rows {predicting.shape[0]} do not match answer length {n_answer}"
        )
    return predicting


def summed_nll_from_causal_logits(
    full_logits: torch.Tensor, input_ids: torch.Tensor | Sequence[int], prompt_length: int
) -> torch.Tensor:
    """Summed NLL using the causal-LM shift. Differentiable in ``full_logits``."""
    logits, ids = _as_sequence(full_logits, input_ids)
    predicting = aligned_predicting_logits(logits, ids, prompt_length)
    answer_ids = ids[prompt_length:]
    return summed_nll(predicting, answer_ids)


def _working_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dtype in (torch.float16, torch.bfloat16):
        return logits.float()
    if logits.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"logits dtype {logits.dtype} is not floating")
    return logits


def _token_tensor(token_ids: Sequence[int] | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    values = _token_list(token_ids)
    return torch.tensor(values, dtype=torch.long, device=device)


def _token_list(token_ids: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1 or token_ids.numel() < 1:
            raise ValueError("token ids must be a non-empty vector")
        return [int(value) for value in token_ids.detach().cpu().tolist()]
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("token ids must be a sequence of ints")
    values = list(token_ids)
    if not values:
        raise ValueError("candidate has no tokens")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise TypeError("token ids must be ints")
    return [int(value) for value in values]


def _as_sequence(
    full_logits: torch.Tensor, input_ids: torch.Tensor | Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(full_logits, torch.Tensor):
        raise TypeError("logits must be a tensor")
    logits = full_logits
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(f"expected one sequence, got batch {logits.shape[0]}")
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError("full logits must have shape [time, vocab] or [1, time, vocab]")
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids
    else:
        ids = torch.tensor(_token_list(input_ids), dtype=torch.long)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError(f"expected one sequence, got batch {ids.shape[0]}")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError("input_ids must have shape [time] or [1, time]")
    if logits.shape[0] != ids.shape[0]:
        raise ValueError(f"logits time {logits.shape[0]} does not match input_ids {tuple(ids.shape)}")
    return logits, ids


def _answer_count(prompt_length: int, time_steps: int) -> int:
    if isinstance(prompt_length, bool) or not isinstance(prompt_length, int) or prompt_length < 1:
        raise ValueError("prompt_length must be a positive int")
    if prompt_length >= time_steps:
        raise ValueError("prompt_length must leave at least one answer token")
    return time_steps - prompt_length


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value.detach()).all():
        raise RuntimeError(f"{name} is not finite")
