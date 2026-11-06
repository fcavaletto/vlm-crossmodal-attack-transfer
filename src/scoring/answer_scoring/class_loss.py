"""Class negative log-likelihood over parser-accepted surfaces.

For one answer position with logits z and a set S of token ids that the strict
parser already accepts as class y:

    log P(y) = log sum_{i in S} softmax(z)_i
    L(y) = -log P(y)

L is the sum-token loss when every surface is one token. Extra surfaces can
only decrease L relative to the single canonical token. The canonical-word loss
is the special case S = {canonical token}. Putting probability on a digit alias
increases that word loss and decreases this class loss. The class loss is the
scalar an image gradient would differentiate. This module does not take that
image gradient and does not run an attack.
"""

from __future__ import annotations

import torch


def _vocab_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 1:
        raise ValueError(f"logits must have shape [vocab], got {tuple(logits.shape)}")
    if logits.numel() < 2:
        raise ValueError("logits must cover at least two vocabulary ids")
    return logits


def token_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Log-softmax in float32, retaining the caller graph when logits require grad."""
    flat = _vocab_logits(logits)
    return torch.log_softmax(flat.float(), dim=-1)


def word_only_nll(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """Negative log-probability of one canonical token. This is the D1 surface."""
    if token_id < 0 or token_id >= _vocab_logits(logits).shape[0]:
        raise ValueError(f"token id {token_id} is outside the logit vector")
    return -token_log_softmax(logits)[token_id]


def class_nll(logits: torch.Tensor, token_ids: list[int] | tuple[int, ...]) -> torch.Tensor:
    """Negative log-probability of the union of single-token ``token_ids``."""
    flat = _vocab_logits(logits)
    if not token_ids:
        raise ValueError("a class needs at least one surface token")
    unique = tuple(dict.fromkeys(int(token) for token in token_ids))
    if any(token < 0 or token >= flat.shape[0] for token in unique):
        raise ValueError("a surface token id is outside the logit vector")
    gathered = token_log_softmax(logits)[list(unique)]
    return -torch.logsumexp(gathered, dim=0)


def mixture_nll(logprobs: torch.Tensor) -> torch.Tensor:
    """Negative log-sum-exp of sequence log-probabilities.

    Each entry is already ``log P(surface)``. Surfaces may have different
    lengths. The sum is the probability of their disjoint union.
    """
    if logprobs.ndim != 1 or logprobs.numel() < 1:
        raise ValueError("logprobs must be a non-empty vector of sequence log-probabilities")
    return -torch.logsumexp(logprobs.float(), dim=0)


def two_token_logprob(
    first_logits: torch.Tensor, next_logits: torch.Tensor, first_id: int, second_id: int
) -> torch.Tensor:
    """Joint log-probability of a two-token surface under teacher forcing."""
    return token_log_softmax(first_logits)[first_id] + token_log_softmax(next_logits)[second_id]


def argmin_name(scores: dict[str, float]) -> str:
    """Lowest score, then the name. The same tie break as the Checkpoint 1 rows."""
    if not scores:
        raise ValueError("scores are empty")
    return min(scores, key=lambda name: (scores[name], name))
