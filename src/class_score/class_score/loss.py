"""Differentiable class loss on constructed CPU logits.

For a class y and a practical surface set S(y):

    L(y) = -log sum_{s in S(y)} P(s | x, c)

P(s | x, c) is the joint probability of the token sequence of s. A one-token
surface is one log-softmax entry at the first answer position. A two-token
surface, the leading-space digit on this study's tokenizer, is the sum of the
two teacher-forced token log-probabilities. Duplicate sequences are counted
once. This module differentiates L with respect to those logits. It does not
load a VLM and it does not differentiate an image.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .surfaces import Surface


def require_cpu(logits: torch.Tensor) -> torch.Tensor:
    """Reject every device other than CPU before any arithmetic."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a CPU tensor")
    if logits.device.type != "cpu":
        raise RuntimeError("class_score differentiates constructed CPU logits only")
    return logits


def _vector(logits: torch.Tensor) -> torch.Tensor:
    flat = require_cpu(logits)
    if flat.ndim == 2 and flat.shape[0] == 1:
        flat = flat[0]
    if flat.ndim != 1:
        raise ValueError(f"logits must have shape [vocab], got {tuple(logits.shape)}")
    if flat.numel() < 2:
        raise ValueError("logits must cover at least two vocabulary ids")
    return flat.to(dtype=torch.float64)


def single_position_class_nll(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """L for surfaces that are all single tokens at one position."""
    flat = _vector(logits)
    if not token_ids:
        raise ValueError("a class needs at least one surface token")
    unique = tuple(dict.fromkeys(int(token) for token in token_ids))
    if any(token < 0 or token >= flat.shape[0] for token in unique):
        raise ValueError("a surface token id is outside the logit vector")
    gathered = torch.log_softmax(flat, dim=-1)[list(unique)]
    return -torch.logsumexp(gathered, dim=0)


def word_only_nll(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """Negative log-probability of one canonical token."""
    return single_position_class_nll(logits, (token_id,))


def class_nll_from_surfaces(step_logits: Sequence[torch.Tensor], surfaces: Sequence[Surface]) -> torch.Tensor:
    """L(y) from the joint log-probabilities of ``surfaces``.

    ``step_logits[t]`` predicts answer token t. A one-token surface reads
    position 0. A two-token surface reads positions 0 and 1.
    """
    if not surfaces:
        raise ValueError("a class needs at least one surface")
    if not step_logits:
        raise ValueError("logits are empty")
    pieces: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    for surface in surfaces:
        key = tuple(int(token) for token in surface.token_ids)
        if key in seen:
            continue
        seen.add(key)
        if len(key) < 1 or len(key) > 2:
            raise ValueError("practical surfaces are one token, or the two-token digit")
        if len(key) > len(step_logits):
            raise ValueError("not enough logit positions for a multi-token surface")
        log_probability = torch.zeros((), dtype=torch.float64)
        for position, token in enumerate(key):
            flat = _vector(step_logits[position])
            if token < 0 or token >= flat.shape[0]:
                raise ValueError("a surface token id is outside the logit vector")
            log_probability = log_probability + torch.log_softmax(flat, dim=-1)[token]
        pieces.append(log_probability)
    return -torch.logsumexp(torch.stack(pieces), dim=0)


def finite_difference_positions(
    loss_fn: Callable[..., float | torch.Tensor],
    step_logits: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, ...]:
    """Central difference of a scalar loss with respect to each logit vector.

    ``loss_fn`` is called as ``loss_fn(*step_logits)``. The step is CPU
    arithmetic on the values in those tensors. No image tensor is built.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    bases = [_vector(logits).detach() for logits in step_logits]
    gradients: list[torch.Tensor] = []
    for position, base in enumerate(bases):
        gradient = torch.empty_like(base)
        for index in range(base.numel()):
            plus = [item.clone() for item in bases]
            minus = [item.clone() for item in bases]
            plus[position][index] += epsilon
            minus[position][index] -= epsilon
            higher = float(loss_fn(*plus))
            lower = float(loss_fn(*minus))
            gradient[index] = (higher - lower) / (2.0 * epsilon)
        gradients.append(gradient)
    return tuple(gradients)
