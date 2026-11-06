"""Full-answer log-probability and the loss whose image gradient we expose.

For answer tokens y_1 ... y_T, image x, and prompt text c:

    log P(y | x, c) = sum_t log P(y_t | y_<t, x, c)
    L_c(x) = -log P(y | x, c)
    g_c = gradient of L_c with respect to x

x is the float RGB image in [0, 1], channel-first. T is the number of answer
tokens after the tokenization rule in tokens.py. The sum is the joint
log-probability of the whole answer. Study 1's mean token NLL is L_c(x) / T.
Those two scalars share an argmin when every answer has the same T, and they
differ in scale otherwise.
"""

from __future__ import annotations

import torch


def sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Sum of teacher-forced token log-probabilities for the answer span.

    ``logits[b, i]`` predicts ``input_ids[b, i + 1]``, which is the shift used
    by the installed Transformers causal-LM loss. ``prompt_length`` is the
    number of prompt tokens; answer tokens occupy ``input_ids[:, prompt_length:]``.
    The return value has shape ``(batch,)``.
    """
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [batch, time, vocab] and input_ids [batch, time]")
    if logits.shape[0] != input_ids.shape[0] or logits.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not align with input_ids {tuple(input_ids.shape)}"
        )
    if prompt_length < 1 or prompt_length >= input_ids.shape[1]:
        raise ValueError("prompt_length must leave at least one answer token")
    answer_ids = input_ids[:, prompt_length:]
    predicted = logits[:, prompt_length - 1 : -1, :]
    if predicted.shape[1] != answer_ids.shape[1]:
        raise RuntimeError(
            f"Predicted steps {predicted.shape[1]} do not match answer length {answer_ids.shape[1]}"
        )
    log_probs = torch.log_softmax(predicted.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)


def mean_token_nll(sum_logprob: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Study 1's mean NLL, equal to ``-sum_logprob / T`` for this tokenization."""
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")
    return -sum_logprob / n_tokens
