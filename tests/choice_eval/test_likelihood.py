"""Logit alignment, multi-token likelihood, and a finite differentiable loss."""

from __future__ import annotations

import pytest
import torch
from vlm_iface.logprob import sequence_logprob

from choice_eval.likelihood import (
    mean_nll,
    summed_nll,
    summed_nll_from_causal_logits,
    aligned_predicting_logits,
)


def test_causal_logits_read_the_shifted_answer_row():
    full = torch.full((4, 5), -30.0)
    full[2, 4] = 30.0
    full[3, 0] = 30.0
    full[3, 4] = -30.0
    input_ids = torch.tensor([0, 1, 2, 4])
    prompt_length = 3
    predicting = aligned_predicting_logits(full, input_ids, prompt_length)
    assert predicting.shape == (1, 5)
    assert torch.equal(predicting[0], full[prompt_length - 1])
    assert not torch.equal(predicting[0], full[prompt_length])
    aligned = summed_nll_from_causal_logits(full, input_ids, prompt_length)
    shifted = summed_nll(full[prompt_length : prompt_length + 1], [4])
    assert float(aligned) < 0.01
    assert float(shifted) > 10.0


def test_summed_nll_matches_the_library_shift():
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(6, 9, generator=generator)
    input_ids = torch.tensor([1, 2, 3, 4, 5, 0])
    prompt_length = 4
    ours = summed_nll_from_causal_logits(full, input_ids, prompt_length)
    library = sequence_logprob(full.view(1, 6, 9), input_ids.view(1, 6), prompt_length)
    assert torch.allclose(ours, -library[0], atol=1e-6, rtol=1e-6)
    batched = summed_nll_from_causal_logits(full.view(1, 6, 9), input_ids.view(1, 6), prompt_length)
    assert torch.allclose(batched, ours)


def test_multi_token_likelihood_sums_the_conditionals():
    predicting = torch.tensor(
        [
            [0.0, 4.0, -2.0],
            [3.0, 0.0, -1.0],
        ],
        dtype=torch.float64,
    )
    token_ids = [1, 0]
    joint = summed_nll(predicting, token_ids)
    first = summed_nll(predicting[0:1], [1])
    second = summed_nll(predicting[1:2], [0])
    assert torch.allclose(joint, first + second)
    assert float(joint) > float(first)
    full = torch.zeros(4, 3, dtype=torch.float64)
    full[1] = predicting[0]
    full[2] = predicting[1]
    full[3] = torch.tensor([10.0, -10.0, -10.0])
    causal = summed_nll_from_causal_logits(full, [7, 8, 1, 0], prompt_length=2)
    assert torch.allclose(causal, joint)
    wrong_slice = summed_nll(full[2:4], [1, 0])
    assert float(wrong_slice) > float(causal) + 1.0


def test_backward_reaches_only_the_aligned_answer_rows():
    leaf = torch.zeros(4, 6, dtype=torch.float64, requires_grad=True)
    input_ids = [0, 1, 3, 4]
    loss = summed_nll_from_causal_logits(leaf, input_ids, prompt_length=2)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert torch.count_nonzero(leaf.grad[0]) == 0
    assert torch.count_nonzero(leaf.grad[3]) == 0
    assert float(leaf.grad[1, 3]) < 0
    assert float(leaf.grad[2, 4]) < 0
    assert float(leaf.grad[1, 0]) > 0


def test_finite_difference_matches_autograd():
    logits = torch.tensor(
        [[0.2, -0.4, 1.1, 0.3], [0.5, -1.0, 0.2, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    token_ids = [2, 3]
    loss = summed_nll(logits, token_ids)
    loss.backward()
    numerical = torch.empty_like(logits)
    epsilon = 1e-5
    base = logits.detach()
    for row in range(base.shape[0]):
        for column in range(base.shape[1]):
            plus = base.clone()
            minus = base.clone()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            higher = summed_nll(plus, token_ids)
            lower = summed_nll(minus, token_ids)
            numerical[row, column] = (higher - lower) / (2.0 * epsilon)
    assert torch.allclose(numerical, logits.grad, atol=1e-6, rtol=1e-5)
    assert float(numerical[0, 2]) < 0
    assert float(numerical[1, 3]) < 0
    assert float(numerical[0, 0]) > 0
    assert torch.isfinite(numerical).all()


def test_bfloat16_loss_is_finite_and_differentiable():
    logits = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.bfloat16, requires_grad=True)
    loss = summed_nll(logits, [1])
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, 1]) < 0


def test_mean_is_the_sum_divided_by_length():
    logits = torch.tensor([[0.0, 1.5, 0.0], [0.2, 0.0, 1.2]], dtype=torch.float64)
    total = summed_nll(logits, [1, 2])
    average = mean_nll(logits, [1, 2])
    assert torch.allclose(average * 2, total)
    assert not torch.allclose(average, total)


def test_nonfinite_logits_are_rejected():
    with pytest.raises(RuntimeError, match="not finite"):
        summed_nll(torch.tensor([[float("nan"), 0.0]]), [0])
    with pytest.raises(ValueError, match="outside"):
        summed_nll(torch.tensor([[0.0, 1.0]]), [4])
    with pytest.raises(ValueError, match="prompt_length"):
        summed_nll_from_causal_logits(torch.zeros(3, 4), [1, 2, 3], prompt_length=3)


def test_batch_greater_than_one_is_rejected():
    with pytest.raises(ValueError, match="batch"):
        summed_nll_from_causal_logits(torch.zeros(2, 3, 4), torch.zeros(2, 3, dtype=torch.long), 1)
