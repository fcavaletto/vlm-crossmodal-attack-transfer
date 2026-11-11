"""Directional check on functions that do not load a model."""

from __future__ import annotations

import torch

from qwen_rgb.finite_difference import (
    BF16_RELATIVE_ERROR_CEILING,
    BF16_ULP_AT_ONE,
    FLOAT32_RELATIVE_ERROR_CEILING,
    PRIMARY_DELTAS,
    analytic_directional,
    ceiling_for_dtype,
    central_difference,
    judge_samples,
    rademacher,
)


def test_precommitted_steps_are_at_least_one_bf16_ulp():
    assert BF16_ULP_AT_ONE == 2.0**-7
    assert PRIMARY_DELTAS == (1.0 / 64.0, 1.0 / 128.0)
    assert min(PRIMARY_DELTAS) >= BF16_ULP_AT_ONE
    assert BF16_RELATIVE_ERROR_CEILING == 0.5
    assert FLOAT32_RELATIVE_ERROR_CEILING == 0.05
    assert ceiling_for_dtype("bfloat16") == 0.5
    assert ceiling_for_dtype("float32") == 0.05


def test_linear_directional_derivative_matches():
    weight = torch.tensor([0.25, -0.5, 0.125])

    def loss_fn(image: torch.Tensor) -> float:
        return float((image * weight).sum())

    image = torch.zeros(3)
    direction = torch.tensor([1.0, -1.0, 1.0])
    analytic = analytic_directional(weight, direction)
    estimate = central_difference(loss_fn, image, direction, PRIMARY_DELTAS[0])
    assert abs(analytic - estimate) < 1e-6
    judged = judge_samples(analytic, [(PRIMARY_DELTAS[0], estimate)], ceiling=ceiling_for_dtype("float32"))
    assert judged["confirmed"] is True


def test_rademacher_direction_is_deterministic_and_signed():
    first = rademacher((3, 4, 5), seed=17)
    second = rademacher((3, 4, 5), seed=17)
    assert torch.equal(first, second)
    assert set(first.unique().tolist()) == {-1.0, 1.0}


def test_bf16_rounding_hides_a_step_below_one_ulp():
    def rounded_identity(image: torch.Tensor) -> float:
        return float(image.to(dtype=torch.bfloat16).float())

    image = torch.tensor(1.0)
    hidden = central_difference(rounded_identity, image, torch.tensor(1.0), 2.0**-10)
    visible = central_difference(rounded_identity, image, torch.tensor(1.0), BF16_ULP_AT_ONE)
    assert hidden == 0.0
    assert visible == 1.0
