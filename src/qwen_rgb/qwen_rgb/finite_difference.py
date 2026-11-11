"""Directional finite-difference check, with the BF16 step made explicit.

The analytic gradient of a BF16 model is a straight-through gradient: the cast
inside the vision tower is treated as the identity in backward. The finite
difference sees the rounded forward. A step smaller than the BF16 unit in the
last place at magnitude 1 can be invisible. The steps below were chosen before
the Qwen run and are not revised to make that run pass.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# BF16 has an 8-bit exponent and 7 explicit mantissa bits. At magnitude 1 the
# unit in the last place is 2^-7.
BF16_ULP_AT_ONE = 2.0**-7
PRIMARY_DELTAS = (1.0 / 64.0, 1.0 / 128.0)
BF16_RELATIVE_ERROR_CEILING = 0.5
FLOAT32_RELATIVE_ERROR_CEILING = 0.05
DIRECTION_SEED = 17


def rademacher(shape: tuple[int, ...], *, seed: int = DIRECTION_SEED) -> torch.Tensor:
    """Deterministic direction with entries +1 and -1. The step multiplies this."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    return bits.mul(2).sub(1).to(dtype=torch.float32)


def central_difference(
    loss_fn: Callable[[torch.Tensor], float],
    image: torch.Tensor,
    direction: torch.Tensor,
    delta: float,
) -> float:
    """[L(x + δv) - L(x - δv)] / (2δ). The loss function must not build a graph."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    if tuple(direction.shape) != tuple(image.shape):
        raise ValueError("direction must have the same shape as the image")
    higher = float(loss_fn(image + delta * direction))
    lower = float(loss_fn(image - delta * direction))
    return (higher - lower) / (2.0 * delta)


def analytic_directional(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    """Inner product of the RGB gradient with the finite-difference direction."""
    if tuple(gradient.shape) != tuple(direction.shape):
        raise ValueError("gradient and direction shapes differ")
    return float((gradient.detach().float() * direction.float()).sum())


def relative_error(analytic: float, estimate: float) -> float:
    """|a - b| / max(|a|, |b|, BF16 ulp at 1). The ulp keeps a zero from dividing."""
    gap = abs(analytic - estimate)
    denominator = max(abs(analytic), abs(estimate), BF16_ULP_AT_ONE)
    return gap / denominator


def same_sign(analytic: float, estimate: float) -> bool:
    """True when the two estimates do not point in opposite directions."""
    if analytic == 0.0 or estimate == 0.0:
        return analytic == estimate
    return analytic * estimate > 0.0


def judge_samples(
    analytic: float,
    samples: list[tuple[float, float]],
    *,
    ceiling: float,
) -> dict[str, object]:
    """Apply the precommitted ceiling. One agreeing step is enough to confirm."""
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")
    rows = []
    confirmed = False
    for delta, estimate in samples:
        error = relative_error(analytic, estimate)
        agrees = same_sign(analytic, estimate)
        passed = agrees and error <= ceiling
        confirmed = confirmed or passed
        rows.append(
            {
                "delta": delta,
                "finite_difference": estimate,
                "relative_error": error,
                "same_sign": agrees,
                "within_ceiling": passed,
            }
        )
    return {
        "analytic_directional": analytic,
        "relative_error_ceiling": ceiling,
        "confirmed": confirmed,
        "samples": rows,
    }


def ceiling_for_dtype(dtype_name: str) -> float:
    """BF16 uses the wide ceiling. Float32 uses the tighter one. Both are precommitted."""
    if dtype_name == "bfloat16":
        return BF16_RELATIVE_ERROR_CEILING
    if dtype_name == "float32":
        return FLOAT32_RELATIVE_ERROR_CEILING
    raise ValueError(f"No finite-difference ceiling is defined for dtype {dtype_name}")
