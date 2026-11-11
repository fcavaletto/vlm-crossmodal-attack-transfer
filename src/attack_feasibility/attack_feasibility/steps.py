"""Signed RGB steps inside the Checkpoint 1.8 feasible set.

The projection is ``qwen_rgb.constraints.project_linf``. This module does not
load a model and does not iterate.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from qwen_rgb.constraints import linf_distance, project_linf, round_trip_uint8
from qwen_rgb.finite_difference import rademacher, same_sign


def scene_seed(prefix: str, scene_id: str) -> int:
    """First 31 bits of SHA-256. ``hash()`` is not used."""
    digest = hashlib.sha256(f"{prefix}|{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def matched_rademacher(gradient: torch.Tensor, seed: int) -> torch.Tensor:
    """±1 on the support of ``sign(gradient)``, and 0 where the gradient is 0."""
    if not gradient.dtype.is_floating_point:
        raise TypeError("gradient must be floating point")
    direction = rademacher(tuple(gradient.shape), seed=int(seed))
    support = torch.sign(gradient.detach()).ne(0)
    return direction * support.to(dtype=direction.dtype)


def project_signed_step(
    clean: torch.Tensor,
    gradient: torch.Tensor,
    epsilon: float,
    mode: str,
    random_signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """One projected step. ``mode`` is ``positive``, ``negative``, or ``random``."""
    sign = torch.sign(gradient.detach())
    if mode == "positive":
        direction = sign
    elif mode == "negative":
        direction = -sign
    elif mode == "random":
        if random_signs is None:
            raise ValueError("a random step needs its matched signs")
        if tuple(random_signs.shape) != tuple(gradient.shape):
            raise ValueError("random signs do not match the gradient")
        direction = random_signs.detach()
    else:
        raise ValueError(f"unknown step mode {mode!r}")
    proposed = clean.detach() + float(epsilon) * direction
    return project_linf(proposed, clean.detach(), float(epsilon))


def pgd_update(
    current: torch.Tensor,
    clean: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    epsilon: float,
) -> torch.Tensor:
    """One ascent step of ``step_size``, then the L-infinity ball of radius ``epsilon``."""
    proposed = current.detach() + float(step_size) * torch.sign(gradient.detach())
    return project_linf(proposed, clean.detach(), float(epsilon))


def first_order_change(gradient: torch.Tensor, clean: torch.Tensor, projected: torch.Tensor) -> float:
    """``dot(gradient, projected - clean)`` in float64."""
    delta = projected.detach() - clean.detach()
    return float((gradient.detach().double() * delta.double()).sum().cpu())


def round_trip_metrics(clean: torch.Tensor, projected: torch.Tensor) -> dict[str, Any]:
    """Distances for the projected image and its official 8-bit round trip."""
    stored = round_trip_uint8(projected)
    changed = (stored - clean.detach()).abs().amax(dim=0).gt(0)
    return {
        "stored": stored,
        "linf_float": float(linf_distance(projected, clean)),
        "linf_stored_vs_clean": float(linf_distance(stored, clean)),
        "linf_stored_vs_projected": float(linf_distance(stored, projected)),
        "fraction_pixels_changed": float(changed.float().mean().cpu()),
    }


def directional_agreement(predicted: float, observed: float) -> bool:
    return bool(same_sign(float(predicted), float(observed)))
