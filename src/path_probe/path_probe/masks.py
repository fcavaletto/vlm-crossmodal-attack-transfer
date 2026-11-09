"""Masked-FGSM perturbations for the Checkpoint 2.3b control (MASK_CONTROL_PLAN.md). Pure tensor code."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from qwen_rgb.constraints import round_trip_uint8

LEVEL = 1.0 / 255.0


def rounded_delta(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The change the official 8-bit round trip actually applies."""
    return round_trip_uint8(x + delta) - x


def support(r: torch.Tensor) -> torch.Tensor:
    return r.abs() > 0.5 * LEVEL


def structure(r_f: torch.Tensor, r_p: torch.Tensor) -> dict[str, Any]:
    """Is the rounded PGD perturbation a sign-preserving subset of rounded FGSM?"""
    s_f, s_p = support(r_f), support(r_p)
    both = s_f & s_p
    levels = torch.round(r_p.abs() / LEVEL)
    return {
        "n_fgsm_support": int(s_f.sum()),
        "n_pgd_support": int(s_p.sum()),
        "n_elements": int(r_f.numel()),
        "density_in_fgsm_support": float(s_p.sum()) / max(1, int(s_f.sum())),
        "pgd_outside_fgsm": int((s_p & ~s_f).sum()),
        "sign_reversals": int((both & (torch.sign(r_f) != torch.sign(r_p))).sum()),
        "pgd_levels_max": int(levels.max()),
        "is_sign_preserving_subset": bool(int((s_p & ~s_f).sum()) == 0
                                          and int((both & (torch.sign(r_f) != torch.sign(r_p))).sum()) == 0
                                          and int(levels.max()) <= 1),
    }


def random_mask(s_f: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    idx = s_f.flatten().nonzero().squeeze(1)
    if k > idx.numel():
        raise ValueError("k exceeds the FGSM support")
    gen = torch.Generator().manual_seed(int(seed))
    pick = idx[torch.randperm(idx.numel(), generator=gen)[:k]]
    m = torch.zeros(s_f.numel(), dtype=torch.bool)
    m[pick] = True
    return m.view_as(s_f)


def topk_mask(s_f: torch.Tensor, g: torch.Tensor, k: int) -> torch.Tensor:
    score = g.detach().abs().flatten().to(torch.float64).numpy().copy()
    score[~s_f.flatten().numpy()] = -np.inf
    order = np.argsort(-score, kind="stable")[:k]
    if k and not np.isfinite(score[order]).all():
        raise ValueError("k exceeds the FGSM support")
    m = torch.zeros(s_f.numel(), dtype=torch.bool)
    m[torch.from_numpy(order)] = True
    return m.view_as(s_f)


def apply(x: torch.Tensor, r_f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x + r_f * mask.to(r_f.dtype)


def replay_pgd(x: torch.Tensor, codes: dict[str, torch.Tensor], steps: int, epsilon: float) -> torch.Tensor:
    """Rebuild the float PGD iterate with the frozen update, from the stored int8 iterates.

    x + decode(pgd_k) is not bit-identical to the accumulated float iterate, and at
    half-level ties (a clipped pixel moved by eps/2) the 8-bit rounding can differ.
    Step k's sign is the sign of (pgd_k - pgd_{k-1}); it is 0 only where the step was
    clipped at 0 or 1, where a zero step leaves the same float value.
    """
    from attack_feasibility.steps import pgd_update

    current = x.detach().clone()
    previous = torch.zeros_like(codes["pgd1"])
    for k in range(1, steps + 1):
        code = codes[f"pgd{k}"]
        sigma = torch.sign((code.to(torch.int16) - previous.to(torch.int16)).to(torch.float32))
        current = pgd_update(current, x, sigma, epsilon / 4.0, epsilon)
        previous = code
    return current
