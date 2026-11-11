"""Vector geometry for perturbations and gradients. Pure torch, no model.

Every reduction is in float64 so that the reported cosines and dot products do
not inherit bf16 or float32 rounding from the tensors they summarize.
"""

from __future__ import annotations

from typing import Any

import torch


def _d(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(dtype=torch.float64, device="cpu").reshape(-1)


def dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((_d(a) * _d(b)).sum())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    x, y = _d(a), _d(b)
    nx, ny = float(x.norm()), float(y.norm())
    if nx == 0.0 or ny == 0.0:
        return None
    return float((x * y).sum()) / (nx * ny)


def norms(g: torch.Tensor) -> dict[str, float]:
    x = _d(g)
    return {"l1": float(x.abs().sum()), "l2": float(x.norm()), "linf": float(x.abs().max())}


def sign_agreement(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor | None = None) -> float | None:
    """Share of coordinates with equal sign, among coordinates where both are nonzero.

    With ``weight``, each coordinate counts by ``|weight|``.
    """
    x, y = _d(a), _d(b)
    both = (x != 0) & (y != 0)
    if not bool(both.any()):
        return None
    same = (torch.sign(x) == torch.sign(y)) & both
    if weight is None:
        return float(same.sum()) / float(both.sum())
    w = _d(weight).abs()
    total = float((w * both).sum())
    if total == 0.0:
        return None
    return float((w * same).sum()) / total


def alignment(g_target: torch.Tensor, g_source: torch.Tensor) -> float | None:
    """A(source, target) = g_target' sign(g_source) / ||g_target||_1, the Checkpoint 1.8 definition."""
    t, s = _d(g_target), _d(g_source)
    l1 = float(t.abs().sum())
    if l1 == 0.0:
        return None
    return float((t * torch.sign(s)).sum()) / l1


def delta_geometry(delta: torch.Tensor, epsilon: float, reference: torch.Tensor | None = None) -> dict[str, Any]:
    """Size and shape of a perturbation, and its relation to a reference (FGSM)."""
    x = _d(delta)
    tol = 1e-4 * float(epsilon)
    out: dict[str, Any] = {
        "l1": float(x.abs().sum()),
        "l2": float(x.norm()),
        "linf": float(x.abs().max()),
        "frac_at_eps": float((x.abs() >= float(epsilon) - tol).double().mean()),
        "frac_zero": float((x.abs() <= tol).double().mean()),
    }
    if reference is not None:
        r = _d(reference)
        out["cos_ref"] = cosine(x, r)
        agree = sign_agreement(torch.where(x.abs() > tol, x, torch.zeros_like(x)),
                               torch.where(r.abs() > tol, r, torch.zeros_like(r)))
        out["sign_disagreement_ref"] = None if agree is None else 1.0 - agree
    return out


def encode_delta(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    """int8 multiples of epsilon/4. PGD iterates and FGSM are exact on this grid."""
    units = delta.detach().to(torch.float64) / (float(epsilon) / 4.0)
    rounded = units.round()
    if float((units - rounded).abs().max()) > 1e-3:
        raise ValueError("delta is not on the epsilon/4 grid")
    return rounded.clamp(-8, 8).to(torch.int8)


def decode_delta(code: torch.Tensor, epsilon: float) -> torch.Tensor:
    return code.to(torch.float32) * (float(epsilon) / 4.0)


def interpolate_at(xs: list[float], ys: list[float], x0: float) -> float | None:
    """Linear interpolation of y at x0 along a path. None if x0 is outside the path's range.

    ``xs`` need not be monotone. The first segment that brackets x0 is used.
    """
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if (a - x0) * (b - x0) <= 0 and a != b:
            w = (x0 - a) / (b - a)
            return ys[i] + w * (ys[i + 1] - ys[i])
        if a == b == x0:
            return ys[i]
    return None
