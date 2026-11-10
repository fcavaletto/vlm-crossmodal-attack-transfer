"""Gradient diagnostics on the surrogate path. A large finite-difference error is a prompt to
investigate (step size, kinks, preprocessing, precision), not by itself a verdict."""

from __future__ import annotations

from typing import Any

import torch

from .pixels import surrogate_pixels

STEPS_LEVELS = (1 / 16, 1 / 4, 1.0, 4.0)  # in 8-bit levels; eps = 1 level in this study
SMALL_STEPS_LEVELS = (1 / 1024, 1 / 256, 1 / 64, 1 / 16)  # follow-up: step-size dependence (fp32)
N_DIRECTIONS = 4
SEED = 20261110


def _rel(fd: float, an: float) -> float:
    return abs(fd - an) / max(abs(fd) + abs(an), 1e-12)


def kinks(image01: torch.Tensor, tiling: str) -> dict[str, float]:
    """Where the objective is not differentiable: saturated RGB values and clamped resized values."""
    x = image01
    pre = surrogate_pixels(x, tiling, clamp=False)
    post = surrogate_pixels(x, tiling, clamp=True)
    return {"frac_rgb_at_0_or_1": float(((x <= 0) | (x >= 1)).float().mean()),
            "frac_resized_values_clamped": float((pre != post).float().mean())}


def validate(m, image01: torch.Tensor, prompt, label: str, monitor, steps=STEPS_LEVELS) -> tuple[dict[str, Any], torch.Tensor]:
    gen = torch.Generator().manual_seed(SEED)
    f0, g = m.gradient(image01, prompt, label, "nll")
    f0b = m.objective_value(image01, prompt, label, "nll")
    _, g_rep = m.gradient(image01, prompt, label, "nll")
    monitor.check()
    out: dict[str, Any] = {
        "objective_with_grad": f0, "objective_no_grad": f0b, "objective_grad_vs_nograd_abs": abs(f0 - f0b),
        "gradient_repeat_bit_identical": bool(torch.equal(g, g_rep)),
        "gradient_finite": bool(torch.isfinite(g).all()), "gradient_l2": float(g.norm()), "gradient_l1": float(g.abs().sum()),
        "frac_zero": float((g == 0).float().mean()), "kinks": kinks(image01, m.tiling),
    }
    # random dense directions, central differences at several step sizes
    dirs = []
    for k in range(N_DIRECTIONS):
        u = torch.randint(0, 2, image01.shape, generator=gen).float().mul(2).sub(1)
        an = float((g * u).sum())
        per_h = []
        for hl in steps:
            h = hl / 255
            fp = m.objective_value(image01 + h * u, prompt, label)
            fm = m.objective_value(image01 - h * u, prompt, label)
            fd = (fp - fm) / (2 * h)
            per_h.append({"h_levels": hl, "fd": fd, "plus": fp - f0b, "minus": fm - f0b, "rel_err": _rel(fd, an),
                          "asymmetry": abs((fp - f0b) + (fm - f0b)) / max(abs(fp - fm), 1e-12)})
            monitor.check()
        dirs.append({"analytic": an, "steps": per_h, "best_rel_err": min(s["rel_err"] for s in per_h)})
    out["random_directions"] = dirs
    # along the attack's own direction (ascent of the NLL), one-sided: the line the FGSM step follows
    s = torch.sign(g)
    line = []
    for hl in (1 / 4, 1.0):
        h = hl / 255
        ff = m.objective_value((image01 + h * s).clamp(0, 1), prompt, label)
        line.append({"h_levels": hl, "actual": ff - f0b, "first_order": float(g.abs().sum()) * h,
                     "ratio_actual_to_first_order": (ff - f0b) / (float(g.abs().sum()) * h)})
    out["fgsm_line_surrogate"] = line
    # the official 8-bit path for the same one-level FGSM step
    off0 = m.score(image01, prompt, label, "official")
    off1 = m.score((image01 + s / 255).clamp(0, 1), prompt, label, "official")
    out["fgsm_one_level_official"] = {"nll_change": off1["label_nll"] - off0["label_nll"],
                                      "margin_change": off1["margin"] - off0["margin"],
                                      "first_order": float(g.abs().sum()) / 255}
    return out, g
