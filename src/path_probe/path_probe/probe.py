"""Per-scene measurements. Model access goes through a ``Backend`` so the logic
can be tested with a toy model.

A backend provides:
- ``label``: the true color;
- ``grad_nll(image, ctx)``: the frozen labeled-color summed-NLL RGB gradient;
- ``grad_margin(image, ctx, comp)``: the RGB gradient of ``z_label - z_comp``;
- ``score_float(image, ctx)`` and ``score_official(image, ctx)``: dicts with
  ``nll``, ``margin``, ``chosen`` (bf16 logits) and ``nll32``, ``margin32``
  (fp32 projection).

The step rules are the frozen ones (``attack_feasibility.steps``).
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
from attack_feasibility.steps import matched_rademacher, pgd_update, project_signed_step
from qwen_rgb.constraints import round_trip_uint8

from . import geometry as geo

CONTEXTS = ("neutral", "helpful_nonrevealing")
LINE_FRACTIONS = (0.125, 0.25, 0.5, 0.75)
SCORE_KEYS = ("nll", "margin", "chosen", "nll32", "margin32")


def other(ctx: str) -> str:
    return CONTEXTS[1] if ctx == CONTEXTS[0] else CONTEXTS[0]


def _keep(score: dict[str, Any]) -> dict[str, Any]:
    return {k: score.get(k) for k in SCORE_KEYS}


def best_competitor(score: dict[str, Any], label: str) -> str:
    summed = score["summed_nll"]
    return min((c for c in summed if c != label), key=lambda c: summed[c])


ALL_GRADS = ("L_neutral", "M_neutral", "L_helpful_nonrevealing", "M_helpful_nonrevealing")


def clean_block(backend, x: torch.Tensor, grad_keys: tuple[str, ...] = ALL_GRADS) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    label = backend.label
    row: dict[str, Any] = {"kind": "clean", "scores": {}, "repeat_float": {}, "competitor": {}}
    grads: dict[str, torch.Tensor] = {}
    for ctx in CONTEXTS:
        first = backend.score_float(x, ctx)
        second = backend.score_float(x, ctx)
        row["scores"][ctx] = {"float": _keep(first), "official": _keep(backend.score_official(x, ctx))}
        row["repeat_float"][ctx] = {
            "nll_abs_diff": abs(first["nll"] - second["nll"]),
            "margin_abs_diff": abs(first["margin"] - second["margin"]),
            "nll32_abs_diff": abs(first["nll32"] - second["nll32"]),
        }
        comp = best_competitor(first, label)
        row["competitor"][ctx] = comp
        if f"L_{ctx}" in grad_keys:
            grads[f"L_{ctx}"] = backend.grad_nll(x, ctx)
        if f"M_{ctx}" in grad_keys:
            grads[f"M_{ctx}"] = backend.grad_margin(x, ctx, comp)
    row["grad_norms"] = {k: geo.norms(v) for k, v in grads.items()}
    if set(grad_keys) != set(ALL_GRADS):
        gn, gh = grads.get("L_neutral"), grads.get("L_helpful_nonrevealing")
        row["pairwise"] = {"cos_Ln_Lh": None if gn is None or gh is None else geo.cosine(gn, gh),
                           "cos_Ln_Mn": None if gn is None or "M_neutral" not in grads else geo.cosine(gn, grads["M_neutral"])}
        return row, grads
    gn, gh = grads["L_neutral"], grads["L_helpful_nonrevealing"]
    row["pairwise"] = {
        "cos_Ln_Lh": geo.cosine(gn, gh),
        "cos_Mn_Mh": geo.cosine(grads["M_neutral"], grads["M_helpful_nonrevealing"]),
        "cos_Ln_Mn": geo.cosine(gn, grads["M_neutral"]),
        "cos_Lh_Mh": geo.cosine(gh, grads["M_helpful_nonrevealing"]),
        "A_nh": geo.alignment(gh, gn),
        "A_hn": geo.alignment(gn, gh),
        "sign_agree_Ln_Lh": geo.sign_agreement(gn, gh),
        "sign_agree_Ln_Lh_weighted_by_Ln": geo.sign_agreement(gn, gh, gn),
    }
    return row, grads


def _predictions(grads: dict[str, torch.Tensor], delta: torch.Tensor) -> dict[str, Any]:
    out = {}
    for ctx in CONTEXTS:
        g_l = grads.get(f"L_{ctx}")
        g_m = grads.get(f"M_{ctx}")
        out[ctx] = {
            "nll": None if g_l is None else geo.dot(g_l, delta),
            # The margin falls when the attack works, so the predicted margin change is +g_M . delta.
            "margin": None if g_m is None else geo.dot(g_m, delta),
        }
    return out


def _point(backend, x, delta, grads, eps, ref, *, kind, official: bool, extra=None) -> dict[str, Any]:
    image = x + delta
    row: dict[str, Any] = {"kind": kind, "scores": {}, "pred_clean_gradient": _predictions(grads, delta),
                           "geometry": geo.delta_geometry(delta, eps, ref)}
    for ctx in CONTEXTS:
        row["scores"][ctx] = {"float": _keep(backend.score_float(image, ctx))}
        if official:
            row["scores"][ctx]["official"] = _keep(backend.score_official(image, ctx))
    if official:
        rounded = round_trip_uint8(image) - x
        row["geometry_rounded"] = geo.delta_geometry(rounded, eps, ref)
        row["pred_clean_gradient_rounded"] = _predictions(grads, rounded)
    if extra:
        row.update(extra)
    return row


def direction_block(
    backend,
    x: torch.Tensor,
    source: str,
    grads: dict[str, torch.Tensor],
    *,
    eps: float,
    steps: int,
    official_steps: tuple[int, ...],
    random_seed: int,
    clean_source_nll: float,
    line: bool = True,
    deadline: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """FGSM, random sign, line scan, and PGD from ``source``, scored under both contexts."""
    g0 = grads[f"L_{source}"]
    rows: list[dict[str, Any]] = []
    saved: dict[str, torch.Tensor] = {}
    fgsm = project_signed_step(x, g0, eps, "positive") - x
    saved["fgsm"] = geo.encode_delta(fgsm, eps)
    rows.append(_point(backend, x, fgsm, grads, eps, fgsm, kind="fgsm", official=True))
    signs = matched_rademacher(g0, random_seed)
    rnd = project_signed_step(x, g0, eps, "random", signs) - x
    rows.append(_point(backend, x, rnd, grads, eps, fgsm, kind="random", official=True, extra={"seed": random_seed}))
    if line:
        for t in LINE_FRACTIONS:
            d = t * fgsm
            rows.append(_point(backend, x, d, grads, eps, fgsm, kind="line", official=False, extra={"t": t}))
    current = x.detach().clone()
    g_prev = g0
    previous_delta = torch.zeros_like(x)
    previous_source = float(clean_source_nll)
    for k in range(1, steps + 1):
        if k > 1:
            if deadline is not None and deadline():
                rows.append({"kind": "pgd_truncated", "at_step": k, "reason": "time cap"})
                break
            started = time.perf_counter()
            g_k = backend.grad_nll(current, source)
            grad_row = {
                "kind": "pgd_gradient", "k": k - 1,
                "norms": geo.norms(g_k),
                "cos_g0": geo.cosine(g_k, g0),
                "cos_prev": geo.cosine(g_k, g_prev),
                "sign_agree_g0": geo.sign_agreement(g_k, g0),
                "sign_agree_g0_weighted": geo.sign_agreement(g_k, g0, g0),
                "seconds": round(time.perf_counter() - started, 3),
            }
            rows.append(grad_row)
            g_prev = g_k
        current = pgd_update(current, x, g_prev, eps / 4.0, eps)
        delta = current - x
        saved[f"pgd{k}"] = geo.encode_delta(delta, eps)
        local = geo.dot(g_prev, delta - previous_delta)
        row = _point(backend, x, delta, grads, eps, fgsm, kind="pgd", official=k in official_steps,
                     extra={"k": k, "local_pred_nll_step": local})
        src_nll = row["scores"][source]["float"]["nll"]
        row["realized_nll_step"] = src_nll - previous_source
        previous_source = src_nll
        previous_delta = delta
        rows.append(row)
    for r in rows:
        r["source"] = source
    return rows, saved


def cross_evaluate(backend, x, deltas: dict[str, torch.Tensor], grads, eps, *, crafted_in: str) -> list[dict[str, Any]]:
    """Score perturbations crafted in another precision mode. Float and official, both contexts."""
    rows = []
    ref = deltas.get("fgsm")
    for name, delta in deltas.items():
        rows.append(_point(backend, x, delta, grads, eps, ref, kind="cross", official=True,
                           extra={"crafted_in": crafted_in, "perturbation": name}))
    return rows
