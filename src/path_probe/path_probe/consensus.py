"""Checkpoint 3 pilot: gradient-consensus masks and their controls (docs/checkpoint-3-mechanism/PLAN.md).

Pure tensor code; tested on toy tensors. Every mask selects k coordinates of the FGSM support
S_F and is applied with FGSM's own rounded values r_F (``masks.apply``), so count and signs
match the PGD-3 control exactly.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

EPS = 1.0 / 255.0
PATCH = 16   # vision patch side in pixels
TOKEN = 32   # merged visual-token footprint (2 x 2 patches)


def _signs(shape, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 2, tuple(shape), generator=gen).to(torch.float32) * 2 - 1


def _keys(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).random(n)


def point_dense(x: torch.Tensor, scale: float, seed: int) -> torch.Tensor:
    """x + scale * sigma on every coordinate (sigma random +-1), clamped to [0, 1]."""
    return (x + scale * _signs(x.shape, seed)).clamp(0.0, 1.0)


def point_quantized(x: torch.Tensor, seed: int, density: float = 1.0 / 16.0) -> torch.Tensor:
    """x + EPS * sigma on a random `density` of coordinates: a valid 8-bit image when x is."""
    gen = torch.Generator().manual_seed(int(seed))
    sel = torch.rand(tuple(x.shape), generator=gen) < density
    sig = torch.randint(0, 2, tuple(x.shape), generator=gen).to(torch.float32) * 2 - 1
    return (x + EPS * sig * sel.to(torch.float32)).clamp(0.0, 1.0)


def _topk(score: np.ndarray, s_f: torch.Tensor, k: int) -> torch.Tensor:
    score = score.astype(np.float64).copy()
    score[~s_f.flatten().numpy()] = -np.inf
    order = np.argsort(-score, kind="stable")[:k]
    if k and not np.isfinite(score[order]).all():
        raise ValueError("k exceeds the FGSM support")
    m = torch.zeros(s_f.numel(), dtype=torch.bool)
    m[torch.from_numpy(order)] = True
    return m.view_as(s_f)


def agreement(r_f: torch.Tensor, grads: list[torch.Tensor]) -> torch.Tensor:
    """Number of gradients whose sign equals sign(r_F), per coordinate (0 gradient = disagree)."""
    s = torch.sign(r_f)
    return sum((torch.sign(g) == s).to(torch.int16) for g in grads)


def vote_mask(r_f: torch.Tensor, s_f: torch.Tensor, grads: list[torch.Tensor], k: int, seed: int) -> torch.Tensor:
    """Top-k coordinates of S_F by sign-agreement count; ties broken by a seeded uniform key."""
    a = agreement(r_f, grads).flatten().numpy().astype(np.float64)
    return _topk(a + _keys(a.size, seed), s_f, k)


def smooth_mask(r_f: torch.Tensor, s_f: torch.Tensor, grads: list[torch.Tensor], k: int) -> torch.Tensor:
    """Top-k coordinates of S_F by sign(r_F) * mean(grads) (SmoothGrad aligned with FGSM's sign)."""
    mean = torch.stack([g.to(torch.float64) for g in grads]).mean(0)
    return _topk((torch.sign(r_f).to(torch.float64) * mean).flatten().numpy(), s_f, k)


def patch_ids(shape, side: int) -> np.ndarray:
    """Patch index of every coordinate of a (C, H, W) tensor; all channels share a patch."""
    c, h, w = shape
    rows = np.arange(h) // side
    cols = np.arange(w) // side
    pid = rows[:, None] * ((w + side - 1) // side) + cols[None, :]
    return np.broadcast_to(pid, (c, h, w)).reshape(-1)


def patch_matched_random(mask: torch.Tensor, s_f: torch.Tensor, seed: int, side: int = PATCH) -> torch.Tensor:
    """Random coordinates of S_F with the same count as `mask` in every side x side patch."""
    m = (mask & s_f).flatten().numpy()
    sf = s_f.flatten().numpy()
    pid = patch_ids(tuple(mask.shape), side)
    need = np.bincount(pid[m], minlength=pid.max() + 1)
    idx = np.flatnonzero(sf)
    order = np.lexsort((_keys(idx.size, seed), pid[idx]))  # by patch, random within patch
    idx = idx[order]
    p = pid[idx]
    start = np.searchsorted(p, np.arange(pid.max() + 1))
    rank = np.arange(idx.size) - start[p]
    out = np.zeros(sf.size, dtype=bool)
    out[idx[rank < need[p]]] = True
    return torch.from_numpy(out).view_as(mask)


def overlap(a: torch.Tensor, p: torch.Tensor, s_f: torch.Tensor) -> dict[str, float]:
    k = int(a.sum())
    inter = int((a & p).sum())
    return {"overlap": inter / max(1, k), "chance": int(p.sum()) / max(1, int(s_f.sum()))}


def token_density(mask: torch.Tensor, side: int = TOKEN) -> np.ndarray:
    pid = patch_ids(tuple(mask.shape), side)
    return np.bincount(pid[mask.flatten().numpy()], minlength=pid.max() + 1).astype(np.float64)


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / counts[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


QUESTION = re.compile(r"What color is the (\w+) (\w+) in the (top|bottom) row\?")


def referent(record: dict[str, Any]) -> dict[str, Any]:
    """The unique object the question names (size, category, row)."""
    m = QUESTION.fullmatch(record["question"].strip())
    if not m:
        raise ValueError(f"unparsed question: {record['question']}")
    size, cat, row = m.group(1), m.group(2), 0 if m.group(3) == "top" else 1
    hits = [o for o in record["objects"] if o["attributes"]["size"] == size and o["category"] == cat and o["position"]["row"] == row]
    if len(hits) != 1 or hits[0]["attributes"]["color"] != record["answer"]:
        raise ValueError(f"referent not unique or color mismatch in {record['scene_id']}")
    return hits[0]


def referent_enrichment(mask: torch.Tensor, ref: dict[str, Any]) -> float:
    """Mask density inside the referent's grid cell over density in the whole image."""
    c, h, w = mask.shape
    cw, ch = w // 4, h // 2
    col, row = ref["position"]["col"], ref["position"]["row"]
    cell = mask[:, row * ch:(row + 1) * ch, col * cw:(col + 1) * cw]
    whole = float(mask.float().mean())
    return float(cell.float().mean()) / whole if whole > 0 else float("nan")


def geometry(g: torch.Tensor, g0: torch.Tensor) -> dict[str, float]:
    a, b = g.flatten().to(torch.float64), g0.flatten().to(torch.float64)
    cos = float(a @ b / (a.norm() * b.norm()))
    nz = (a != 0) & (b != 0)
    return {"cos_g0": cos, "sign_agree_g0": float((torch.sign(a[nz]) == torch.sign(b[nz])).to(torch.float64).mean())}
