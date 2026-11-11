"""Checkpoint 3b partition-and-union pilot (docs/checkpoint-3b-partition/PLAN.md). Pure code, no model.

Masks, quarter splits and structure checks (tensor code), and the scoring and analysis of the
records (plain Python and numpy). The runner is ``partition_stage``.

Score conventions, from the frozen scorer (``attack_feasibility.score.score_official``):
- ``summed_nll[c]`` is -log softmax(z)[token of c] over the full vocabulary, in nats. Differences
  between colours are logit differences, because the normaliser cancels.
- The official margin is ``qwen_rgb_loss.records.clean_margin``: min over c != label of s_c, minus s_label.
- The decision is ``choice_eval.decision.choose_color``: lowest score; equal scores break by colour name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

N_QUARTERS = 4
N_RANDOM = 8
N_PGD_SPLITS = 2
BOOT_DRAWS = 9999
BOOT_SEED = 20261111
IDENTITY_TOL = 1e-9
CHANNELS = ("fixed", "switching", "margin")
CONTEXT = "neutral"
COLORS = ("red", "blue", "green", "yellow")


class PartitionError(ValueError):
    pass


# ---------------------------------------------------------------- masks and splits (tensor code)

def quarter_sizes(k: int) -> list[int]:
    """Sizes of the four pieces of a k-coordinate mask: floor(k/4), the first k mod 4 pieces one larger."""
    if k < N_QUARTERS:
        raise PartitionError(f"k = {k} is too small to split into {N_QUARTERS} non-empty pieces")
    return [k // N_QUARTERS + (1 if j < k % N_QUARTERS else 0) for j in range(N_QUARTERS)]


def split(mask, seed: int) -> list:
    """Four disjoint pieces of ``mask``: a seeded uniform random order of its coordinates, cut into
    consecutive blocks of ``quarter_sizes(k)``."""
    import torch

    idx = mask.flatten().nonzero().squeeze(1)
    sizes = quarter_sizes(int(idx.numel()))
    gen = torch.Generator().manual_seed(int(seed))
    order = idx[torch.randperm(idx.numel(), generator=gen)]
    pieces, start = [], 0
    for n in sizes:
        m = torch.zeros(mask.numel(), dtype=torch.bool)
        m[order[start:start + n]] = True
        pieces.append(m.view_as(mask))
        start += n
    return pieces


def check_split(mask, pieces) -> None:
    """Disjoint, union equal to the mask, planned sizes."""
    import torch

    k = int(mask.sum())
    if [int(p.sum()) for p in pieces] != quarter_sizes(k):
        raise PartitionError("piece sizes differ from quarter_sizes(k)")
    total = torch.zeros_like(mask, dtype=torch.int16)
    for p in pieces:
        if bool((p & ~mask).any()):
            raise PartitionError("a piece leaves its mask")
        total += p.to(torch.int16)
    if int(total.max()) > 1:
        raise PartitionError("pieces overlap")
    if not torch.equal(total.bool(), mask):
        raise PartitionError("the union of the pieces is not the mask")


def sign_counts(r_f, mask) -> dict[str, int]:
    """Coordinates in the mask, and how many of them FGSM moves up and down."""
    vals = r_f[mask]
    return {"count": int(mask.sum()), "n_pos": int((vals > 0).sum()), "n_neg": int((vals < 0).sum())}


def pgd_structure(r_f, r_p, level: float) -> dict[str, Any]:
    """The retained overlap P = S_F & S_P, the excluded tie coordinates S_P minus S_F, and the sign check.

    Raises if the rounded PGD perturbation and FGSM disagree in sign anywhere on the overlap, or if
    PGD moves any coordinate by more than one 8-bit level."""
    import torch

    s_f = r_f.abs() > 0.5 * level
    s_p = r_p.abs() > 0.5 * level
    P = s_f & s_p
    disagree = int((torch.sign(r_f[P]) != torch.sign(r_p[P])).sum())
    levels_max = int(torch.round(r_p.abs() / level).max()) if r_p.numel() else 0
    out = {"n_fgsm_support": int(s_f.sum()), "n_pgd_support": int(s_p.sum()), "k_retained": int(P.sum()),
           "n_excluded_outside_fgsm": int((s_p & ~s_f).sum()), "sign_disagreements_on_overlap": disagree,
           "pgd_levels_max": levels_max}
    if disagree:
        raise PartitionError(f"rounded PGD and FGSM disagree in sign on {disagree} retained coordinates")
    if levels_max > 1:
        raise PartitionError(f"rounded PGD moves a coordinate by {levels_max} levels")
    return {**out, "P": P, "S_F": s_f}


def image_change_check(x, y, r_f, mask, level: float) -> None:
    """The stored 8-bit image of y differs from x by exactly FGSM's one-level sign on the mask, and nowhere else."""
    import torch

    from qwen_rgb.constraints import round_trip_uint8

    step = torch.round(round_trip_uint8(y) * 255) - torch.round(x * 255)
    want = torch.where(mask, torch.round(r_f / level), torch.zeros_like(r_f))
    if not torch.equal(step, want):
        raise PartitionError("the 8-bit image does not equal x plus FGSM's level on the mask")


# ---------------------------------------------------------------- four colour scores

def competitor(summed: Mapping[str, float], label: str) -> str:
    """Lowest-scoring colour other than the label; equal scores break by colour name (the decision's rule)."""
    others = [c for c in COLORS if c != label]
    return min(others, key=lambda c: (float(summed[c]), c))


def best_rival_score(summed: Mapping[str, float], label: str) -> float:
    return min(float(summed[c]) for c in COLORS if c != label)


def identity_check(score: Mapping[str, Any], label: str) -> None:
    """The stored margin and decision are the frozen functions of the stored four scores."""
    from choice_eval.decision import choose_color
    from qwen_rgb_loss.records import clean_margin

    summed = {c: float(score["summed_nll"][c]) for c in COLORS}
    if set(score["summed_nll"]) != set(COLORS):
        raise PartitionError("summed_nll must hold exactly the four colours")
    if float(score["margin"]) != clean_margin(summed, label):
        raise PartitionError("stored margin != min rival score - label score")
    if score["chosen"] != choose_color(summed):
        raise PartitionError("stored decision != lowest score with the name tie rule")


def decompose(clean: Mapping[str, Any], image: Mapping[str, Any], label: str) -> dict[str, float]:
    """Damage of one image on the fixed clean-competitor contrast, its switching term, and the official margin.

    f(y) = s_c0(y) - s_label(y); W(y) = s_c0(y) - min_{c != label} s_c(y) >= 0; m(y) = f(y) - W(y).
    D_fixed = f(x) - f(y); D_margin = m(x) - m(y) = D_fixed + W(y), because W(x) = 0."""
    c0 = competitor(clean["summed_nll"], label)
    s0, s = clean["summed_nll"], image["summed_nll"]
    f0 = float(s0[c0]) - float(s0[label])
    f = float(s[c0]) - float(s[label])
    w0 = float(s0[c0]) - best_rival_score(s0, label)
    w = float(s[c0]) - best_rival_score(s, label)
    if w0 != 0.0 or w < 0.0:
        raise PartitionError("switching term must be 0 on the clean image and >= 0 elsewhere")
    d_fixed = f0 - f
    d_margin = float(clean["margin"]) - float(image["margin"])
    if abs(d_margin - (d_fixed + w)) > IDENTITY_TOL:
        raise PartitionError(f"D_margin {d_margin} != D_fixed {d_fixed} + W {w}")
    return {"fixed": d_fixed, "switching": w, "margin": d_margin,
            "rival_switch": competitor(s, label) != c0, "label_ties_rival": float(s[label]) == best_rival_score(s, label)}


# ---------------------------------------------------------------- per scene

def _expected_keys() -> set[tuple[str, int, str]]:
    keys = {("pgd", 0, "full")}
    keys |= {("pgd", a, str(j)) for a in range(N_PGD_SPLITS) for j in range(N_QUARTERS)}
    keys |= {("random", s, p) for s in range(N_RANDOM) for p in ["full", *map(str, range(N_QUARTERS))]}
    return keys


def scene_quantities(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """A, Q, Delta and each family's full-minus-sum-of-quarters, per channel, for one scene.

    Rows are the runner's ``image`` rows of one scene: one clean row and the 50 masked images.
    Random seeds are averaged within the scene; PGD's two splits are averaged within the scene."""
    rows = list(rows)
    sids = {r["scene_id"] for r in rows}
    if len(sids) != 1:
        raise PartitionError("rows from more than one scene")
    clean = [r for r in rows if r["family"] == "clean"]
    if len(clean) != 1:
        raise PartitionError("need exactly one clean row")
    label = clean[0]["label"]
    cs = clean[0]["scores"][CONTEXT]
    identity_check(cs, label)
    imgs: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for r in rows:
        if r["family"] not in ("pgd", "random"):
            continue
        key = (r["family"], int(r["draw"]), str(r["piece"]))
        if key in imgs:
            raise PartitionError(f"duplicate image {key}")
        imgs[key] = r
    if set(imgs) != _expected_keys():
        missing = sorted(_expected_keys() - set(imgs))
        raise PartitionError(f"incomplete scene: missing {missing[:3]}{'...' if len(missing) > 3 else ''}")
    # Matching: every full mask has the same count; piece j has the same size in every split and family.
    k = imgs[("pgd", 0, "full")]["count"]
    sizes = quarter_sizes(k)
    for (fam, d, p), r in imgs.items():
        want = k if p == "full" else sizes[int(p)]
        if r["count"] != want or r["n_pos"] + r["n_neg"] != r["count"]:
            raise PartitionError(f"{fam} {d} {p}: count {r['count']} != planned {want}")
        if r["label"] != label:
            raise PartitionError("label differs between rows")
    dmg = {}
    for key, r in imgs.items():
        identity_check(r["scores"][CONTEXT], label)
        dmg[key] = decompose(cs, r["scores"][CONTEXT], label)

    out: dict[str, Any] = {"scene_id": sids.pop(), "label": label, "clean_competitor": competitor(cs["summed_nll"], label),
                           "clean_margin": float(cs["margin"]), "k": k, "quarter_sizes": sizes}
    for ch in CHANNELS:
        def d(key):
            return dmg[key][ch]
        pgd_full = d(("pgd", 0, "full"))
        pgd_sum = sum(sum(d(("pgd", a, str(j))) for j in range(N_QUARTERS)) for a in range(N_PGD_SPLITS)) / N_PGD_SPLITS
        rnd_full = sum(d(("random", s, "full")) for s in range(N_RANDOM)) / N_RANDOM
        rnd_sum = sum(sum(d(("random", s, str(j))) for j in range(N_QUARTERS)) for s in range(N_RANDOM)) / N_RANDOM
        A = pgd_full - rnd_full
        Q = pgd_sum - rnd_sum
        out[ch] = {"A": A, "Q": Q, "Delta": A - Q,
                   "pgd_full": pgd_full, "pgd_sum_quarters": pgd_sum, "pgd_full_minus_sum": pgd_full - pgd_sum,
                   "random_full": rnd_full, "random_sum_quarters": rnd_sum, "random_full_minus_sum": rnd_full - rnd_sum}
    for q in out["fixed"]:
        if abs(out["margin"][q] - (out["fixed"][q] + out["switching"][q])) > 1e-6:
            raise PartitionError(f"switching identity fails for {q}")
    out["rival_switches"] = {fam: sum(1 for (f, _, _), v in dmg.items() if f == fam and v["rival_switch"]) for fam in ("pgd", "random")}
    out["label_rival_ties"] = sum(1 for v in dmg.values() if v["label_ties_rival"])
    rivals = sorted(float(cs["summed_nll"][c]) for c in COLORS if c != label)
    out["clean_rival_tie"] = rivals[0] == rivals[1]  # c0 then comes from the name rule
    out["sign_counts"] = {f"{f}|{d_}|{p}": [r["n_pos"], r["n_neg"]] for (f, d_, p), r in sorted(imgs.items())}
    return out


# ---------------------------------------------------------------- across scenes

def _ci(draws, alpha: float = 0.05) -> list[float]:
    import numpy as np

    return [float(np.percentile(draws, 100 * alpha / 2)), float(np.percentile(draws, 100 * (1 - alpha / 2)))]


def direction(ci: list[float]) -> str:
    if ci[0] > 0:
        return "positive"
    if ci[1] < 0:
        return "negative"
    return "unresolved"


QUANTITIES = ("A", "Q", "Delta", "pgd_full_minus_sum", "random_full_minus_sum")


def panel(per_scene: list[Mapping[str, Any]], planned: list[str], *, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Across-scene means with a paired percentile scene bootstrap (same resampled scenes for every quantity).

    Refuses an incomplete panel: the planned analysis is defined only for all planned scenes."""
    import numpy as np

    got = [s["scene_id"] for s in per_scene]
    if got != list(planned):
        raise PartitionError(f"incomplete or reordered panel: {len(got)} of {len(planned)} planned scenes")
    n = len(per_scene)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    out: dict[str, Any] = {"n_scenes": n, "bootstrap": {"draws": draws, "seed": seed, "method": "percentile, scenes resampled with replacement, same draws for all quantities"}}
    for ch in CHANNELS:
        out[ch] = {}
        for q in QUANTITIES:
            v = np.array([s[ch][q] for s in per_scene], dtype=float)
            boot = v[idx].mean(axis=1)
            ci = _ci(boot)
            out[ch][q] = {"mean": float(v.mean()), "ci95": ci, "direction": direction(ci),
                          "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum())}
    # rho = mean(Q) / mean(A), fixed contrast, secondary.
    a = np.array([s["fixed"]["A"] for s in per_scene], dtype=float)
    qv = np.array([s["fixed"]["Q"] for s in per_scene], dtype=float)
    a_boot, q_boot = a[idx].mean(axis=1), qv[idx].mean(axis=1)
    a_ci = out["fixed"]["A"]["ci95"]
    if a_ci[0] > 0 and bool((a_boot > 0).all()):
        out["rho"] = {"estimate": float(qv.mean() / a.mean()), "ci95": _ci(q_boot / a_boot), "defined": True}
    else:
        out["rho"] = {"defined": False, "reason": "denominator mean(A) not bounded away from 0 (95% CI reaches 0 or a bootstrap draw <= 0)"}
    return out
