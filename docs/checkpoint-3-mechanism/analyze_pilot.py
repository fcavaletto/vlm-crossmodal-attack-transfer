"""Checkpoint 3 pilot analysis (PLAN.md sections 5-6). EXPLORATORY, development data only.

Joins the pilot records with the 2.3 (FGSM, PGD-3) and 2.3b (top-|grad|, 8 random masks) official
scores of the same scenes, neutral source. Runs once on the complete pilot.

Run: python docs/checkpoint-3-mechanism/analyze_pilot.py [pilot_dir] [out.json]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = ROOT / "docs/checkpoint-2.3-exploratory"
SRC, TGT = "neutral", "helpful_nonrevealing"
B, SEED = 9999, 20261110


def boot(cols: np.ndarray, fn) -> dict:
    n = cols.shape[0]
    rng = np.random.default_rng(SEED)
    draws = np.array([fn(cols[i]) for i in rng.integers(0, n, size=(B, n))], dtype=float)
    est = float(fn(cols))
    ok = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if ok.size else (math.nan, math.nan))
    r = lambda v: round(float(v), 4) if math.isfinite(v) else None  # noqa: E731
    return {"n": int(n), "estimate": r(est), "ci95": [r(lo), r(hi)], "nonfinite_draws": int((~np.isfinite(draws)).sum())}


def mean_ci(v) -> dict:
    return boot(np.asarray(v, float)[:, None], lambda a: a[:, 0].mean())


def ratio_ci(num, den) -> dict:
    return boot(np.stack([num, den], 1).astype(float), lambda a: a[:, 0].mean() / a[:, 1].mean() if a[:, 1].mean() > 0 else math.nan)


def references(sids: set[str]) -> dict:
    ref = defaultdict(lambda: {"random": {}})
    for line in (DEV / "paths/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("mode") != "B" or r.get("scene_id") not in sids or r.get("source") != SRC:
            continue
        if r["kind"] == "fgsm":
            ref[r["scene_id"]]["fgsm"] = {c: r["scores"][c]["official"] for c in r["scores"]}
        elif r["kind"] == "pgd" and r.get("k") == 3:
            ref[r["scene_id"]]["pgd3"] = {c: r["scores"][c]["official"] for c in r["scores"]}
    for line in (DEV / "masks/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("scene_id") not in sids or r.get("kind") != "mask" or r["source"] != SRC:
            continue
        if r["mask"] == "topk_grad":
            ref[r["scene_id"]]["topk_grad"] = r["scores"]
        elif r["mask"] == "random":
            ref[r["scene_id"]]["random"][r["seed_index"]] = r["scores"]
    return ref


def analyze(pilot: Path) -> dict:
    rows = [json.loads(x) for x in (pilot / "records.jsonl").read_text().splitlines() if x.strip()]
    clean = {r["scene_id"]: r["scores"] for r in rows if r["kind"] == "clean"}
    sids = sorted(s for s in clean if any(r["kind"] == "structure" and r["scene_id"] == s for r in rows))
    ref = references(set(sids))
    grad_evals = {"fgsm": 1, "pgd3": 3, "topk_grad": 1, "random": 1}
    imgs: dict[str, dict[str, dict]] = defaultdict(dict)   # name -> sid -> scores
    meta: dict[str, dict[str, dict]] = defaultdict(dict)
    for s in sids:
        for name in ("fgsm", "pgd3", "topk_grad"):
            imgs[name][s] = ref[s][name]
        if sorted(ref[s]["random"]) != list(range(8)):
            raise SystemExit(f"{s}: the 2.3b random masks are incomplete")
    for r in rows:
        if r["kind"] in ("mask", "attack") and r["scene_id"] in sids:
            imgs[r["name"]][r["scene_id"]] = r["scores"]
            meta[r["name"]][r["scene_id"]] = r
            grad_evals[r["name"]] = r["grad_evals"]

    def D(s, scores, ctx):
        return clean[s][ctx]["margin"] - scores[ctx]["margin"]

    def flip(s, scores, ctx):
        return float(scores[ctx]["chosen"] != clean[s][ctx]["chosen"])

    out = {"schema": "checkpoint-3.pilot.analysis.v1", "status": "EXPLORATORY, development only",
           "n_scenes": len(sids), "scenes": sids, "bootstrap": {"draws": B, "seed": SEED}, "conditions": {}}
    per = {}
    for ctx, tag in ((SRC, "source"), (TGT, "target")):
        rand = np.array([np.mean([D(s, ref[s]["random"][i], ctx) for i in range(8)]) for s in sids])
        pgd_sel = np.array([D(s, imgs["pgd3"][s], ctx) for s in sids]) - rand
        per[tag] = {"rand": rand, "pgd_sel": pgd_sel}
    for name in imgs:
        if len(imgs[name]) != len(sids):
            continue
        c = {"grad_evals": grad_evals.get(name)}
        for ctx, tag in ((SRC, "source"), (TGT, "target")):
            d = np.array([D(s, imgs[name][s], ctx) for s in sids])
            sel = d - per[tag]["rand"]
            c[f"D_{tag}"] = mean_ci(d)
            c[f"sel_{tag}"] = mean_ci(sel)
            c[f"R_{tag}"] = ratio_ci(sel, per[tag]["pgd_sel"])
            c[f"flip_{tag}"] = round(float(np.mean([flip(s, imgs[name][s], ctx) for s in sids])), 4)
        if name in meta:
            m = [meta[name][s] for s in sids]
            if "overlap" in m[0]:
                c["overlap_minus_chance"] = mean_ci([x["overlap"] - x["chance"] for x in m])
                c["overlap"] = round(float(np.mean([x["overlap"] for x in m])), 4)
                c["chance"] = round(float(np.mean([x["chance"] for x in m])), 4)
                c["token_spearman_vs_pgd"] = mean_ci([x["token_spearman_vs_pgd"] for x in m])
                c["referent_enrichment"] = mean_ci([x["referent_enrichment"] for x in m])
            if "sign_agree_fgsm" in m[0]:
                c["sign_agree_fgsm"] = round(float(np.mean([x["sign_agree_fgsm"] for x in m])), 4)
            c["pred_clean_gradient_M_source"] = round(float(np.mean([-x["pred_clean_gradient"][f"M_{SRC}"] for x in m])), 3)
        out["conditions"][name] = c

    refm = defaultdict(list)
    for r in rows:
        if r["kind"] == "reference_mask" and r["scene_id"] in sids:
            refm[r["name"]].append(r)
    out["reference_masks_spatial"] = {
        n: {"overlap_minus_chance": mean_ci([x["overlap"] - x["chance"] for x in v]),
            "token_spearman_vs_pgd": mean_ci([x["token_spearman_vs_pgd"] for x in v]),
            "referent_enrichment": mean_ci([x["referent_enrichment"] for x in v])} for n, v in refm.items()}
    geo = defaultdict(list)
    for r in rows:
        if r["kind"] == "gradient" and r["scene_id"] in sids:
            geo[r["point"][0]].append(r)
    out["gradient_geometry"] = {k: {"n": len(v), "cos_g0": mean_ci([x["cos_g0"] for x in v]),
                                    "sign_agree_g0": mean_ci([x["sign_agree_g0"] for x in v]),
                                    "seconds_median": float(np.median([x["seconds"] for x in v]))} for k, v in geo.items()}
    stc = [r for r in rows if r["kind"] == "structure" and r["scene_id"] in sids]
    out["precision"] = {"fp32_fgsm_sign_agree_on_S_F": mean_ci([r["fp32_fgsm_sign_agree_on_S_F"] for r in stc]),
                        "k_median": float(np.median([r["k"] for r in stc]))}

    # Readings (PLAN.md section 6).
    cond = out["conditions"]
    def est(n, key="R_source"):
        return cond[n][key]["estimate"]
    R8 = cond["vote_R8"]["R_source"]
    lo, hi = R8["ci95"]
    a = (R8["estimate"] >= 0.8 and lo is not None and lo >= 0.5
         and cond["vote_R8"]["overlap_minus_chance"]["estimate"] >= 0.1 and est("vote_J2") <= 0.3)
    c_ = lo is None or hi is None or (lo <= 0 <= hi) or R8["estimate"] < 0.2
    reading = "A: consensus explains PGD selection" if a else ("C: consensus fails" if c_ else "B: consensus predicts part of the selection")
    pgd_patch = float(np.mean([est("pgd_patch_s0"), est("pgd_patch_s1")]))
    out["readings"] = {
        "reading": reading, "R_vote_R8": R8, "R_vote_R2_equal_budget": cond["vote_R2"]["R_source"],
        "R_vote_J2": cond["vote_J2"]["R_source"], "overlap_minus_chance_vote_R8": cond["vote_R8"]["overlap_minus_chance"],
        "qualifiers": {
            "spatial_pgd_patch_R_mean_of_2_seeds": round(pgd_patch, 4),
            "spatial_regional_if_ge_0.7": pgd_patch >= 0.7,
            "voteR8_patch_R": cond["voteR8_patch"]["R_source"],
            "quantized_vs_dense_R4": {"vote_Q4": est("vote_Q4"), "vote_R4": est("vote_R4"),
                                      "within_0.2": abs(est("vote_Q4") - est("vote_R4")) <= 0.2},
            "precision_vote_P32_R": cond["vote_P32"]["R_source"],
            "fp32vision_pgd_mask_overlap_minus_chance": out["reference_masks_spatial"].get("pgd3_fp32vision", {}).get("overlap_minus_chance"),
            "budget_curve_R": {n: est(n) for n in ("vote_R1", "vote_R2", "vote_R4", "vote_R8", "smooth_R2", "smooth_R8")},
            "exploit_D_source": {n: cond[n]["D_source"] for n in ("fgsm", "pgd3", "sfgsm_R2", "sfgsm_R8")},
        },
    }
    return out


def main() -> None:
    pilot = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pilot"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "analysis" / "pilot_analysis.json"
    out = analyze(pilot)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["readings"], indent=1))


if __name__ == "__main__":
    main()
