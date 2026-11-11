"""Checkpoint 2.4 confirmation analysis (PROTOCOL.md sections 5-7). Frozen before any inference
on the confirmation split. Reads records only; runs once on the complete table.

Run:  python docs/checkpoint-2.4-confirm/analyze_confirm.py [records.jsonl] [out.json]
Default input  docs/checkpoint-2.4-confirm/run/records.jsonl
Default output docs/checkpoint-2.4-confirm/analysis/confirm_analysis.json (+ per_group.csv)
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CTX = ("neutral", "helpful_nonrevealing")
OTHER = {CTX[0]: CTX[1], CTX[1]: CTX[0]}
B, SEED = 9999, 20261116
N_SEEDS = 8
SHARE_MAX = 0.25


def boot(per_unit: np.ndarray, fn) -> dict:
    """Percentile bootstrap over groups (rows). Non-finite draws are kept as +inf (they count against S1)."""
    n = per_unit.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.array([fn(per_unit[i]) for i in idx], dtype=float)
    draws[~np.isfinite(draws)] = np.inf
    est = float(fn(per_unit))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n": int(n), "estimate": round(est, 5) if math.isfinite(est) else None,
            "ci95": [round(float(lo), 5) if math.isfinite(lo) else None, round(float(hi), 5) if math.isfinite(hi) else None],
            "nonfinite_draws": int((~np.isfinite(draws)).sum())}


def mean_ci(v: np.ndarray) -> dict:
    return {"mean": boot(v, np.mean), "median": round(float(np.median(v)), 5)}


def share_fn(a: np.ndarray) -> float:
    num = a[:, 1].mean() - a[:, 0].mean()
    den = a[:, 2].mean() - a[:, 0].mean()
    return num / den if den > 0 else math.inf


def load(path: Path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    clean, label, units, structure = {}, {}, defaultdict(dict), []
    for r in rows:
        if r["kind"] == "clean":
            clean[r["scene_id"]] = {c: r["scores"][c]["official"] for c in CTX}
            label[r["scene_id"]] = r["label"]
        elif r["kind"] == "structure":
            structure.append(r)
        elif r["kind"] in ("fgsm", "pgd3", "topk_grad"):
            units[(r["scene_id"], r["source"])][r["kind"]] = r
        elif r["kind"] == "random":
            units[(r["scene_id"], r["source"])].setdefault("random", {})[r["seed_index"]] = r
    return clean, label, units, structure


def analyze(path: Path) -> tuple[dict, list[dict]]:
    clean, label, units, structure = load(path)

    def complete(sid):
        return sid in clean and all(
            all(m in units.get((sid, s), {}) for m in ("fgsm", "pgd3", "topk_grad"))
            and sorted(units[(sid, s)].get("random", {})) == list(range(N_SEEDS)) for s in CTX)

    seen = sorted(clean)
    groups = [s for s in seen if complete(s)]
    excluded = [s for s in seen if s not in groups]

    def D(sid, src, row, where):
        ctx = src if where == "source" else OTHER[src]
        return clean[sid][ctx]["margin"] - row["scores"][ctx]["margin"]

    def flip(sid, src, row, where):
        ctx = src if where == "source" else OTHER[src]
        return float(row["scores"][ctx]["chosen"] != label[sid])

    def vals(sid, src, where, f):
        u = units[(sid, src)]
        rnd = [f(sid, src, u["random"][i], where) for i in range(N_SEEDS)]
        return {"fgsm": f(sid, src, u["fgsm"], where), "pgd": f(sid, src, u["pgd3"], where),
                "topk": f(sid, src, u["topk_grad"], where), "rand": float(np.mean(rnd)), "rand_seeds": rnd}

    out: dict = {"schema": "checkpoint-2.4.confirm.analysis.v1", "input": str(path), "bootstrap": {"draws": B, "seed": SEED},
                 "n_groups_seen": len(seen), "n_groups_analyzed": len(groups), "excluded_incomplete": excluded}

    # Structure (descriptive).
    out["structure"] = {
        "units": len(structure),
        "sign_reversals_total": sum(s["sign_reversals"] for s in structure),
        "max_levels": max((s["pgd_levels_max"] for s in structure), default=None),
        "units_with_coords_outside_fgsm_support": sum(s["pgd_outside_fgsm"] > 0 for s in structure),
        "max_outside_coords": max((s["pgd_outside_fgsm"] for s in structure), default=None),
        "outside_all_half_level_ties": all(s["outside_all_half_level_ties"] for s in structure),
        "outside_all_zero_clean_gradient": all(s["outside_where_clean_grad_zero"] == s["pgd_outside_fgsm"] for s in structure),
        "density_in_fgsm_support": mean_ci(np.array([s["density_in_fgsm_support"] for s in structure])) if structure else None,
        "n_elements": structure[0]["n_elements"] if structure else None,
    }

    per_group = []
    for where in ("source", "target"):
        block = {}
        for src in (*CTX, "pooled"):
            srcs = CTX if src == "pooled" else (src,)
            v = {m: np.array([np.mean([vals(s, so, where, D)[m] for so in srcs]) for s in groups])
                 for m in ("fgsm", "pgd", "topk", "rand")}
            b = {f"D_{m}": mean_ci(x) for m, x in v.items()}
            b["selection_pgd_minus_rand"] = mean_ci(v["pgd"] - v["rand"])
            b["sparsity_rand_minus_fgsm"] = mean_ci(v["rand"] - v["fgsm"])
            b["magnitude_topk_minus_rand"] = mean_ci(v["topk"] - v["rand"])
            b["total_pgd_minus_fgsm"] = mean_ci(v["pgd"] - v["fgsm"])
            b["share_of_gap_from_sparsity"] = boot(np.stack([v["fgsm"], v["rand"], v["pgd"]], 1), share_fn)
            if src != "pooled":
                ranks, beats, below, sds = [], 0, 0, []
                for s in groups:
                    u = vals(s, src, where, D)
                    rs = np.array(u["rand_seeds"])
                    ranks.append(float(np.mean((u["pgd"] > rs) + 0.5 * (u["pgd"] == rs))))
                    beats += int((u["pgd"] > rs).all())
                    below += int((u["pgd"] < rs).all())
                    sds.append(float(rs.std(ddof=1)))
                b["pgd_rank_among_seeds"] = mean_ci(np.array(ranks))
                b["groups_pgd_beats_all_seeds"] = beats
                b["groups_pgd_below_all_seeds"] = below
                b["within_group_seed_sd_median"] = round(float(np.median(sds)), 4)
                cc = [s for s in groups if all(clean[s][c]["chosen"] == label[s] for c in CTX)]
                fl = {m: np.array([vals(s, src, where, flip)[m] for s in cc]) for m in ("fgsm", "pgd", "topk", "rand")}
                b["n_clean_correct"] = len(cc)
                b["flip_rate"] = {m: boot(x, np.mean) for m, x in fl.items()}
                b["flip_pgd_minus_rand"] = boot(fl["pgd"] - fl["rand"], np.mean)
                b["flip_topk_minus_rand"] = boot(fl["topk"] - fl["rand"], np.mean)
            block[src] = b
        out[f"margin_drop_{where}"] = block

    for s in groups:
        row = {"scene_id": s, "label": label[s],
               "clean_correct_both": all(clean[s][c]["chosen"] == label[s] for c in CTX)}
        for src in CTX:
            u = vals(s, src, "source", D)
            t = vals(s, src, "target", D)
            for m in ("fgsm", "pgd", "topk", "rand"):
                row[f"D_{m}_{src}"] = round(u[m], 6)
                row[f"Dtarget_{m}_{src}"] = round(t[m], 6)
            row[f"k_{src}"] = units[(s, src)]["pgd3"]["k"]
        per_group.append(row)

    # First-order prediction of the source margin drop, -g_M . r (descriptive).
    lin = {}
    for src in CTX:
        pv = {m: [] for m in ("pgd", "topk", "rand")}
        for s in groups:
            u = units[(s, src)]
            pv["pgd"].append(-u["pgd3"]["pred_clean_gradient"][f"M_{src}"])
            pv["topk"].append(-u["topk_grad"]["pred_clean_gradient"][f"M_{src}"])
            pv["rand"].append(float(np.mean([-u["random"][i]["pred_clean_gradient"][f"M_{src}"] for i in range(N_SEEDS)])))
        pv = {m: np.array(x) for m, x in pv.items()}
        lin[src] = {f"pred_drop_{m}": mean_ci(x) for m, x in pv.items()}
        lin[src]["ratio_of_means_pgd_over_rand"] = round(float(pv["pgd"].mean() / pv["rand"].mean()), 4)
        lin[src]["ratio_of_means_topk_over_rand"] = round(float(pv["topk"].mean() / pv["rand"].mean()), 4)
    out["first_order_prediction_source_margin"] = lin

    # Pre-registered decisions (PROTOCOL.md section 6).
    src_block = out["margin_drop_source"]
    sel = {c: src_block[c]["selection_pgd_minus_rand"]["mean"] for c in (*CTX, "pooled")}
    primary = (len(groups) > 1 and sel["neutral"]["estimate"] > 0 and sel["helpful_nonrevealing"]["estimate"] > 0
               and sel["pooled"]["ci95"][0] is not None and sel["pooled"]["ci95"][0] > 0)
    share = src_block["pooled"]["share_of_gap_from_sparsity"]
    s1 = share["estimate"] is not None and share["ci95"][1] is not None and share["ci95"][1] <= SHARE_MAX
    mag = {c: src_block[c]["magnitude_topk_minus_rand"]["mean"] for c in (*CTX, "pooled")}
    s2 = (mag["neutral"]["estimate"] < 0 and mag["helpful_nonrevealing"]["estimate"] < 0
          and mag["pooled"]["ci95"][1] is not None and mag["pooled"]["ci95"][1] < 0)
    out["decisions"] = {
        "P1_selection_beats_matched_random": {
            "rule": "mean(D_PGD3 - mean_r D_rand_r) > 0 in neutral and in helpful, and pooled 95% CI lower bound > 0",
            "neutral": sel["neutral"], "helpful_nonrevealing": sel["helpful_nonrevealing"], "pooled": sel["pooled"],
            "supported": bool(primary)},
        "S1_sparsity_share_at_most_0.25": {
            "rule": "pooled share = (mean D_rand - mean D_FGSM)/(mean D_PGD3 - mean D_FGSM); 95% CI upper bound <= 0.25",
            "share": share, "supported": bool(s1)},
        "S2_topk_gradient_worse_than_random": {
            "rule": "mean(D_topk - D_rand) < 0 in neutral and in helpful, and pooled 95% CI upper bound < 0",
            "neutral": mag["neutral"], "helpful_nonrevealing": mag["helpful_nonrevealing"], "pooled": mag["pooled"],
            "supported": bool(s2)},
        "complete_sample": len(groups) == 48 and not excluded,
    }
    return out, per_group


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run" / "records.jsonl"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "analysis" / "confirm_analysis.json"
    out, per_group = analyze(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    with dest.with_name(dest.stem.replace("analysis", "per_group") + ".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_group[0]))
        w.writeheader()
        w.writerows(per_group)
    print(json.dumps(out["decisions"], indent=1))


if __name__ == "__main__":
    main()
