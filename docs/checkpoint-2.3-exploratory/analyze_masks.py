"""Checkpoint 2.3b analysis (MASK_CONTROL_PLAN.md section 6). Descriptive; reads only.

Run: python docs/checkpoint-2.3-exploratory/analyze_masks.py [masks_dir]
Writes analysis/masks_analysis.json (or masks_analysis_<dir>.json).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIR = HERE / (sys.argv[1] if len(sys.argv) > 1 else "masks")
CTX = ("neutral", "helpful_nonrevealing")
OTHER = {CTX[0]: CTX[1], CTX[1]: CTX[0]}
B, SEED = 9999, 20261109


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def boot_stat(per_scene: np.ndarray, fn=np.mean) -> dict:
    """Percentile bootstrap over scenes (rows). per_scene is 1-D, or 2-D (scene x columns) for ratio statistics."""
    n = per_scene.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.array([fn(per_scene[i]) for i in idx])
    draws = draws[np.isfinite(draws)]
    est = fn(per_scene)
    return {"n": int(n), "estimate": round(float(est), 5),
            "ci95": [round(float(np.percentile(draws, 2.5)), 5), round(float(np.percentile(draws, 97.5)), 5)]}


def summary(v: np.ndarray) -> dict:
    return {"mean": boot_stat(v, np.mean), "median": boot_stat(v, np.median)}


def rate(flags: np.ndarray) -> dict:
    return boot_stat(flags.astype(float), np.mean)


def main() -> None:
    labels = {}
    sys.path[:0] = [str(p) for p in (ROOT / "src").glob("*/")]
    from dev_pilot.fixtures import load_manifest  # noqa: E402
    for r in load_manifest(ROOT):
        labels[r["scene_id"]] = str(r["answer"])

    base = load(HERE / "paths" / "records.jsonl")
    clean, dense = {}, {}
    for r in base:
        if r.get("mode") != "B":
            continue
        sid = r["scene_id"]
        if r["kind"] == "clean":
            clean[sid] = {c: r["scores"][c]["official"] for c in CTX}
        elif r["kind"] == "fgsm":
            dense[(sid, r["source"], "fgsm")] = {c: r["scores"][c]["official"] for c in CTX}
        elif r["kind"] == "pgd" and r.get("k") == 3:
            dense[(sid, r["source"], "pgd_recorded")] = {c: r["scores"][c]["official"] for c in CTX}

    rows = load(DIR / "records.jsonl")
    prog = json.loads((DIR / "progress.json").read_text())
    structure = [r for r in rows if r["kind"] == "structure"]
    masks = defaultdict(dict)  # (sid, source) -> name -> scores ; random -> {seed_index: scores}
    preds = defaultdict(dict)
    pipe = []
    for r in rows:
        if r["kind"] != "mask":
            continue
        key = (r["scene_id"], r["source"])
        if r["mask"] == "random":
            masks[key].setdefault("random", {})[r["seed_index"]] = r["scores"]
            preds[key].setdefault("random", {})[r["seed_index"]] = r["pred_clean_gradient"]
        else:
            masks[key][r["mask"]] = r["scores"]
            preds[key][r["mask"]] = r["pred_clean_gradient"]
            if r["mask"] == "pgd_mask":
                pipe.append(r["pipeline_check"])

    units = sorted(k for k in masks if "pgd_mask" in masks[k] and "topk_grad" in masks[k])
    seed_sets = [set(masks[k].get("random", {})) for k in units]
    primary = sorted(set.intersection(*seed_sets)) if seed_sets else []
    scenes = sorted({k[0] for k in units if all((k[0], s) in units for s in CTX)})
    cc = [s for s in scenes if all(clean[s][c]["chosen"] == labels[s] for c in CTX)]

    def drop(sid, source, scores, where):
        ctx = source if where == "source" else OTHER[source]
        return clean[sid][ctx]["margin"] - scores[ctx]["margin"]

    def flipped(sid, source, scores, where):
        ctx = source if where == "source" else OTHER[source]
        return float(scores[ctx]["chosen"] != labels[sid])

    def unit_values(sid, source, where, f):
        k = (sid, source)
        rnd = [f(sid, source, masks[k]["random"][s], where) for s in primary]
        return {"fgsm": f(sid, source, dense[(sid, source, "fgsm")], where),
                "pgd": f(sid, source, masks[k]["pgd_mask"], where),
                "topk": f(sid, source, masks[k]["topk_grad"], where),
                "rand": float(np.mean(rnd)), "rand_seeds": rnd}

    out: dict = {
        "schema": "checkpoint-2.3b.masks.analysis.v1", "status": "exploratory, development data, descriptive",
        "run_status": prog.get("status"), "stopped_reason": prog.get("stopped_reason"), "limitations": prog.get("limitations"),
        "n_units": len(units), "n_scenes": len(scenes), "primary_seeds": primary, "n_clean_correct_scenes": len(cc),
        "pipeline_check": {"n": len(pipe), "bit_exact": sum(p["bit_exact_vs_recorded"] is True for p in pipe),
                           "same_image": sum(p["same_image_as_official_pgd3"] for p in pipe)},
        "structure": {
            "units": len(structure),
            "sign_reversals_total": sum(s["sign_reversals"] for s in structure),
            "max_levels": max((s["pgd_levels_max"] for s in structure), default=None),
            "units_with_outside_coords": sum(s["pgd_outside_fgsm"] > 0 for s in structure),
            "max_outside_coords": max((s["pgd_outside_fgsm"] for s in structure), default=None),
            "outside_all_half_level_ties": all(s["outside_all_half_level_ties"] for s in structure),
            "outside_all_zero_clean_gradient": all(s["outside_where_clean_grad_zero"] == s["pgd_outside_fgsm"] for s in structure),
            "density_in_fgsm_support": summary(np.array([s["density_in_fgsm_support"] for s in structure])),
        },
    }

    for where in ("source", "target"):
        block = {}
        for source in (*CTX, "pooled"):
            srcs = CTX if source == "pooled" else (source,)
            vals = {m: np.array([np.mean([unit_values(s, so, where, drop)[m] for so in srcs]) for s in scenes])
                    for m in ("fgsm", "pgd", "topk", "rand")}
            b = {f"D_{m}": summary(v) for m, v in vals.items()}
            b["sparsity_rand_minus_fgsm"] = summary(vals["rand"] - vals["fgsm"])
            b["selection_pgd_minus_rand"] = summary(vals["pgd"] - vals["rand"])
            b["magnitude_topk_minus_rand"] = summary(vals["topk"] - vals["rand"])
            b["pgd_minus_topk"] = summary(vals["pgd"] - vals["topk"])
            b["total_pgd_minus_fgsm"] = summary(vals["pgd"] - vals["fgsm"])
            stack = np.stack([vals["fgsm"], vals["rand"], vals["pgd"]], 1)
            b["share_of_gap_from_sparsity"] = boot_stat(
                stack, lambda a: (a[:, 1].mean() - a[:, 0].mean()) / (a[:, 2].mean() - a[:, 0].mean()))
            if source != "pooled":
                ranks, beats_all, below_all, sds = [], 0, 0, []
                for s in scenes:
                    u = unit_values(s, source, where, drop)
                    rs = np.array(u["rand_seeds"])
                    ranks.append(float(np.mean((u["pgd"] > rs) + 0.5 * (u["pgd"] == rs))))
                    beats_all += int((u["pgd"] > rs).all())
                    below_all += int((u["pgd"] < rs).all())
                    sds.append(float(rs.std(ddof=1)) if len(rs) > 1 else float("nan"))
                b["pgd_rank_among_random_seeds"] = summary(np.array(ranks))
                b["units_pgd_beats_all_seeds"] = beats_all
                b["units_pgd_below_all_seeds"] = below_all
                b["within_unit_seed_sd_median"] = round(float(np.nanmedian(sds)), 4) if sds else None
                # Flips on clean-correct scenes.
                fl = {m: np.array([unit_values(s, source, where, flipped)[m] for s in cc]) for m in ("fgsm", "pgd", "topk", "rand")}
                b["flip_rate"] = {m: rate(v) for m, v in fl.items()}
                b["flip_pgd_minus_rand"] = boot_stat(fl["pgd"] - fl["rand"])
                b["flip_rand_minus_fgsm"] = boot_stat(fl["rand"] - fl["fgsm"])
                b["flip_topk_minus_rand"] = boot_stat(fl["topk"] - fl["rand"])
            block[source] = b
        out[f"margin_drop_{where}"] = block

    # First-order predictions on the source margin gradient (predicted drop = -g_M . r).
    lin = {}
    for source in CTX:
        def pdrop(p):
            return -p[f"M_{source}"]
        pv = {"pgd": [], "topk": [], "rand": []}
        for s in scenes:
            k = (s, source)
            pv["pgd"].append(pdrop(preds[k]["pgd_mask"]))
            pv["topk"].append(pdrop(preds[k]["topk_grad"]))
            pv["rand"].append(float(np.mean([pdrop(preds[k]["random"][i]) for i in primary])))
        pv = {m: np.array(v) for m, v in pv.items()}
        lin[source] = {f"pred_drop_{m}": summary(v) for m, v in pv.items()}
        lin[source]["pred_pgd_over_rand"] = summary(pv["pgd"] / pv["rand"])
        lin[source]["pred_topk_over_rand"] = summary(pv["topk"] / pv["rand"])
    out["first_order_prediction_source_margin"] = lin

    dest = HERE / "analysis" / ("masks_analysis.json" if DIR.name == "masks" else f"masks_analysis_{DIR.name}.json")
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
