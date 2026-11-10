"""Checkpoint 4 Stage 1 analysis (STAGE1_PROTOCOL.md section 6). Committed before any attack.
Reads records only. EXPLORATORY, development data.

Run:  python docs/checkpoint-4-internvl/analyze_stage1.py [stage1_dir] [out.json]
Default input  docs/checkpoint-4-internvl/stage1/{bf16,fp32}/records.jsonl
Default output docs/checkpoint-4-internvl/analysis/stage1_analysis.json

Unit of resampling: the scene. Every paired quantity (precision, path, source, model) is
first formed per scene, so the bootstrap keeps the dependence between precision modes,
paths, attack sources and scoring contexts.
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
CTX = ("neutral", "helpful_nonrevealing")
OTHER = {CTX[0]: CTX[1], CTX[1]: CTX[0]}
PRECISIONS = ("bf16", "fp32")
PATHS = ("official", "float")
SOURCES = (*CTX, "pooled")
B, SEED = 9999, 20261111
N_SEEDS = 8
METHODS = ("fgsm", "pgd", "topk", "rand")


def boot(v: np.ndarray, fn=np.mean) -> dict:
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(B, n))
    draws = np.array([fn(v[i]) for i in idx], dtype=float)
    finite = np.isfinite(draws)
    est = float(fn(v))
    lo, hi = (np.percentile(draws[finite], [2.5, 97.5]) if finite.any() else (math.nan, math.nan))
    r = lambda x: round(float(x), 5) if math.isfinite(x) else None  # noqa: E731
    return {"n": int(n), "estimate": r(est), "ci95": [r(lo), r(hi)], "nonfinite_draws": int((~finite).sum())}


def share_fn(a: np.ndarray) -> float:
    den = a[:, 2].mean() - a[:, 0].mean()
    return (a[:, 1].mean() - a[:, 0].mean()) / den if den > 0 else math.inf


def ratio_fn(a: np.ndarray) -> float:
    return a[:, 1].mean() / a[:, 0].mean() if a[:, 0].mean() > 0 else math.inf


# ---------------------------------------------------------------- InternVL records
def load_internvl(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    d = {"clean": {}, "label": {}, "units": defaultdict(dict), "structure": {}}
    for r in rows:
        if r["kind"] == "clean":
            d["clean"][r["scene_id"]] = r["scores"]
            d["label"][r["scene_id"]] = r["label"]
        elif r["kind"] == "structure":
            d["structure"][(r["scene_id"], r["source"])] = r
        elif r["kind"] == "random":
            d["units"][(r["scene_id"], r["source"])].setdefault("random", {})[r["seed_index"]] = r
        else:
            d["units"][(r["scene_id"], r["source"])][r["kind"]] = r
    return d


def complete_scenes(d: dict) -> list[str]:
    def ok(sid):
        return sid in d["clean"] and all(
            all(k in d["units"].get((sid, s), {}) for k in ("fgsm", "pgd3", "topk_grad"))
            and sorted(d["units"][(sid, s)].get("random", {})) == list(range(N_SEEDS)) for s in CTX)
    return sorted(s for s in d["clean"] if ok(s))


def internvl_values(d: dict, sid: str, src: str, where: str, path: str) -> dict:
    ctx = src if where == "source" else OTHER[src]
    c = d["clean"][sid][ctx][path]["margin"]
    u = d["units"][(sid, src)]
    D = lambda row: c - row["scores"][ctx][path]["margin"]  # noqa: E731
    seeds = [D(u["random"][i]) for i in range(N_SEEDS)]
    return {"fgsm": D(u["fgsm"]), "pgd": D(u["pgd3"]), "topk": D(u["topk_grad"]), "rand": float(np.mean(seeds)),
            "rand_seeds": seeds,
            "flip": {m: float(u[k]["scores"][ctx][path]["chosen"] != d["label"][sid])
                     for m, k in (("fgsm", "fgsm"), ("pgd", "pgd3"), ("topk", "topk_grad"))}
                    | {"rand": float(np.mean([u["random"][i]["scores"][ctx][path]["chosen"] != d["label"][sid] for i in range(N_SEEDS)]))}}


# ---------------------------------------------------------------- Qwen development records (matched scenes)
def load_qwen_dev() -> dict | None:
    """Qwen3-VL-2B bf16 on the same 48 development scenes: clean/FGSM from 2.3 paths (mode B),
    PGD-3 and top-|grad| (pass A) and random seeds 0-7 (passes A and B) from 2.3b masks. Official path only."""
    paths = ROOT / "docs/checkpoint-2.3-exploratory/paths/records.jsonl"
    masks = ROOT / "docs/checkpoint-2.3-exploratory/masks/records.jsonl"
    if not (paths.is_file() and masks.is_file()):
        return None
    clean, fgsm, units = {}, {}, defaultdict(dict)
    for line in paths.read_text().splitlines():
        r = json.loads(line)
        if r.get("mode") != "B":
            continue
        if r["kind"] == "clean":
            clean[r["scene_id"]] = {c: r["scores"][c]["official"] for c in CTX}
        elif r["kind"] == "fgsm":
            fgsm[(r["scene_id"], r["source"])] = {c: r["scores"][c]["official"] for c in CTX}
    for line in masks.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] != "mask":
            continue
        key = (r["scene_id"], r["source"])
        if r["mask"] == "random":
            units[key].setdefault("random", {})[r["seed_index"]] = r["scores"]
        else:
            units[key][r["mask"]] = r["scores"]
    return {"clean": clean, "fgsm": fgsm, "units": units}


def qwen_values(q: dict, sid: str, src: str, where: str) -> dict | None:
    ctx = src if where == "source" else OTHER[src]
    u = q["units"].get((sid, src), {})
    if sid not in q["clean"] or (sid, src) not in q["fgsm"] or "pgd_mask" not in u or sorted(u.get("random", {}))[:N_SEEDS] != list(range(N_SEEDS)):
        return None
    c = q["clean"][sid][ctx]["margin"]
    seeds = [c - u["random"][i][ctx]["margin"] for i in range(N_SEEDS)]
    return {"fgsm": c - q["fgsm"][(sid, src)][ctx]["margin"], "pgd": c - u["pgd_mask"][ctx]["margin"],
            "topk": c - u["topk_grad"][ctx]["margin"], "rand": float(np.mean(seeds))}


# ---------------------------------------------------------------- blocks
def arrays(getter, scenes, src):
    srcs = CTX if src == "pooled" else (src,)
    return {m: np.array([np.mean([getter(s, so)[m] for so in srcs]) for s in scenes]) for m in METHODS}


def quantities(v: dict) -> dict:
    b = {f"D_{m}": boot(x) for m, x in v.items()}
    b["selection_pgd_minus_rand"] = boot(v["pgd"] - v["rand"])
    b["sparsity_rand_minus_fgsm"] = boot(v["rand"] - v["fgsm"])
    b["magnitude_topk_minus_rand"] = boot(v["topk"] - v["rand"])
    b["total_pgd_minus_fgsm"] = boot(v["pgd"] - v["fgsm"])
    b["share_of_gap_from_sparsity"] = boot(np.stack([v["fgsm"], v["rand"], v["pgd"]], 1), share_fn)
    b["ratio_pgd_over_fgsm"] = boot(np.stack([v["fgsm"], v["pgd"]], 1), ratio_fn)
    return b


def sel(v):
    return v["pgd"] - v["rand"]


def positive(b) -> bool:
    lo = b["ci95"][0]
    return lo is not None and lo > 0


def analyze(stage1_dir: Path) -> dict:
    data = {p: load_internvl(stage1_dir / p / "records.jsonl") for p in PRECISIONS if (stage1_dir / p / "records.jsonl").is_file()}
    out: dict = {"schema": "checkpoint-4.stage1.analysis.v1", "label": "EXPLORATORY, DEVELOPMENT DATA, single-tile preprocessing",
                 "bootstrap": {"draws": B, "seed": SEED, "unit": "scene"}, "precisions_present": sorted(data)}
    scenes_by_p = {p: complete_scenes(d) for p, d in data.items()}
    out["n_complete_scenes"] = {p: len(s) for p, s in scenes_by_p.items()}
    common = sorted(set.intersection(*(set(s) for s in scenes_by_p.values()))) if data else []
    out["n_scenes_both_precisions"] = len(common)

    # integrity and clean
    out["structure"], out["clean"], out["alignment"] = {}, {}, {}
    for p, d in data.items():
        st = [d["structure"][(s, so)] for s in scenes_by_p[p] for so in CTX]
        out["structure"][p] = {
            "units": len(st), "sign_reversals_total": sum(x["sign_reversals"] for x in st),
            "max_levels": max(x["pgd_levels_max"] for x in st) if st else None,
            "units_with_coords_outside_fgsm_support": sum(x["pgd_outside_fgsm"] > 0 for x in st),
            "max_outside_coords": max((x["pgd_outside_fgsm"] for x in st), default=None),
            "outside_all_half_level_ties": all(x["outside_all_half_level_ties"] for x in st),
            "density_in_fgsm_support": boot([x["density_in_fgsm_support"] for x in st]) if st else None}
        out["clean"][p] = {q: {c: {"accuracy": float(np.mean([d["clean"][s][c][q]["chosen"] == d["label"][s] for s in scenes_by_p[p]])),
                                   "mean_margin": float(np.mean([d["clean"][s][c][q]["margin"] for s in scenes_by_p[p]]))}
                               for c in CTX} for q in PATHS}
        al = {}
        for so in (*CTX, "pooled"):
            srcs = CTX if so == "pooled" else (so,)
            for key in ("cos_g0_g1", "sign_g0_g1", "cos_g0_g2", "cos_g1_g2"):
                al.setdefault(so, {})[key] = boot([np.mean([d["structure"][(s, x)]["alignment"][key] for x in srcs]) for s in scenes_by_p[p]])
        out["alignment"][p] = al

    # per precision x path x source x where
    blocks: dict = {}
    for p, d in data.items():
        for q in PATHS:
            for where in ("source", "target"):
                for src in SOURCES:
                    v = arrays(lambda s, so: internvl_values(d, s, so, where, q), scenes_by_p[p], src)
                    b = quantities(v)
                    if src != "pooled" and where == "source":
                        beats = below = 0
                        for s in scenes_by_p[p]:
                            u = internvl_values(d, s, src, where, q)
                            rs = np.array(u["rand_seeds"])
                            beats += int((u["pgd"] > rs).all()); below += int((u["pgd"] < rs).all())
                        b["scenes_pgd_beats_all_seeds"], b["scenes_pgd_below_all_seeds"] = beats, below
                        cc = [s for s in scenes_by_p[p] if all(d["clean"][s][c][q]["chosen"] == d["label"][s] for c in CTX)]
                        b["n_clean_correct"] = len(cc)
                        b["flip_rate"] = {m: boot([internvl_values(d, s, src, where, q)["flip"][m] for s in cc]) if cc else None for m in METHODS}
                    blocks.setdefault(p, {}).setdefault(q, {}).setdefault(where, {})[src] = b
    out["margin_drop"] = blocks

    # paired differences (per-scene first)
    def per_scene(p, q, src, where="source"):
        return arrays(lambda s, so: internvl_values(data[p], s, so, where, q), common, src)

    paired: dict = {}
    if len(data) == 2 and common:
        for q in PATHS:
            for src in SOURCES:
                a, b = per_scene("bf16", q, src), per_scene("fp32", q, src)
                paired.setdefault("fp32_minus_bf16", {}).setdefault(q, {})[src] = {
                    "selection": boot(sel(b) - sel(a)), "magnitude_topk_minus_rand": boot((b["topk"] - b["rand"]) - (a["topk"] - a["rand"])),
                    "total_pgd_minus_fgsm": boot((b["pgd"] - b["fgsm"]) - (a["pgd"] - a["fgsm"])),
                    "D_fgsm": boot(b["fgsm"] - a["fgsm"]), "D_pgd": boot(b["pgd"] - a["pgd"])}
    for p in data:
        for src in SOURCES:
            o = arrays(lambda s, so: internvl_values(data[p], s, so, "source", "official"), scenes_by_p[p], src)
            f = arrays(lambda s, so: internvl_values(data[p], s, so, "source", "float"), scenes_by_p[p], src)
            paired.setdefault("official_minus_float", {}).setdefault(p, {})[src] = {
                **{f"D_{m}": boot(o[m] - f[m]) for m in METHODS}, "selection": boot(sel(o) - sel(f)),
                "magnitude_topk_minus_rand": boot((o["topk"] - o["rand"]) - (f["topk"] - f["rand"]))}
    out["paired"] = paired

    # Qwen3-VL-2B, same 48 development scenes (matched scenes and design; different model, preprocessing)
    q = load_qwen_dev()
    if q is not None:
        qs = [s for s in sorted(q["clean"]) if all(qwen_values(q, s, so, "source") for so in CTX)]
        qv = {src: arrays(lambda s, so: qwen_values(q, s, so, "source"), qs, src) for src in SOURCES}
        out["qwen_dev_reference"] = {"n_scenes": len(qs), "note": "Qwen3-VL-2B bf16, official path, Checkpoint 2.3/2.3b records",
                                     **{src: quantities(v) for src, v in qv.items()}}
        cmp: dict = {}
        for p in data:
            both = sorted(set(qs) & set(scenes_by_p[p]))
            for src in SOURCES:
                iv = arrays(lambda s, so: internvl_values(data[p], s, so, "source", "official"), both, src)
                qq = arrays(lambda s, so: qwen_values(q, s, so, "source"), both, src)
                cmp.setdefault(p, {})[src] = {"n": len(both), "selection": boot(sel(iv) - sel(qq)),
                                              "magnitude_topk_minus_rand": boot((iv["topk"] - iv["rand"]) - (qq["topk"] - qq["rand"])),
                                              "D_fgsm": boot(iv["fgsm"] - qq["fgsm"]), "D_pgd": boot(iv["pgd"] - qq["pgd"])}
        out["internvl_official_minus_qwen"] = cmp

    # readings fixed in the protocol (section 7); primary = neutral source, source-context margin
    rd: dict = {}
    if data:
        S = lambda p, qq, src="neutral": blocks[p][qq]["source"][src]["selection_pgd_minus_rand"]  # noqa: E731
        have = [p for p in PRECISIONS if p in data]
        rd["selection_positive"] = {p: {qq: {src: positive(S(p, qq, src)) for src in SOURCES} for qq in PATHS} for p in have}
        if len(have) == 2:
            rd["R1_persists_in_both_precisions"] = positive(S("bf16", "official")) and positive(S("fp32", "official"))
            dsel = paired["fp32_minus_bf16"]["official"]["neutral"]["selection"]
            e_b, e_f = S("bf16", "official")["estimate"], S("fp32", "official")["estimate"]
            rd["R2_much_stronger_in_bf16"] = (dsel["ci95"][1] is not None and dsel["ci95"][1] < 0
                                             and e_b is not None and e_f is not None and e_b > 0 and e_f < 0.5 * e_b)
            rd["R4_does_not_replicate"] = not any(positive(S(p, qq)) for p in have for qq in PATHS)
        r3 = {}
        for p in have:
            dpath = paired["official_minus_float"][p]["neutral"]["selection"]
            r3[p] = (positive(S(p, "official")) and not positive(S(p, "float")) and positive(dpath))
        rd["R3_only_after_official_preprocessing"] = r3
        rd["where_selection_works"] = {p: ("both paths" if positive(S(p, "official")) and positive(S(p, "float"))
                                           else "official path only" if positive(S(p, "official"))
                                           else "float path only" if positive(S(p, "float")) else "neither path") for p in have}
    out["readings"] = rd
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "stage1"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "analysis" / "stage1_analysis.json"
    out = analyze(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["readings"], indent=1))


if __name__ == "__main__":
    main()
