"""Checkpoint 5 analysis (PLAN.md sections 5-6). Committed before any GPU inference.
Reads records only. EXPLORATORY, development data.

Run:  python docs/checkpoint-5-qwen-fp32/analyze_qwen_precision.py [run_dir] [out.json]
Default input  docs/checkpoint-5-qwen-fp32/run/{bf16,fp32}/records.jsonl (+ ../audit/*.json)
Default output docs/checkpoint-5-qwen-fp32/analysis/qwen_precision_analysis.json

Unit of resampling: the scene. Paired quantities are formed per scene first.
"""

from __future__ import annotations

import importlib.util
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
SOURCES = (*CTX, "pooled")
B, SEED = 9999, 20261112
N_SEEDS = 8
METHODS = ("fgsm", "pgd", "topk", "rand")
EQ_MARGIN = {"selection": 1.0, "topk_minus_rand": 0.5}
BAND = {"selection_low": 4.3, "selection_high": 7.3}  # MPS development +5.80 +/- 1.5
G5 = {"median_abs": 0.25, "p95_abs": 0.75, "max_abs": 1.5, "min_chosen_agree": 94}


def boot(v, fn=np.mean, levels=(95, 90)) -> dict:
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    rng = np.random.default_rng(SEED)
    draws = np.array([fn(v[i]) for i in rng.integers(0, n, size=(B, n))], dtype=float)
    finite = draws[np.isfinite(draws)]
    r = lambda x: round(float(x), 5) if math.isfinite(x) else None  # noqa: E731
    out = {"n": int(n), "estimate": r(float(fn(v)))}
    for lv in levels:
        a = (100 - lv) / 2
        lo, hi = np.percentile(finite, [a, 100 - a]) if finite.size else (math.nan, math.nan)
        out[f"ci{lv}"] = [r(lo), r(hi)]
    return out


def share_fn(a):
    den = a[:, 2].mean() - a[:, 0].mean()
    return (a[:, 1].mean() - a[:, 0].mean()) / den if den > 0 else math.inf


def ratio_fn(a):
    return a[:, 1].mean() / a[:, 0].mean() if a[:, 0].mean() > 0 else math.inf


def load(path: Path) -> dict:
    d = {"clean": {}, "label": {}, "units": defaultdict(dict), "structure": {}, "diag": {}}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r["kind"]
        if k == "clean":
            d["clean"][r["scene_id"]] = {c: r["scores"][c]["official"] for c in CTX}
            d["label"][r["scene_id"]] = r["label"]
        elif k == "structure":
            d["structure"][(r["scene_id"], r["source"])] = r
        elif k == "diagnostics":
            d["diag"][r["scene_id"]] = r
        elif k == "random":
            d["units"][(r["scene_id"], r["source"])].setdefault("random", {})[r["seed_index"]] = r
        elif k in ("fgsm", "pgd3", "topk_grad"):
            d["units"][(r["scene_id"], r["source"])][k] = r
    return d


def complete(d) -> list[str]:
    return sorted(s for s in d["clean"] if all(
        all(k in d["units"].get((s, so), {}) for k in ("fgsm", "pgd3", "topk_grad"))
        and sorted(d["units"][(s, so)].get("random", {})) == list(range(N_SEEDS)) for so in CTX))


def values(d, sid, src, where="source") -> dict:
    ctx = src if where == "source" else OTHER[src]
    c = d["clean"][sid][ctx]["margin"]
    u = d["units"][(sid, src)]
    D = lambda row: c - row["scores"][ctx]["margin"]  # noqa: E731
    seeds = [D(u["random"][i]) for i in range(N_SEEDS)]
    lab = d["label"][sid]
    return {"fgsm": D(u["fgsm"]), "pgd": D(u["pgd3"]), "topk": D(u["topk_grad"]), "rand": float(np.mean(seeds)),
            "rand_seeds": seeds,
            "flip": {"fgsm": float(u["fgsm"]["scores"][ctx]["chosen"] != lab), "pgd": float(u["pgd3"]["scores"][ctx]["chosen"] != lab),
                     "topk": float(u["topk_grad"]["scores"][ctx]["chosen"] != lab),
                     "rand": float(np.mean([u["random"][i]["scores"][ctx]["chosen"] != lab for i in range(N_SEEDS)]))}}


def arrays(get, scenes, src):
    srcs = CTX if src == "pooled" else (src,)
    return {m: np.array([np.mean([get(s, so)[m] for so in srcs]) for s in scenes]) for m in METHODS}


def quantities(v) -> dict:
    b = {f"D_{m}": boot(x) for m, x in v.items()}
    b["selection"] = boot(v["pgd"] - v["rand"])
    b["sparsity_rand_minus_fgsm"] = boot(v["rand"] - v["fgsm"])
    b["topk_minus_rand"] = boot(v["topk"] - v["rand"])
    b["total_pgd_minus_fgsm"] = boot(v["pgd"] - v["fgsm"])
    b["share_of_gap_from_sparsity"] = boot(np.stack([v["fgsm"], v["rand"], v["pgd"]], 1), share_fn)
    b["ratio_pgd_over_fgsm"] = boot(np.stack([v["fgsm"], v["pgd"]], 1), ratio_fn)
    return b


def lo(b, lv=95):
    return b[f"ci{lv}"][0]


def hi(b, lv=95):
    return b[f"ci{lv}"][1]


def equivalent(b, margin) -> bool:
    return lo(b, 90) is not None and hi(b, 90) is not None and -margin < lo(b, 90) and hi(b, 90) < margin


def verdict(b, margin) -> str:
    """Non-zero (95% CI excludes 0) and equivalent (90% CI inside the margin) are separate questions."""
    nonzero = lo(b) is not None and (lo(b) > 0 or hi(b) < 0)
    eq = equivalent(b, margin)
    if nonzero and eq:
        return "non-zero but within the equivalence margin"
    if eq:
        return "equivalent within margin"
    if nonzero:
        return "different, not within the equivalence margin"
    return "inconclusive"


def mps_reference():
    """Qwen bf16 on MPS, same 48 scenes (Checkpoint 2.3/2.3b), via the Checkpoint 4 loader."""
    spec = importlib.util.spec_from_file_location("a4", ROOT / "docs/checkpoint-4-internvl/analyze_stage1.py")
    a4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a4)
    q = a4.load_qwen_dev()
    if q is None:
        return None, None
    return q, a4


def audits(run_dir: Path) -> dict:
    out = {}
    for p in sorted((run_dir.parent / "audit").glob("*.json")) if (run_dir.parent / "audit").is_dir() else []:
        out[p.stem] = json.loads(p.read_text())
    return out


def gate_g5(clean_cuda: dict, q) -> dict | None:
    """CUDA bf16 vs MPS bf16 clean official margins on 96 prompts."""
    if clean_cuda is None or q is None:
        return None
    diffs, agree, n = [], 0, 0
    for sid, sc in q["clean"].items():
        for c in CTX:
            k = f"bf16|{sid}|{c}"
            if k not in clean_cuda["scores"]:
                continue
            a = clean_cuda["scores"][k]
            diffs.append(abs(a["margin"] - sc[c]["margin"]))
            agree += int(a["chosen"] == sc[c]["chosen"])
            n += 1
    if not n:
        return None
    d = np.array(diffs)
    res = {"n": n, "median_abs": float(np.median(d)), "p95_abs": float(np.percentile(d, 95)), "max_abs": float(d.max()),
           "chosen_agree": agree, "thresholds": G5}
    res["pass"] = (n == 96 and res["median_abs"] <= G5["median_abs"] and res["p95_abs"] <= G5["p95_abs"]
                   and res["max_abs"] <= G5["max_abs"] and agree >= G5["min_chosen_agree"])
    return res


def analyze(run_dir: Path) -> dict:
    data = {p: load(run_dir / p / "records.jsonl") for p in ("bf16", "fp32") if (run_dir / p / "records.jsonl").is_file()}
    out: dict = {"schema": "checkpoint-5.analysis.v1", "label": "EXPLORATORY, DEVELOPMENT DATA",
                 "bootstrap": {"draws": B, "seed": SEED, "unit": "scene", "ci_levels": [95, 90]},
                 "equivalence_margins": EQ_MARGIN, "reproduction_band": BAND}
    sc = {p: complete(d) for p, d in data.items()}
    out["n_complete"] = {p: len(s) for p, s in sc.items()}
    common = sorted(set.intersection(*(set(s) for s in sc.values()))) if data else []
    out["n_paired"] = len(common)

    out["structure"], out["clean"], out["alignment"], out["float_path"] = {}, {}, {}, {}
    for p, d in data.items():
        st = [d["structure"][(s, so)] for s in sc[p] for so in CTX]
        out["structure"][p] = {"units": len(st), "sign_reversals": sum(x["sign_reversals"] for x in st),
                               "max_levels": max((x["pgd_levels_max"] for x in st), default=None),
                               "max_outside_fgsm": max((x["pgd_outside_fgsm"] for x in st), default=None),
                               "outside_all_zero_gradient": all(x["outside_where_clean_grad_zero"] == x["pgd_outside_fgsm"] for x in st)}
        out["clean"][p] = {c: {"accuracy": float(np.mean([d["clean"][s][c]["chosen"] == d["label"][s] for s in sc[p]])),
                               "mean_margin": float(np.mean([d["clean"][s][c]["margin"] for s in sc[p]]))} for c in CTX}
        out["alignment"][p] = {so: {k: boot([np.mean([d["diag"][s]["alignment"][x][k] for x in (CTX if so == "pooled" else (so,))])
                                             for s in sc[p]])
                                    for k in ("cos_g0_g1", "sign_g0_g1", "cos_g0_g2", "cos_g1_g2")} for so in SOURCES}
        fp = defaultdict(list)
        for s in sc[p]:
            for e in d["diag"][s]["float_path"]:
                fp[e["kind"]].append(e["official"]["margin"] - e["float"]["margin"])
        out["float_path"][p] = {k: {"official_minus_float_margin": boot(v), "max_abs": float(np.max(np.abs(v)))} for k, v in fp.items()}
        out["float_path"][p]["clean_pixels_max_abs"] = float(max(d["diag"][s]["clean_pixels_official_vs_differentiable_max_abs"] for s in sc[p]))

    blocks: dict = {}
    for p, d in data.items():
        for where in ("source", "target"):
            for src in SOURCES:
                v = arrays(lambda s, so: values(d, s, so, where), sc[p], src)
                b = quantities(v)
                if where == "source" and src != "pooled":
                    beats = sum(int((values(d, s, src)["pgd"] > np.array(values(d, s, src)["rand_seeds"])).all()) for s in sc[p])
                    cc = [s for s in sc[p] if all(d["clean"][s][c]["chosen"] == d["label"][s] for c in CTX)]
                    b["scenes_pgd_beats_all_seeds"] = beats
                    b["n_clean_correct"] = len(cc)
                    b["flip_rate"] = {m: boot([values(d, s, src)["flip"][m] for s in cc]) if cc else None for m in METHODS}
                blocks.setdefault(p, {}).setdefault(where, {})[src] = b
    out["margin_drop"] = blocks

    # primary: paired fp32 - bf16
    paired = {}
    if len(data) == 2 and common:
        for src in SOURCES:
            a = arrays(lambda s, so: values(data["bf16"], s, so), common, src)
            f = arrays(lambda s, so: values(data["fp32"], s, so), common, src)
            sel = boot((f["pgd"] - f["rand"]) - (a["pgd"] - a["rand"]))
            tk = boot((f["topk"] - f["rand"]) - (a["topk"] - a["rand"]))
            paired[src] = {"selection": {**sel, "verdict": verdict(sel, EQ_MARGIN["selection"])},
                           "topk_minus_rand": {**tk, "verdict": verdict(tk, EQ_MARGIN["topk_minus_rand"])},
                           "D_fgsm": boot(f["fgsm"] - a["fgsm"]), "D_pgd": boot(f["pgd"] - a["pgd"]),
                           "total_pgd_minus_fgsm": boot((f["pgd"] - f["fgsm"]) - (a["pgd"] - a["fgsm"]))}
    out["paired_fp32_minus_bf16"] = paired

    # cross-device: CUDA bf16 vs MPS bf16 (same scenes)
    q, a4 = mps_reference()
    au = audits(run_dir)
    out["audits_present"] = sorted(au)
    if q is not None and "bf16" in data:
        both = [s for s in sc["bf16"] if all(a4.qwen_values(q, s, so, "source") for so in CTX)]
        cross = {}
        for src in SOURCES:
            cu = arrays(lambda s, so: values(data["bf16"], s, so), both, src)
            mp = arrays(lambda s, so: a4.qwen_values(q, s, so, "source"), both, src)
            cross[src] = {"selection_cuda_minus_mps": boot((cu["pgd"] - cu["rand"]) - (mp["pgd"] - mp["rand"])),
                          "topk_minus_rand_cuda_minus_mps": boot((cu["topk"] - cu["rand"]) - (mp["topk"] - mp["rand"])),
                          "D_fgsm_cuda_minus_mps": boot(cu["fgsm"] - mp["fgsm"]), "D_pgd_cuda_minus_mps": boot(cu["pgd"] - mp["pgd"])}
        out["cuda_bf16_minus_mps_bf16"] = cross
    clean_cuda = next((v for k, v in au.items() if k.startswith("clean_cuda")), None)
    out["G5_cross_device_clean"] = gate_g5(clean_cuda, q)

    # outcomes (PLAN 6.2-6.3)
    rd: dict = {}
    if len(data) == 2:
        sb = blocks["bf16"]["source"]["neutral"]
        sf = blocks["fp32"]["source"]["neutral"]
        band = (sb["selection"]["estimate"] is not None and BAND["selection_low"] <= sb["selection"]["estimate"] <= BAND["selection_high"]
                and lo(sb["selection"]) > 0 and hi(sb["topk_minus_rand"]) < 0)
        gates = {}
        for name, key, test in (("G1_tf32", "audit_fp32_cuda", lambda a: a["fp32_matmul_probe"]["passes_1e-5"]
                                 and not a["numerics"].get("cuda_matmul_allow_tf32") and not a["numerics"].get("cudnn_allow_tf32")
                                 and not a["numerics"].get("autocast_enabled")),
                                ("G2_fp32_dtype", "audit_fp32_cuda", lambda a: not a["modules_not_in_dtype"]),
                                ("G3_inputs", "input_hashes_compare", lambda a: a["pass"]),
                                ("G7_fp32_fd", "fd_fp32_cuda", lambda a: a["G7_pass"])):
            gates[name] = test(au[key]) if key in au else None
        gates["G5_cross_device_clean"] = None if out["G5_cross_device_clean"] is None else out["G5_cross_device_clean"]["pass"]
        drift = None
        prog = run_dir / "progress.json"
        if prog.is_file():
            drift = json.loads(prog.read_text()).get("drift_check_exact")
        gates["drift_exact"] = None if drift is None else all(drift.values())
        rd["gates"] = gates
        rd["reproduction_band_met"] = bool(band)
        D = (not band) or any(v is False for v in gates.values())
        sel_present = lo(sf["selection"]) > 0
        inv_present = hi(sf["topk_minus_rand"]) < 0
        p1 = paired["neutral"]["selection"]
        p2 = paired["neutral"]["topk_minus_rand"]
        e_sb, e_sf = sb["selection"]["estimate"], sf["selection"]["estimate"]
        e_tb, e_tf = sb["topk_minus_rand"]["estimate"], sf["topk_minus_rand"]["estimate"]
        C = (e_sb > 0 and e_sf < 0.5 * e_sb and e_tb < 0 and abs(e_tf) < 0.5 * abs(e_tb)
             and (lo(p1) > 0 or hi(p1) < 0) and (lo(p2) > 0 or hi(p2) < 0))
        A = (not D) and sel_present and inv_present and not C
        Bo = (not D) and sel_present and not inv_present and lo(p2) > 0
        rd.update({"A_both_persist": bool(A), "B_selection_persists_inversion_disappears": bool(Bo),
                   "C_both_substantially_weaker": bool((not D) and C), "D_uncontrolled_or_not_reproduced": bool(D)})
        rd["mixed"] = not any(rd[k] for k in ("A_both_persist", "B_selection_persists_inversion_disappears",
                                              "C_both_substantially_weaker", "D_uncontrolled_or_not_reproduced"))
        rd["unverified_gates"] = [k for k, v in gates.items() if v is None]
    out["readings"] = rd
    return out


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "analysis" / "qwen_precision_analysis.json"
    out = analyze(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["readings"], indent=1))


if __name__ == "__main__":
    main()
