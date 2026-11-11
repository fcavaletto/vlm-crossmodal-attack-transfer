"""Checkpoint 2.3 exploratory analysis (EXPLORATORY_PLAN.md section 6). DESCRIPTIVE ONLY.

Written while the run was in progress, before any full-run record was read, and
debugged on the one-scene smoke output. Reads paths/records.jsonl and
paths/progress.json, plus the frozen Checkpoint 2.1 records (read only).

Uncertainty: percentile bootstrap over scenes (one scene per group, so this is a
group bootstrap), 9,999 draws, seed 20261109. Paired quantities are differenced
within scene before bootstrapping. No p-value decides anything.

Run: python3 docs/checkpoint-2.3-exploratory/analyze_paths.py [paths_dir]
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
RUN = HERE / (sys.argv[1] if len(sys.argv) > 1 else "paths")
OUT = HERE / "analysis"
CTX = ("neutral", "helpful_nonrevealing")
SHORT = {"neutral": "n", "helpful_nonrevealing": "h"}
SEED, B = 20261109, 9999
RNG = np.random.default_rng(SEED)


def other(c):
    return CTX[1] if c == CTX[0] else CTX[0]


def boot(values, stat="median"):
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    if x.size == 0:
        return {"n": 0}
    f = np.median if stat == "median" else np.mean
    if x.size == 1:
        return {"n": 1, stat: float(f(x)), "ci95": None}
    idx = RNG.integers(0, x.size, size=(B, x.size))
    draws = f(x[idx], axis=1)
    return {"n": int(x.size), stat: round(float(f(x)), 5),
            "ci95": [round(float(np.quantile(draws, 0.025)), 5), round(float(np.quantile(draws, 0.975)), 5)],
            "p25": round(float(np.quantile(x, 0.25)), 5), "p75": round(float(np.quantile(x, 0.75)), 5)}


def frac(bools):
    b = [bool(v) for v in bools if v is not None]
    if not b:
        return {"n": 0}
    x = np.asarray(b, float)
    idx = RNG.integers(0, x.size, size=(B, x.size))
    d = x[idx].mean(axis=1)
    return {"n": int(x.size), "rate": round(float(x.mean()), 4),
            "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)]}


def ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def load():
    rows = [json.loads(x) for x in (RUN / "records.jsonl").read_text().splitlines()]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["scene_id"], r["mode"])][r["kind"]].append(r)
    return rows, by


def main() -> None:
    rows, by = load()
    prog = json.loads((RUN / "progress.json").read_text())
    scenes_B = sorted({s for (s, m) in by if m == "B" and by[(s, m)]["clean"]})
    scenes_V = sorted({s for (s, m) in by if m == "V32" and by[(s, m)]["clean"]})
    out: dict = {"schema": "checkpoint-2.3.analysis.v1", "status": "exploratory, development data, descriptive",
                 "n_scenes_B": len(scenes_B), "n_scenes_V32": len(scenes_V), "run_status": prog.get("status"),
                 "stopped_reason": prog.get("stopped_reason"), "limitations": prog.get("limitations")}

    def clean(s, mode="B"):
        return by[(s, mode)]["clean"][0]

    def pts(s, mode, source, kind):
        return [r for r in by[(s, mode)][kind] if r.get("source") == source]

    def pgd(s, mode, source, k):
        r = [x for x in pts(s, mode, source, "pgd") if x["k"] == k]
        return r[0] if r else None

    def m0(s, ctx, path="float", mode="B", key="margin"):
        return clean(s, mode)["scores"][ctx][path][key]

    def drop(row, s, ctx, path="float", mode="B", key="margin"):
        """Margin drop (positive = damage)."""
        return m0(s, ctx, path, mode, key) - row["scores"][ctx][path][key]

    def rise(row, s, ctx, path="float", mode="B"):
        return row["scores"][ctx][path]["nll"] - m0(s, ctx, path, mode, "nll")

    correct = [s for s in scenes_B if all(m0(s, c, "official") > 0 for c in CTX)]
    out["clean_correct_both_contexts"] = {"n": len(correct), "excluded": [s for s in scenes_B if s not in correct]}

    # ---------- 0. Reproduction against frozen 2.1 ----------
    out["reproduction"] = prog.get("reproduction_check")
    det = [max(v.values()) for s in scenes_B for v in clean(s)["repeat_float"].values()]
    out["determinism_max_abs_diff"] = max(det) if det else None

    # ---------- 1. First-order accuracy (float path, unrounded delta, exact margin gradient) ----------
    fo = {}
    for mode, sc in (("B", scenes_B), ("V32", scenes_V)):
        for source in CTX:
            if mode == "V32" and source != "neutral":
                continue
            for ctx in CTX:
                if mode == "V32" and ctx != "neutral":
                    continue  # V32 has no helpful margin gradient
                for label, getter in (("fgsm", lambda s: pts(s, mode, source, "fgsm")),
                                      ("pgd1", lambda s: [pgd(s, mode, source, 1)]),
                                      ("pgd3", lambda s: [pgd(s, mode, source, 3)]),
                                      ("pgd7", lambda s: [pgd(s, mode, source, 7)])):
                    rm, rl, rm32 = [], [], []
                    for s in sc:
                        g = [x for x in getter(s) if x]
                        if not g:
                            continue
                        r = g[0]
                        pm = r["pred_clean_gradient"][ctx]["margin"]
                        pl = r["pred_clean_gradient"][ctx]["nll"]
                        rm.append(ratio(-drop(r, s, ctx, mode=mode), pm))
                        rm32.append(ratio(-drop(r, s, ctx, mode=mode, key="margin32"), pm))
                        rl.append(ratio(rise(r, s, ctx, mode=mode), pl))
                    if any(v is not None for v in rm):
                        fo[f"{mode}|{SHORT[source]}->{SHORT[ctx]}|{label}"] = {
                            "realized_over_pred_margin": boot(rm), "realized_over_pred_margin_fp32proj": boot(rm32),
                            "realized_over_pred_nll": boot(rl)}
    out["first_order"] = fo

    # ---------- 2. FGSM versus PGD (mode B, both directions) ----------
    meth = {}
    for source in CTX:
        d_f, d_p3, d_p7, mis, pf, pp3 = [], [], [], [], [], []
        geo3 = defaultdict(list)
        for s in scenes_B:
            f = pts(s, "B", source, "fgsm")[0]
            p3 = pgd(s, "B", source, 3)
            p7 = pgd(s, "B", source, 7)
            df = drop(f, s, source, "official")
            dp = drop(p3, s, source, "official")
            d_f.append(df)
            d_p3.append(dp)
            if p7 and "official" in p7["scores"][source]:
                d_p7.append(drop(p7, s, source, "official"))
            # clean-gradient margin prediction for the stored (rounded) perturbations
            prf = -f["pred_clean_gradient_rounded"][source]["margin"]
            prp = -p3["pred_clean_gradient_rounded"][source]["margin"]
            pf.append(prf)
            pp3.append(prp)
            mis.append((prf > prp) and (dp > df))
            for k in ("frac_zero", "frac_at_eps", "cos_ref", "sign_disagreement_ref", "l2"):
                geo3["rounded_" + k].append(p3["geometry_rounded"][k])
                geo3["float_" + k].append(p3["geometry"][k])
            geo3["fgsm_rounded_l2"].append(f["geometry_rounded"]["l2"])
        meth[SHORT[source]] = {
            "official_margin_drop_fgsm": boot(d_f), "official_margin_drop_pgd3": boot(d_p3),
            "official_margin_drop_pgd7_S16": boot(d_p7),
            "paired_pgd3_minus_fgsm": boot([a - b for a, b in zip(d_p3, d_f)]),
            "paired_pgd3_over_fgsm": boot([ratio(a, b) for a, b in zip(d_p3, d_f) if b > 0.25]),
            "pred_margin_drop_fgsm_rounded": boot(pf), "pred_margin_drop_pgd3_rounded": boot(pp3),
            "pred_pgd3_over_pred_fgsm": boot([ratio(a, b) for a, b in zip(pp3, pf)]),
            "clean_linear_model_misranks_fgsm_vs_pgd3": frac(mis),
            "pgd3_geometry": {k: boot(v) for k, v in geo3.items()},
        }
    out["fgsm_vs_pgd"] = meth

    # gradient rotation and local first-order accuracy along PGD
    rot = {}
    for mode, sc in (("B", scenes_B), ("V32", scenes_V)):
        for source in CTX:
            per_k = defaultdict(lambda: defaultdict(list))
            for s in sc:
                for g in pts(s, mode, source, "pgd_gradient"):
                    for key in ("cos_g0", "cos_prev", "sign_agree_g0", "sign_agree_g0_weighted"):
                        per_k[g["k"]][key].append(g[key])
                    per_k[g["k"]]["l1_over_g0"].append(g["norms"]["l1"] / clean(s, mode)["grad_norms"][f"L_{source}"]["l1"])
                for p in pts(s, mode, source, "pgd"):
                    per_k[p["k"]]["local_realized_over_pred_nll"].append(ratio(p["realized_nll_step"], p["local_pred_nll_step"]))
            if per_k:
                rot[f"{mode}|{SHORT[source]}"] = {str(k): {key: boot(v) for key, v in d.items()} for k, d in sorted(per_k.items())}
    out["path"] = rot

    # line scan shape
    line = {}
    for mode, sc in (("B", scenes_B), ("V32", scenes_V)):
        for source in CTX:
            if mode == "V32" and source != "neutral":
                continue
            per_t = defaultdict(list)
            per_t_pred = defaultdict(list)
            for s in sc:
                f = pts(s, mode, source, "fgsm")
                if not f:
                    continue
                full = drop(f[0], s, source, mode=mode)
                for r in pts(s, mode, source, "line"):
                    per_t[r["t"]].append(ratio(drop(r, s, source, mode=mode), full) if abs(full) > 0.25 else None)
                    per_t_pred[r["t"]].append(ratio(-drop(r, s, source, mode=mode), r["pred_clean_gradient"][source]["margin"]))
                per_t[1.0].append(1.0 if abs(full) > 0.25 else None)
                per_t_pred[1.0].append(ratio(-full, f[0]["pred_clean_gradient"][source]["margin"]))
            if per_t:
                line[f"{mode}|{SHORT[source]}"] = {str(t): {"drop_over_fgsm_drop": boot(per_t[t]),
                                                            "realized_over_pred_margin": boot(per_t_pred[t])} for t in sorted(per_t)}
    out["line_scan"] = line

    # ---------- 3. Precision ----------
    prec = {}
    gap = [abs(r["scores"][c][p]["margin"] - r["scores"][c][p]["margin32"])
           for r in rows if r["mode"] == "B" and "scores" in r for c in CTX for p in r["scores"][c]]
    flips_differ = [(r["scores"][c][p]["margin"] > 0) != (r["scores"][c][p]["margin32"] > 0)
                    for r in rows if r["mode"] == "B" and "scores" in r for c in CTX for p in r["scores"][c]]
    prec["bf16_logits_vs_fp32_projection"] = {"abs_margin_gap": boot(gap), "max_abs_margin_gap": max(gap) if gap else None,
                                              "decision_differs": frac(flips_differ)}
    agree = defaultdict(lambda: defaultdict(list))
    for s in scenes_V:
        for k, v in (clean(s, "V32").get("agreement_with_B") or {}).items():
            for kk, vv in v.items():
                agree[k][kk].append(vv)
    prec["clean_gradient_B_vs_V32"] = {k: {kk: boot(vv) for kk, vv in d.items()} for k, d in agree.items()}
    pv = defaultdict(list)
    for s in scenes_V:
        for kind in ("fgsm", "pgd3", "random"):
            rB = pts(s, "B", "neutral", "fgsm")[0] if kind == "fgsm" else (pgd(s, "B", "neutral", 3) if kind == "pgd3" else pts(s, "B", "neutral", "random")[0])
            rV = pts(s, "V32", "neutral", "fgsm")[0] if kind == "fgsm" else (pgd(s, "V32", "neutral", 3) if kind == "pgd3" else pts(s, "V32", "neutral", "random")[0])
            for ctx in CTX:
                dB = drop(rB, s, ctx, "official", "B")
                dV = drop(rV, s, ctx, "official", "V32")
                pv[f"{kind}|n->{SHORT[ctx]}|B"].append(dB)
                pv[f"{kind}|n->{SHORT[ctx]}|V32"].append(dV)
                pv[f"{kind}|n->{SHORT[ctx]}|V32_minus_B"].append(dV - dB)
        # cross evaluation: same perturbation, other precision
        for r in by[(s, "V32")]["cross"]:
            name = r["perturbation"]
            native = pts(s, "B", "neutral", "fgsm")[0] if name == "fgsm" else pgd(s, "B", "neutral", int(name[3:]))
            for ctx in CTX:
                pv[f"B-crafted {name}|{SHORT[ctx]}|eval V32 minus eval B"].append(
                    drop(r, s, ctx, "official", "V32") - drop(native, s, ctx, "official", "B"))
        for r in by[(s, "B_cross")]["cross"]:
            name = r["perturbation"]
            native = pts(s, "V32", "neutral", "fgsm")[0] if name == "fgsm" else pgd(s, "V32", "neutral", int(name[3:]))
            for ctx in CTX:
                pv[f"V32-crafted {name}|{SHORT[ctx]}|eval B minus eval V32"].append(
                    drop(r, s, ctx, "official", "B") - drop(native, s, ctx, "official", "V32"))
    prec["attack_damage_official_margin_drop_S16"] = {k: boot(v) for k, v in pv.items()}
    noise = defaultdict(list)
    for mode, sc in (("B", scenes_B), ("V32", scenes_V)):
        for s in sc:
            r = pts(s, mode, "neutral", "random")
            if r:
                noise[mode].append(abs(drop(r[0], s, "neutral", mode=mode)))
                noise[mode + "_official"].append(abs(drop(r[0], s, "neutral", "official", mode)))
    prec["random_sign_abs_margin_change"] = {k: boot(v) for k, v in noise.items()}
    q0p = HERE / "paths" / "q0.json"
    if q0p.exists():
        q0 = json.loads(q0p.read_text())
        prec["q0_bf16_input_survival"] = {str(t): boot([sc["rows"][i]["frac_changed_bf16_among_moved"] for sc in q0["scenes"]])
                                          for i, t in enumerate((0.125, 0.25, 0.5, 0.75, 1.0))}
    out["precision"] = prec

    # ---------- 4. Direction (mode B, matched) ----------
    dirn = {}
    per = {}
    for source in CTX:
        tgt = other(source)
        src_drop, tgt_drop, src_rise, tgt_rise, sflip, tflip_given, eff, fo_eff = [], [], [], [], [], [], [], []
        for s in scenes_B:
            p3 = pgd(s, "B", source, 3)
            sd, td = drop(p3, s, source, "official"), drop(p3, s, tgt, "official")
            src_drop.append(sd)
            tgt_drop.append(td)
            src_rise.append(rise(p3, s, source, "official"))
            tgt_rise.append(rise(p3, s, tgt, "official"))
            sf = p3["scores"][source]["official"]["margin"] < 0 and m0(s, source, "official") > 0
            tf = p3["scores"][tgt]["official"]["margin"] < 0 and m0(s, tgt, "official") > 0
            sflip.append(sf)
            if sf:
                tflip_given.append(tf)
            eff.append(ratio(td, sd) if sd > 0.5 else None)
            f = pts(s, "B", source, "fgsm")[0]
            fo_eff.append(ratio(f["pred_clean_gradient"][tgt]["margin"], f["pred_clean_gradient"][source]["margin"]))
        per[source] = {"src_drop": src_drop, "tgt_drop": tgt_drop, "eff": eff, "fo_eff": fo_eff}
        dirn[f"{SHORT[source]}->{SHORT[tgt]}"] = {
            "source_margin_drop_pgd3": boot(src_drop), "source_nll_rise_pgd3": boot(src_rise),
            "source_flip_rate_pgd3": frac(sflip), "conditional_transfer_rate": frac(tflip_given),
            "target_margin_drop_pgd3": boot(tgt_drop), "target_nll_rise_pgd3": boot(tgt_rise),
            "transfer_efficiency_target_over_source_drop": boot(eff),
            "first_order_predicted_efficiency_fgsm": boot(fo_eff)}
    n_, h_ = CTX
    dirn["paired_n_source_minus_h_source"] = {
        "source_margin_drop": boot([a - b for a, b in zip(per[n_]["src_drop"], per[h_]["src_drop"])]),
        "target_margin_drop": boot([a - b for a, b in zip(per[n_]["tgt_drop"], per[h_]["tgt_drop"])]),
        "transfer_efficiency": boot([a - b for a, b in zip(per[n_]["eff"], per[h_]["eff"]) if a is not None and b is not None]),
        "first_order_predicted_efficiency": boot([a - b for a, b in zip(per[n_]["fo_eff"], per[h_]["fo_eff"]) if a is not None and b is not None]),
    }
    # matched source damage along the float-path PGD trajectory
    from_path = defaultdict(lambda: defaultdict(list))
    for s in scenes_B:
        for source in CTX:
            tgt = other(source)
            xs, ys = [0.0], [0.0]
            for p in sorted(pts(s, "B", source, "pgd"), key=lambda r: r["k"]):
                xs.append(drop(p, s, source))
                ys.append(drop(p, s, tgt))
            for level in (2.0, 4.0, 6.0):
                from_path[level][source].append(_interp(xs, ys, level))
    matched = {}
    for level, d in from_path.items():
        a, b = d[n_], d[h_]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        matched[str(level)] = {"target_drop_n_source": boot([x for x, _ in pairs]), "target_drop_h_source": boot([y for _, y in pairs]),
                               "paired_n_minus_h": boot([x - y for x, y in pairs]), "n_pairs": len(pairs)}
    dirn["matched_source_margin_drop_float_path"] = matched
    pw = [clean(s)["pairwise"] for s in scenes_B]
    dirn["clean_gradient_geometry"] = {k: boot([p.get(k) for p in pw]) for k in
                                       ("cos_Ln_Lh", "cos_Mn_Mh", "A_nh", "A_hn", "sign_agree_Ln_Lh", "cos_Ln_Mn", "cos_Lh_Mh")}
    dirn["clean_gradient_l1_log_ratio_h_over_n"] = boot([math.log(clean(s)["grad_norms"]["L_helpful_nonrevealing"]["l1"] /
                                                                  clean(s)["grad_norms"]["L_neutral"]["l1"]) for s in scenes_B])
    dirn["clean_margin_gradient_l1_log_ratio_h_over_n"] = boot([math.log(clean(s)["grad_norms"]["M_helpful_nonrevealing"]["l1"] /
                                                                         clean(s)["grad_norms"]["M_neutral"]["l1"]) for s in scenes_B])
    out["direction"] = dirn

    OUT.mkdir(exist_ok=True)
    name = "analysis.json" if RUN.name == "paths" else f"analysis_{RUN.name}.json"
    (OUT / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1)[:6000])


def _interp(xs, ys, x0):
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        if a != b and (a - x0) * (b - x0) <= 0:
            return ys[i] + (x0 - a) / (b - a) * (ys[i + 1] - ys[i])
    return None


if __name__ == "__main__":
    main()
