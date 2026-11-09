"""What the committed development records already say about the Track B questions.

EXPLORATORY, DEVELOPMENT DATA ONLY. Written on 2026-11-09, after every result it
reads was known. Nothing here is a test of a hypothesis, and nothing here may be
used to choose a confirmatory specification. No model is loaded and nothing is
rendered: inputs are the Checkpoint 1.8 clean scalars and the Checkpoint 2.1
attack records, as in mechanistic_predictor.py.

Parts:
1. Clamping: can the [0, 1] box absorb a +-1/255 step? (palette bounds only)
2. Output precision: the spacing of stored margins and NLLs.
3. FGSM vs PGD on the same L-inf ball after the 8-bit round trip: damage in NLL
   and in margin, flips, and the PGD-3 -> PGD-7 gain.
4. Saturation of the NLL link: how much of the "first-order error" is the
   softmax rather than the image map. Uses margin, which is closer to linear.
5. Direction asymmetry in the clean gradients: A(n,h) against A(h,n).
6. Context: clean margins and damage under neutral vs helpful text, and the
   transfer ratio dL_h / dL_n per attack.

Run: python3 docs/checkpoint-2.2-draft/track_b_existing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src/scene_v4"))
from scene_v4.vocabulary import BACKGROUND, OUTLINE, PALETTE  # noqa: E402

EPS = 1 / 255
ATTACKS = {("pgd", 3): "pgd3", ("pgd", 7): "pgd7", ("fgsm", 1): "fgsm", ("random", 1): "random"}


def q(x, p):
    return float(np.quantile(x, p))


def summ(x):
    x = np.asarray(x, float)
    return {"n": int(x.size), "median": round(q(x, 0.5), 4), "p10": round(q(x, 0.1), 4), "p90": round(q(x, 0.9), 4)}


def spacing_share(vals, step):
    v = np.asarray(vals, float)
    return round(float(np.mean(np.abs(v / step - np.round(v / step)) < 1e-4)), 4)


def main() -> None:
    clean = json.loads((ROOT / "docs/checkpoint-1.8/qwen_rgb_loss/progress.json").read_text())["scenes"]
    recs = [json.loads(x) for x in (ROOT / "docs/checkpoint-2.1/development/records.jsonl").read_text().splitlines()]
    by: dict[str, dict[str, dict]] = {}
    for r in recs:
        if r.get("error"):
            continue
        by.setdefault(r["scene_id"], {})[ATTACKS[(r["method"], r["steps"])]] = r
    sids = sorted(s for s in by if len(by[s]) == 4)
    out: dict[str, object] = {"schema": "checkpoint-2.2-draft.track_b_existing.v1", "status": "exploratory, development data, post hoc", "n_scenes": len(sids)}

    # 1. clamping
    chans = [c for rgb in [*PALETTE.values(), BACKGROUND, OUTLINE] for c in rgb]
    out["clamping"] = {"min_channel": min(chans), "max_channel": max(chans),
                       "any_channel_within_1_of_bounds": bool(min(chans) < 1 or max(chans) > 254),
                       "note": "renderer draws without antialiasing, so every pixel is one of these colors"}

    # 2. precision
    margins = [r["clean"][c]["margin"] for s in sids for r in by[s].values() for c in ("neutral", "helpful_nonrevealing")]
    margins += [r["perturbed"][c]["margin"] for s in sids for r in by[s].values() for c in ("neutral", "helpful_nonrevealing")]
    out["output_precision"] = {"n_margins": len(margins), "share_multiple_of_0.25": spacing_share(margins, 0.25),
                               "share_multiple_of_0.125": spacing_share(margins, 0.125),
                               "share_multiple_of_0.0625": spacing_share(margins, 0.0625),
                               "smallest_nonzero_abs_margin_change": None}
    dm = [abs(by[s][a]["perturbed"][c]["margin"] - by[s][a]["clean"][c]["margin"]) for s in sids for a in by[s] for c in ("neutral", "helpful_nonrevealing")]
    dm = [x for x in dm if x > 1e-6]
    out["output_precision"]["smallest_nonzero_abs_margin_change"] = round(min(dm), 4)
    out["output_precision"]["share_of_scene_attack_context_cells_with_zero_margin_change"] = round(
        1 - len(dm) / (len(sids) * 4 * 2), 4)

    # 3 + 4. attack comparison, NLL and margin
    att = {}
    for a in ("fgsm", "pgd3", "pgd7", "random"):
        row = {}
        for c, tag in (("neutral", "source_n"), ("helpful_nonrevealing", "transfer_h")):
            dL = [by[s][a]["perturbed"][c]["nll"] - by[s][a]["clean"][c]["nll"] for s in sids]
            dM = [by[s][a]["clean"][c]["margin"] - by[s][a]["perturbed"][c]["margin"] for s in sids]
            flip = [by[s][a]["perturbed"][c]["decision"] != by[s][a]["clean"][c]["decision"] for s in sids]
            row[tag] = {"dNLL": summ(dL), "margin_drop": summ(dM), "flip_rate": round(float(np.mean(flip)), 4)}
        row["linf_after_round_trip_median"] = round(q([by[s][a]["linf_after_round_trip"] for s in sids], 0.5) * 255, 4)
        row["linf_before_round_trip_median"] = round(q([by[s][a]["linf_before_round_trip"] for s in sids], 0.5) * 255, 4)
        att[a] = row
    out["attacks_same_ball"] = att
    # paired PGD3 / FGSM margin-drop ratio, source side
    r_m = []
    for s in sids:
        f = by[s]["fgsm"]["clean"]["neutral"]["margin"] - by[s]["fgsm"]["perturbed"]["neutral"]["margin"]
        p = by[s]["pgd3"]["clean"]["neutral"]["margin"] - by[s]["pgd3"]["perturbed"]["neutral"]["margin"]
        if f > 0.2:
            r_m.append(p / f)
    out["pgd3_over_fgsm_source_margin_drop"] = {**summ(r_m), "restricted_to_fgsm_drop_gt_0.2": True}

    # 4. first-order in margin space, approximately.
    # dNLL/dx = -(1 - p) dm/dx when one competitor dominates, so a first-order margin
    # prediction is eps*||g_NLL||_1 / (1 - p). 1 - p = 1 - exp(-nll).
    fo = {}
    for a, scale in (("fgsm", 1.0), ("pgd3", 0.75)):
        ratios_m, ratios_l = [], []
        for s in sids:
            g1 = clean[s]["neutral"]["gradient"]["l1"]
            nll0 = by[s][a]["clean"]["neutral"]["nll"]
            one_minus_p = -np.expm1(-nll0)
            pred_l = scale * EPS * g1
            pred_m = pred_l / max(one_minus_p, 1e-12)
            real_m = by[s][a]["clean"]["neutral"]["margin"] - by[s][a]["perturbed"]["neutral"]["margin"]
            real_l = by[s][a]["perturbed"]["neutral"]["nll"] - nll0
            ratios_m.append(real_m / pred_m)
            ratios_l.append(real_l / pred_l)
        fo[a] = {"realized_over_predicted_NLL": summ(ratios_l), "realized_over_predicted_margin_approx": summ(ratios_m)}
    fo["note"] = ("margin prediction assumes one dominant competitor; it is an approximation, "
                  "and stored margins are quantized (part 2), so small drops read as 0")
    out["first_order_nll_vs_margin"] = fo

    # 5. direction asymmetry of clean alignment
    Anh = np.array([clean[s]["alignment_neutral_to_helpful"] for s in sids], float)
    Ahn = np.array([clean[s]["alignment_helpful_to_neutral"] for s in sids], float)
    L1n = np.array([clean[s]["neutral"]["gradient"]["l1"] for s in sids], float)
    L1h = np.array([clean[s]["helpful_nonrevealing"]["gradient"]["l1"] for s in sids], float)
    ok = np.isfinite(Anh) & np.isfinite(Ahn)
    out["alignment_asymmetry"] = {
        "A_nh": summ(Anh[ok]), "A_hn": summ(Ahn[ok]),
        "corr_A_nh_A_hn": round(float(np.corrcoef(Anh[ok], Ahn[ok])[0, 1]), 4),
        "median_A_nh_minus_A_hn": round(q(Anh[ok] - Ahn[ok], 0.5), 4),
        "share_A_nh_gt_A_hn": round(float(np.mean(Anh[ok] > Ahn[ok])), 4),
        "log_L1_h_over_L1_n": summ(np.log(L1h / L1n)),
        "predicted_first_order_transfer_n_to_h_over_h_to_n": summ((Anh * L1h / (Ahn * L1n))[ok & (Ahn > 0) & (Anh > 0)]),
        "note": ("A(n,h) = g_h' sign(g_n)/||g_h||_1. Predicted first-order transfer n->h is eps*A(n,h)*||g_h||_1 "
                 "and h->n is eps*A(h,n)*||g_n||_1. No helpful-source attack was ever run, so realized "
                 "asymmetry is unknown."),
    }

    # 6. contexts
    Mn = np.array([clean[s]["neutral"]["margin"] for s in sids])
    Mh = np.array([clean[s]["helpful_nonrevealing"]["margin"] for s in sids])
    ctx = {"clean_margin_neutral": summ(Mn), "clean_margin_helpful": summ(Mh),
           "share_helpful_margin_gt_neutral": round(float(np.mean(Mh > Mn)), 4),
           "clean_L1_neutral": summ(L1n), "clean_L1_helpful": summ(L1h)}
    for a in ("fgsm", "pgd3", "pgd7"):
        dn = np.array([by[s][a]["perturbed"]["neutral"]["nll"] - by[s][a]["clean"]["neutral"]["nll"] for s in sids])
        dh = np.array([by[s][a]["perturbed"]["helpful_nonrevealing"]["nll"] - by[s][a]["clean"]["helpful_nonrevealing"]["nll"] for s in sids])
        good = dn > 0.05
        ctx[f"{a}_transfer_ratio_dLh_over_dLn"] = summ(dh[good] / dn[good])
    ctx["note"] = ("transfer ratio < 1 is expected even with identical robustness, because the attack "
                   "optimized the neutral loss. It is not evidence that helpful text is protective.")
    out["contexts"] = ctx

    (HERE / "track_b_existing.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
