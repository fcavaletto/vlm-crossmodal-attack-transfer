"""Development analysis of Q = g_h' sign(g_n). DEVELOPMENT DATA, EXPLORATORY.

Model choice and the c-calibration rule were committed before this script existed
(MODEL_CHOICE_PRECOMMIT.md, commit d2321f7). Nothing here changes them.

Inputs, all committed, none needs the model:
- Checkpoint 1.8 clean scalars for the 96 development_v4 scenes: A(n,h), A(h,n),
  both margins, and the L1 and L2 norms of both bf16 RGB gradients.
  Because A(n,h) = g_h' sign(g_n) / ||g_h||_1, Q = A(n,h) * ||g_h||_1 is exact.
- Checkpoint 2.1 records: labeled-color NLL under both contexts after the frozen
  neutral PGD-3 (primary), PGD-7, FGSM, and matched random-sign images.

Not available on development: cos(g_n, g_h) (never stored for this pair) and the
realized perturbation delta (never stored), so g_h' delta cannot be computed.

Parts:
1. Gradient shape: ||g||_1 / (sqrt(d) ||g||_2) against the Gaussian sqrt(2/pi).
2. First-order accuracy, source side and transfer side, per attack.
3. c calibration by the precommitted rule (source side only).
4. Incremental information: nested models on each attack's dL_h.
5. Error structure of the null model, for the sample-size simulation.

Run: python3 docs/checkpoint-2.2-draft/mechanistic_predictor.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import hdl_stats as hs  # noqa: E402

SEED = 20261108
D = 3 * 384 * 512
C_GRID = (0.25, 0.5, 1, 2, 4, 8, 16, 32)
REPS = 9999


def load() -> list[dict]:
    clean = json.loads((ROOT / "docs/checkpoint-1.8/qwen_rgb_loss/progress.json").read_text())["scenes"]
    rows = [json.loads(x) for x in (ROOT / "docs/checkpoint-2.1/development/records.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        key = {("pgd", 3): "pgd3", ("pgd", 7): "pgd7", ("fgsm", 1): "fgsm", ("random", 1): "random"}[(r["method"], r["steps"])]
        by.setdefault(r["scene_id"], {})[key] = r
    out = []
    for sid, s in sorted(clean.items()):
        n, h = s["neutral"], s["helpful_nonrevealing"]
        rec = {
            "scene_id": sid, "group": s["counterfactual_group_id"],
            "A_nh": s["alignment_neutral_to_helpful"], "A_hn": s["alignment_helpful_to_neutral"],
            "M_n": n["margin"], "M_h": h["margin"],
            "L1_n": n["gradient"]["l1"], "L2_n": n["gradient"]["l2"],
            "L1_h": h["gradient"]["l1"], "L2_h": h["gradient"]["l2"],
        }
        for key, r in by[sid].items():
            rec[f"dLn_{key}"] = r["perturbed"]["neutral"]["nll"] - r["clean"]["neutral"]["nll"]
            rec[f"dLh_{key}"] = r["perturbed"]["helpful_nonrevealing"]["nll"] - r["clean"]["helpful_nonrevealing"]["nll"]
        rec["Q"] = rec["A_nh"] * rec["L1_h"]
        out.append(rec)
    return out


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def groups(rows):
    return np.unique([r["group"] for r in rows], return_inverse=True)[1]


def first_order(rows) -> dict:
    """Realized change against the first-order prediction s * eps * (||g_n||_1 or Q)."""
    scale = {"fgsm": 1.0, "pgd3": 0.75, "pgd7": 1.0}  # pgd7: 7 steps of eps/4, capped at eps
    out = {}
    for key, s in scale.items():
        for side, pred_key, real_key in (("source_neutral", "L1_n", f"dLn_{key}"), ("transfer_helpful", "Q", f"dLh_{key}")):
            pred = s * hs.EPS * arr(rows, pred_key)
            real = arr(rows, real_key)
            pos = (real > 0) & (pred > 0)
            lr, lp = np.log(real[pos]), np.log(pred[pos])
            slope = float(np.polyfit(lp, lr, 1)[0]) if pos.sum() > 3 else None
            tert = np.quantile(pred, [1 / 3, 2 / 3])
            bins = np.digitize(pred, tert)
            out[f"{key}/{side}"] = {
                "first_order_scale": s,
                "pred_nats_median": float(np.median(pred)), "pred_nats_p90": float(np.quantile(pred, 0.9)),
                "real_nats_median": float(np.median(real)),
                "ratio_median": float(np.median(real / pred)),
                "ratio_median_by_prediction_tercile": [float(np.median((real / pred)[bins == b])) for b in range(3)],
                "fraction_real_positive": float(np.mean(real > 0)),
                "spearman_real_vs_pred": spearman(pred, real),
                "loglog_slope_where_both_positive": slope,
                "n_both_positive": int(pos.sum()),
            }
    return out


def calibrate_c(rows) -> dict:
    """Precommitted rule: source side, PGD-3, smallest RSS of dLn ~ a + b * f_c(0.75 eps ||g_n||_1)."""
    y = arr(rows, "dLn_pgd3")
    p = 0.75 * hs.EPS * arr(rows, "L1_n")
    table = {}
    for c in C_GRID:
        X = np.column_stack([np.ones(len(y)), hs.link(p, c)])
        b, e, _ = hs.ols(X, y)
        table[str(c)] = {"rss": float(e @ e), "slope_b": float(b[1]), "r2": hs.r2(X, y)}
    lin = np.column_stack([np.ones(len(y)), p])
    table["linear (c = inf)"] = {"rss": float(np.sum(hs.ols(lin, y)[1] ** 2)), "r2": hs.r2(lin, y)}
    best = min(C_GRID, key=lambda c: table[str(c)]["rss"])
    return {"grid": table, "chosen_c": best, "on_grid_end": best in (C_GRID[0], C_GRID[-1])}


def designs(rows, c):
    eps = hs.EPS
    P = 0.75 * eps * arr(rows, "Q")
    S = 0.75 * eps * arr(rows, "L1_h")
    one = np.ones(len(rows))
    base0 = [one, arr(rows, "M_n"), arr(rows, "M_h")]
    logs1 = [np.log(arr(rows, "L1_n")), np.log(arr(rows, "L1_h"))]
    logs2 = [np.log(arr(rows, "L2_n")), np.log(arr(rows, "L2_h"))]
    fS, fP = hs.link(S, c), hs.link(P, c)
    A = arr(rows, "A_nh")
    m1_cov = base0 + logs1 + [fS]
    return {
        # name: (columns, index of tested column or None)
        "B0 margins": (base0, None),
        "B1 + log L1, log L2": (base0 + logs1 + logs2, None),
        "B2 + f(S_h)": (base0 + logs1 + logs2 + [fS], None),
        "M1 primary: f(P_h) | M, log L1, f(S_h)": (m1_cov + [fP], len(m1_cov)),
        "f(P_h) added to B2": (base0 + logs1 + logs2 + [fS, fP], len(base0 + logs1 + logs2) + 1),
        "M2 secondary: A | M, log L1": (base0 + logs1 + [A], len(base0 + logs1)),
        "A added to B2": (base0 + logs1 + logs2 + [fS, A], len(base0 + logs1 + logs2) + 1),
        "old (b): raw P_h | S_h, M": (base0 + [S, P], 4),
        "raw Q alone | M (no magnitude control)": (base0 + [arr(rows, "Q")], 3),
    }


def incremental(rows, c, outcome_key, rng) -> dict:
    y = arr(rows, outcome_key)
    g = groups(rows)
    out = {}
    for name, (cols, col) in designs(rows, c).items():
        X = np.column_stack(cols)
        row = {"r2": hs.r2(X, y), "logo_cv_r2": hs.logo_cv_r2(X, y, g)}
        if col is not None:
            b, se, t = hs.cr1(X, y, g, col)
            sd_x = float(np.std(X[:, col]))
            # residualized SD: the variation that identifies the coefficient
            others = np.delete(X, col, axis=1)
            resid_x = X[:, col] - others @ np.linalg.lstsq(others, X[:, col], rcond=None)[0]
            lo, hi = hs.wcu_interval(X, y, g, col, REPS, rng)
            _, se2, _, dof = hs.cr2(X, y, g, col)
            row.update({
                "coef": b, "cr1_se": se, "cr1_t": t, "cr2_se": se2, "cr2_ik_dof": dof,
                "wcr_one_sided_p": hs.wcr_p(X, y, g, col, REPS, rng),
                "wcu_95": [lo, hi],
                "sd_of_tested_column": sd_x, "residualized_sd_of_tested_column": float(np.std(resid_x)),
                "nats_per_residual_sd": b * float(np.std(resid_x)),
            })
        out[name] = row
    return out


def error_structure(rows, c) -> dict:
    """Null-model residuals (M1 without f(P_h)) on PGD-3 dL_h, for the power simulation."""
    y = arr(rows, "dLh_pgd3")
    cols, col = designs(rows, c)["M1 primary: f(P_h) | M, log L1, f(S_h)"]
    X = np.column_stack(cols[:col])
    _, e, _ = hs.ols(X, y)
    g = groups(rows)
    pairs = [e[g == j] for j in np.unique(g) if np.sum(g == j) == 2]
    pr = np.array(pairs)
    icc = float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1])
    loglh = np.log(arr(rows, "L1_h"))
    hetero = float(np.polyfit(loglh, np.abs(e), 1)[0])
    fp = hs.link(0.75 * hs.EPS * arr(rows, "Q"), c)
    fp_pairs = np.array([fp[g == j] for j in np.unique(g) if np.sum(g == j) == 2])
    return {
        "residual_sd": float(np.std(e, ddof=X.shape[1])),
        "residual_sibling_correlation": icc,
        "residual_skew": float(np.mean(((e - e.mean()) / e.std()) ** 3)),
        "abs_residual_slope_on_log_L1_h": hetero,
        "predictor_fP_sibling_correlation": float(np.corrcoef(fp_pairs[:, 0], fp_pairs[:, 1])[0, 1]),
        "A_sibling_correlation": float(np.corrcoef(*np.array([arr(rows, "A_nh")[g == j] for j in np.unique(g) if np.sum(g == j) == 2]).T)[0, 1]),
        "outcome_sd": float(np.std(y, ddof=1)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    allrows = load()
    rows = [r for r in allrows if r["M_n"] > 0 and r["M_h"] > 0]
    shape = {}
    for side in ("n", "h"):
        ratio = arr(allrows, f"L1_{side}") / (math.sqrt(D) * arr(allrows, f"L2_{side}"))
        shape[side] = {"median": float(np.median(ratio)), "p10": float(np.quantile(ratio, 0.1)),
                       "p90": float(np.quantile(ratio, 0.9)), "gaussian_reference": math.sqrt(2 / math.pi)}
    cal = calibrate_c(rows)
    c = cal["chosen_c"]
    fP = hs.link(0.75 * hs.EPS * arr(rows, "Q"), c)
    loglh = np.log(arr(rows, "L1_h"))
    collinear = {
        "corr_fP_with_log_L1_h": float(np.corrcoef(fP, loglh)[0, 1]),
        "corr_fP_with_fS": float(np.corrcoef(fP, hs.link(0.75 * hs.EPS * arr(rows, "L1_h"), c))[0, 1]),
        "corr_A_with_log_L1_h": float(np.corrcoef(arr(rows, "A_nh"), loglh)[0, 1]),
        "corr_log_L1_n_with_log_L1_h": float(np.corrcoef(np.log(arr(rows, "L1_n")), loglh)[0, 1]),
    }
    out = {
        "schema": "checkpoint-2.2-draft.mechanistic_predictor.v1",
        "label": "DEVELOPMENT, EXPLORATORY. Does not choose or change the protocol.",
        "precommit": "MODEL_CHOICE_PRECOMMIT.md at d2321f7",
        "seed": SEED, "bootstrap_reps": REPS,
        "n_scenes": len(allrows), "n_eligible": len(rows), "n_groups_eligible": int(groups(rows).max()) + 1,
        "Q_definition": "Q = A(n,h) * ||g_h||_1 = g_h' sign(g_n), exact from stored scalars",
        "not_available_on_development": ["cos(g_n, g_h)", "g_h' delta for the realized perturbation"],
        "gradient_shape_L1_over_sqrtd_L2": shape,
        "first_order_accuracy": first_order(rows),
        "c_calibration": cal,
        "collinearity": collinear,
        "incremental": {k: incremental(rows, c, k, rng) for k in ("dLh_pgd3", "dLh_fgsm", "dLh_random", "dLh_pgd7")},
        "error_structure_null_model": error_structure(rows, c),
    }
    (HERE / "mechanistic_predictor.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_eligible", "gradient_shape_L1_over_sqrtd_L2", "c_calibration", "collinearity", "error_structure_null_model")}, indent=1))
    for k, v in out["first_order_accuracy"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("ratio_median", "ratio_median_by_prediction_tercile", "spearman_real_vs_pred", "loglog_slope_where_both_positive", "fraction_real_positive")})
    for outcome, block in out["incremental"].items():
        print("==", outcome)
        for name, row in block.items():
            short = {k: (round(v, 3) if isinstance(v, float) else ([round(x, 3) for x in v] if isinstance(v, list) else v)) for k, v in row.items()}
            print(" ", name, short)


if __name__ == "__main__":
    main()
