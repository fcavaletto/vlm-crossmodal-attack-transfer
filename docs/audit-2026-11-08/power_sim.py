"""Planning simulation for H*-style designs. DEVELOPMENT PLANNING ONLY.

Reads only clean-image predictors (Checkpoint 1.8 scalars). It never reads an
attack outcome, so the effect sizes below are assumptions, not estimates.

Design A (binary, as in the Checkpoint 2.1 protocol): eligible scenes are source
successes; T=0 is a transfer failure. Gate: n_fail>=40, n_transfer>=40, G>=6.
Then OLS linear probability model T ~ A + M_n + M_h + L1, CR1 group-clustered
interval for beta_A (a cheap stand-in for the wild cluster bootstrap).
"Detect" = gate passed AND 95% interval lower bound > 0.

Design B (continuous): every source-clean-correct scene contributes Y, e.g. the
change in clean-to-perturbed helpful margin. Y ~ A + same covariates, CR1 interval.
Effect is expressed as the partial R^2 of A.
"""
import json, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
rows = json.load(open(f"{ROOT}/docs/checkpoint-2.1/development/checkpoint_1_8_scalars.json"))
groups = {}
for r in rows:
    groups.setdefault(r["counterfactual_group_id"], []).append(
        [r["alignment_neutral_to_helpful"], r["neutral_margin"], r["helpful_margin"], np.log(r["neutral_gradient_l1"])])
G0 = np.array([groups[g] for g in sorted(groups)])  # (48, 2, 4)
mu = G0.reshape(-1, 4).mean(0); sd = G0.reshape(-1, 4).std(0)
Z0 = (G0 - mu) / sd
rng = np.random.default_rng(20261108)

def draw(n_groups):
    idx = rng.integers(0, len(Z0), n_groups)
    return Z0[idx]  # (n,2,4) standardized; group pairs kept intact

def ols_cr1(X, y, g):
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    k = X.shape[1]; G = g.max() + 1; n = len(y)
    S = np.zeros((G, k)); np.add.at(S, g, X * e[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return b[1], np.sqrt(V[1, 1]), G

TCRIT = 1.96

def sim_binary(n_groups, p_fail, log_or_per_sd, sigma_u=1.0, p_eligible=0.57, reps=1000):
    passed = detect = 0
    for _ in range(reps):
        Z = draw(n_groups)
        u = rng.normal(0, sigma_u, n_groups)[:, None]
        elig = rng.random((n_groups, 2)) < p_eligible
        # failure log-odds: alignment lowers failure if log_or < 0; helpful margin too
        base = np.log(p_fail / (1 - p_fail)) - 0.5 * sigma_u**2 * 0.6
        eta = base + log_or_per_sd * Z[..., 0] - 0.5 * Z[..., 2] + u
        fail = rng.random((n_groups, 2)) < 1 / (1 + np.exp(-eta))
        gid = np.repeat(np.arange(n_groups)[:, None], 2, 1)
        m = elig
        y = (~fail[m]).astype(float); X = np.column_stack([np.ones(m.sum()), Z[m]])
        g = np.unique(gid[m], return_inverse=True)[1]
        nf = int((y == 0).sum()); nt = int((y == 1).sum())
        if nf < 40 or nt < 40 or g.max() + 1 < 6:
            continue
        passed += 1
        b, se, _ = ols_cr1(X, y, g)
        detect += (b - TCRIT * se) > 0
    return passed / reps, detect / reps

def sim_continuous(n_groups, partial_r2, icc=0.3, p_clean=0.95, reps=1000):
    detect = 0
    # choose b so that partial R^2 of A is roughly partial_r2 given unit residual var
    b = np.sqrt(partial_r2 / (1 - partial_r2))
    for _ in range(reps):
        Z = draw(n_groups)
        u = rng.normal(0, np.sqrt(icc), n_groups)[:, None]
        e = rng.normal(0, np.sqrt(1 - icc), (n_groups, 2))
        y = b * Z[..., 0] + 0.4 * Z[..., 1] + 0.3 * Z[..., 3] + u + e
        m = rng.random((n_groups, 2)) < p_clean
        gid = np.repeat(np.arange(n_groups)[:, None], 2, 1)
        X = np.column_stack([np.ones(m.sum()), Z[m]])
        g = np.unique(gid[m], return_inverse=True)[1]
        bh, se, _ = ols_cr1(X, y[m], g)
        detect += (bh - TCRIT * se) > 0
    return detect / reps

out = {"note": "Planning simulation. Effect sizes are assumptions. Reads no attack outcome.",
       "predictor_sd_raw": dict(zip(["alignment", "neutral_margin", "helpful_margin", "log_l1"], sd.round(4).tolist())),
       "predictor_corr": np.corrcoef(G0.reshape(-1, 4).T).round(3).tolist(),
       "binary": [], "continuous": []}
for p_fail in (0.026, 0.073):
    for lor in (0.0, -0.4, -0.7, -1.1):
        for n in (48, 200, 500, 1000, 2000):
            pg, pw = sim_binary(n, p_fail, lor, reps=400 if n >= 1000 else 1000)
            out["binary"].append(dict(p_fail=p_fail, log_or_per_sd=lor, n_groups=n, p_gate=pg, power=pw))
            print("binary", p_fail, lor, n, round(pg, 3), round(pw, 3), flush=True)
# prevalence uncertainty: integrate over Beta posterior from fresh scenes (1 fail of 39) with Jeffreys prior
for n in (500, 1000, 2000):
    ps = rng.beta(1.5, 38.5, 200)
    res = [sim_binary(n, float(max(p, 1e-3)), -0.7, reps=20) for p in ps]
    pg = float(np.mean([r[0] for r in res])); pw = float(np.mean([r[1] for r in res]))
    out["binary"].append(dict(p_fail="Beta(1.5,38.5)", log_or_per_sd=-0.7, n_groups=n, p_gate=pg, power=pw))
    print("binary-integrated", n, round(pg, 3), round(pw, 3), flush=True)
for r2 in (0.02, 0.05, 0.10):
    for icc in (0.3, 0.6):
        for n in (48, 96, 150, 250):
            pw = sim_continuous(n, r2, icc)
            out["continuous"].append(dict(partial_r2=r2, icc=icc, n_groups=n, power=pw))
            print("continuous", r2, icc, n, round(pw, 3), flush=True)
json.dump(out, open(f"{ROOT}/docs/audit-2026-11-08/power_sim_results.json", "w"), indent=1)
