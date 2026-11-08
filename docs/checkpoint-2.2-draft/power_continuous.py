"""Power for the draft Checkpoint 2.2 continuous endpoint. PLANNING ONLY.

Inputs, all already committed:
- Predictors: the Checkpoint 1.8 clean-image scalars for the 96 development_v4
  scenes (alignment in both directions, both clean margins, both gradient L1
  norms). Groups are resampled whole, so sibling pairing and the correlations
  between predictors are kept.
- Outcome spread only: the development Checkpoint 2.1 change in labeled-color NLL
  under helpful text after the frozen neutral PGD-3 attack. This script uses its
  standard deviation, its shape (as an error distribution), and the correlation
  between sibling arms. It never relates the outcome to any predictor, and the
  effect sizes are assumptions.

Primary model (draft protocol, section 5):
    dL_helpful ~ A_nh + M_neutral + M_helpful + log L1_neutral + log L1_helpful
Inference: wild cluster restricted bootstrap with Webb weights, clustered by
counterfactual group, one-sided test of beta_A = 0 at 0.025 (equivalent to the
95% two-sided interval excluding 0 from above). CR1 with t(G-1) is reported
alongside as a cross-check.

Blocks in power_continuous.json:
- grid: additive model, additive world, by partial R^2, ICC, error shape, groups.
- first_order_anchor: additive model when the truth is the local linear
  prediction k * 0.75 * eps * A * ||g_helpful||_1. It has no power.
- scaled_model and rank_model: the two alternatives in both worlds.
- co_primary_bonferroni: additive and scaled models each at one-sided 0.0125.
- asymmetry_mde: the helpful-to-neutral secondary.

Run: python3 docs/checkpoint-2.2-draft/power_continuous.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261108
rng = np.random.default_rng(SEED)
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def load_predictors() -> np.ndarray:
    scenes = json.loads((ROOT / "docs/checkpoint-1.8/qwen_rgb_loss/progress.json").read_text())["scenes"]
    groups: dict[str, list[list[float]]] = {}
    for scene in scenes.values():
        n, h = scene["neutral"], scene["helpful_nonrevealing"]
        groups.setdefault(scene["counterfactual_group_id"], []).append([
            scene["alignment_neutral_to_helpful"], n["margin"], h["margin"],
            math.log(n["gradient"]["l1"]), math.log(h["gradient"]["l1"]),
        ])
    return np.array([groups[g] for g in sorted(groups)])  # (48, 2, 5)


def load_outcome_spread() -> dict:
    rows = [json.loads(line) for line in (ROOT / "docs/checkpoint-2.1/development/records.jsonl").read_text().splitlines()]
    pgd3 = {r["scene_id"]: r for r in rows if (r["method"], r["steps"]) == ("pgd", 3)}
    pairs, values = [], []
    by_group: dict[str, list[float]] = {}
    for r in pgd3.values():
        h = r["perturbed"]["helpful_nonrevealing"]["nll"] - r["clean"]["helpful_nonrevealing"]["nll"]
        values.append(h)
        by_group.setdefault(r["counterfactual_group_id"], []).append(h)
    for v in by_group.values():
        if len(v) == 2:
            pairs.append(v)
    pairs = np.array(pairs)
    values = np.array(values)
    sibling_r = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
    standardized = (values - values.mean()) / values.std()
    return {"sd": float(values.std(ddof=1)), "sibling_correlation": sibling_r,
            "skew": float(np.mean(standardized**3)), "standardized": standardized}


def fit_stats(X, y, g, G):
    """beta_A, CR1 t, and the pieces the bootstrap reuses."""
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    e = y - X @ b
    h = X @ inv[1]  # influence of each row on beta_A
    score = np.bincount(g, weights=h * e, minlength=G)
    n, k = X.shape
    scale = (G / (G - 1)) * ((n - 1) / (n - k))
    se = math.sqrt(scale * np.sum(score**2))
    return b[1], b[1] / se, inv, h, scale


def wild_restricted_p(X, y, g, G, t_obs, reps):
    """One-sided wild cluster restricted bootstrap p-value for beta_A > 0."""
    Xr = np.delete(X, 1, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fitted, resid = Xr @ br, y - Xr @ br
    w = WEBB[rng.integers(0, 6, (G, reps))][g]  # (n, reps)
    Y = fitted[:, None] + resid[:, None] * w
    inv = np.linalg.inv(X.T @ X)
    B = inv @ X.T @ Y
    E = Y - X @ B
    h = X @ inv[1]
    onehot = np.zeros((G, len(y)))
    onehot[g, np.arange(len(y))] = 1.0
    S = onehot @ (h[:, None] * E)
    n, k = X.shape
    se = np.sqrt((G / (G - 1)) * ((n - 1) / (n - k)) * np.sum(S**2, axis=0))
    t_star = B[1] / se
    return float(np.mean(t_star >= t_obs))


ONE_SIDED_Q = 0.975  # main() lowers it to 0.9875 for the Bonferroni co-primary block


def t_crit_one_sided(df: int) -> float:
    """Upper quantile of Student t with df degrees of freedom (Cornish-Fisher; df >= 40)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(ONE_SIDED_Q)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def simulate(Z0, spread, n_groups, partial_r2, icc, errors, reps, boot):
    b = math.sqrt(partial_r2 / (1 - partial_r2))  # 0 checks the test size
    hits_wild = hits_cr1 = 0
    tcrit = t_crit_one_sided(n_groups - 1)
    for _ in range(reps):
        Z = Z0[rng.integers(0, len(Z0), n_groups)]  # (G, 2, 5) standardized
        u = rng.normal(0, math.sqrt(icc), (n_groups, 1))
        if errors == "empirical":
            e = rng.choice(spread["standardized"], (n_groups, 2)) * math.sqrt(1 - icc)
        else:
            e = rng.normal(0, math.sqrt(1 - icc), (n_groups, 2))
        # covariates matter a little; A carries the assumed partial R^2
        y = b * Z[..., 0] + 0.3 * Z[..., 2] + 0.3 * Z[..., 4] + u + e
        keep = rng.random((n_groups, 2)) < 0.94  # clean-correct under both contexts
        gid = np.repeat(np.arange(n_groups)[:, None], 2, axis=1)[keep]
        g = np.unique(gid, return_inverse=True)[1]
        G = int(g.max()) + 1
        X = np.column_stack([np.ones(keep.sum()), Z[keep]])
        beta, t_obs, *_ = fit_stats(X, y[keep], g, G)
        hits_cr1 += t_obs > tcrit
        if boot:
            hits_wild += wild_restricted_p(X, y[keep], g, G, t_obs, boot) < 0.025
    return hits_cr1 / reps, (hits_wild / reps if boot else None)


def simulate_first_order(G_raw, spread, n_groups, k, icc, reps):
    """Outcome = k * 0.75 * eps * A * L1_helpful + noise, the local linear prediction for
    three sign steps of eps/4, scaled by k. Predictors are raw values, so the
    heavy-tailed gradient norm enters as it does in the data."""
    eps = 1 / 255
    hits = 0
    tcrit = t_crit_one_sided(n_groups - 1)
    flat = G_raw.reshape(-1, 5)
    mu, sd = flat.mean(0), flat.std(0)
    for _ in range(reps):
        R = G_raw[rng.integers(0, len(G_raw), n_groups)]
        signal = k * 0.75 * eps * R[..., 0] * np.exp(R[..., 4])
        u = rng.normal(0, math.sqrt(icc), (n_groups, 1))
        e = rng.choice(spread["standardized"], (n_groups, 2)) * math.sqrt(1 - icc)
        y = signal + spread["sd"] * (u + e)
        keep = rng.random((n_groups, 2)) < 0.94
        gid = np.repeat(np.arange(n_groups)[:, None], 2, axis=1)[keep]
        g = np.unique(gid, return_inverse=True)[1]
        X = np.column_stack([np.ones(keep.sum()), ((R - mu) / sd)[keep]])
        _, t_obs, *_ = fit_stats(X, y[keep], g, int(g.max()) + 1)
        hits += t_obs > tcrit
    return hits / reps


def simulate_scaled_model(G_raw, spread, n_groups, scenario, effect, icc, reps):
    """Secondary mechanism model: dL ~ (A * S) + S + margins, S = 0.75 eps L1_helpful.
    scenario 'first_order': dL = effect * A * S + noise.
    scenario 'additive':    dL = effect_nats_per_sd_A * z(A) + noise (the primary's world)."""
    eps = 1 / 255
    hits = 0
    tcrit = t_crit_one_sided(n_groups - 1)
    flat = G_raw.reshape(-1, 5)
    mu, sd = flat.mean(0), flat.std(0)
    for _ in range(reps):
        R = G_raw[rng.integers(0, len(G_raw), n_groups)]
        S = 0.75 * eps * np.exp(R[..., 4])
        if scenario == "first_order":
            signal = effect * R[..., 0] * S
        else:
            signal = effect * (R[..., 0] - mu[0]) / sd[0]
        u = rng.normal(0, math.sqrt(icc), (n_groups, 1))
        e = rng.choice(spread["standardized"], (n_groups, 2)) * math.sqrt(1 - icc)
        y = signal + spread["sd"] * (u + e)
        keep = rng.random((n_groups, 2)) < 0.94
        gid = np.repeat(np.arange(n_groups)[:, None], 2, axis=1)[keep]
        g = np.unique(gid, return_inverse=True)[1]
        Zm = ((R - mu) / sd)[keep]
        X = np.column_stack([np.ones(keep.sum()), (R[..., 0] * S)[keep], S[keep], Zm[:, 1], Zm[:, 2]])
        _, t_obs, *_ = fit_stats(X, y[keep], g, int(g.max()) + 1)
        hits += t_obs > tcrit
    return hits / reps


def simulate_rank_model(G_raw, spread, n_groups, scenario, effect, icc, reps):
    """Rank model: rank(dL) / n ~ A + M_neutral + M_helpful + log L1_neutral + log L1_helpful."""
    eps = 1 / 255
    hits = 0
    tcrit = t_crit_one_sided(n_groups - 1)
    flat = G_raw.reshape(-1, 5)
    mu, sd = flat.mean(0), flat.std(0)
    for _ in range(reps):
        R = G_raw[rng.integers(0, len(G_raw), n_groups)]
        S = 0.75 * eps * np.exp(R[..., 4])
        if scenario == "first_order":
            signal = effect * R[..., 0] * S
        else:
            signal = effect * (R[..., 0] - mu[0]) / sd[0]
        u = rng.normal(0, math.sqrt(icc), (n_groups, 1))
        e = rng.choice(spread["standardized"], (n_groups, 2)) * math.sqrt(1 - icc)
        y = signal + spread["sd"] * (u + e)
        keep = rng.random((n_groups, 2)) < 0.94
        gid = np.repeat(np.arange(n_groups)[:, None], 2, axis=1)[keep]
        g = np.unique(gid, return_inverse=True)[1]
        yk = y[keep]
        ranks = (np.argsort(np.argsort(yk)) + 0.5) / len(yk)
        X = np.column_stack([np.ones(keep.sum()), ((R - mu) / sd)[keep]])
        _, t_obs, *_ = fit_stats(X, ranks, g, int(g.max()) + 1)
        hits += t_obs > tcrit
    return hits / reps


def main() -> None:
    G0 = load_predictors()
    flat = G0.reshape(-1, 5)
    Z0 = (G0 - flat.mean(0)) / flat.std(0)
    spread = load_outcome_spread()
    sd_y = spread["sd"]
    icc_dev = max(0.0, spread["sibling_correlation"])
    out = {
        "schema": "checkpoint-2.2-draft.power_continuous.v1",
        "seed": SEED,
        "planning_only": True,
        "predictors": ["A_neutral_to_helpful", "M_neutral", "M_helpful", "log_L1_neutral", "log_L1_helpful"],
        "predictor_correlation": np.corrcoef(flat.T).round(3).tolist(),
        "development_outcome_spread": {
            "sd_nats": sd_y, "sibling_correlation": spread["sibling_correlation"], "skew": spread["skew"],
            "used_for": "error scale, error shape, and group correlation only; never related to a predictor",
        },
        "sd_of_A": float(flat[:, 0].std()),
        "grid": [],
    }
    for errors in ("empirical", "normal"):
        for icc in sorted({0.3, round(icc_dev, 2), 0.6}):
            for r2 in (0.0, 0.01, 0.02, 0.05, 0.10):
                for n in (48, 64, 96, 128, 192):
                    boot = 199 if (errors == "empirical" and n in (64, 128)) else 0
                    reps = 400 if boot else 1500
                    cr1, wild = simulate(Z0, spread, n, r2, icc, errors, reps, boot)
                    nats_per_sd_A = math.sqrt(r2 / (1 - r2)) * sd_y
                    row = dict(errors=errors, icc=icc, partial_r2=r2, n_groups=n,
                               nats_per_sd_A=round(nats_per_sd_A, 3),
                               nats_per_unit_A=round(nats_per_sd_A / float(flat[:, 0].std()), 2),
                               power_cr1=round(cr1, 3), power_wild=None if wild is None else round(wild, 3))
                    out["grid"].append(row)
                    print(row, flush=True)
    out["first_order_anchor"] = []
    for k in (0.62, 1.0, 3.0, 10.0):
        for n in (128, 192):
            power = simulate_first_order(G0, spread, n, k, icc_dev, 1500)
            out["first_order_anchor"].append(dict(k=k, n_groups=n, power_cr1=round(power, 3)))
            print("first-order", k, n, round(power, 3), flush=True)
    out["scaled_model"] = []
    for scenario, effects in (("first_order", (0.0, 0.62, 1.0)), ("additive", (0.0, 0.423, 0.68))):
        for effect in effects:
            power = simulate_scaled_model(G0, spread, 128, scenario, effect, icc_dev, 1500)
            out["scaled_model"].append(dict(scenario=scenario, effect=effect, n_groups=128, power_cr1=round(power, 3)))
            print("scaled", scenario, effect, round(power, 3), flush=True)
    out["rank_model"] = []
    for scenario, effects in (("first_order", (0.0, 0.62, 1.0)), ("additive", (0.0, 0.423, 0.68))):
        for effect in effects:
            power = simulate_rank_model(G0, spread, 128, scenario, effect, icc_dev, 1500)
            out["rank_model"].append(dict(scenario=scenario, effect=effect, n_groups=128, power_cr1=round(power, 3)))
            print("rank", scenario, effect, round(power, 3), flush=True)
    global ONE_SIDED_Q
    ONE_SIDED_Q = 0.9875
    out["co_primary_bonferroni"] = []
    for r2 in (0.02, 0.05, 0.10):
        cr1, _ = simulate(Z0, spread, 128, r2, icc_dev, "empirical", 1500, 0)
        out["co_primary_bonferroni"].append(dict(model="additive", world="additive", partial_r2=r2, n_groups=128, power_cr1=round(cr1, 3)))
        print("bonf additive", r2, round(cr1, 3), flush=True)
    for k in (0.62, 1.0):
        power = simulate_scaled_model(G0, spread, 128, "first_order", k, icc_dev, 1500)
        out["co_primary_bonferroni"].append(dict(model="scaled", world="first_order", k=k, n_groups=128, power_cr1=round(power, 3)))
        print("bonf scaled", k, round(power, 3), flush=True)
    ONE_SIDED_Q = 0.975
    # Secondary asymmetry: minimum detectable mean paired difference (80% power, two-sided 0.05)
    mde = []
    for rho in (0.5, 0.8):
        for n in (64, 128):
            sd_d = sd_y * math.sqrt(2 * (1 - rho))
            design = 1 + icc_dev  # two arms per group
            mde.append(dict(within_scene_correlation=rho, n_groups=n,
                            mde_nats=round(2.8 * sd_d * math.sqrt(design / (2 * n * 0.94)), 3)))
    out["asymmetry_mde"] = mde
    (HERE / "power_continuous.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
