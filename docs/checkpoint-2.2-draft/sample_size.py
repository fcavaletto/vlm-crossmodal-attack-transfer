"""Checkpoint 2.2 inference validation and sample size. PLANNING ONLY.

Uses the committed development data only through mechanistic_predictor.py:
predictors (Checkpoint 1.8 scalars) and the null-model error structure of the
Checkpoint 2.1 PGD-3 dL_h. Effect sizes are assumptions, anchored to the
source-side response (the neutral self-attack), never to a development estimate
of beta_Q or beta_A.

Group dependence is realistic in two ways:
- "resampled": whole development groups are resampled with their predictors and
  their full-model residuals. KNOWN FLAW, kept for transparency: those residuals
  were made orthogonal to the tested columns by the development fit, so this
  world understates null variability and every test looks conservative in it.
  It is reported in the size block only and is not used for power or coverage.
- "synthetic": predictors are resampled the same way, and errors are rebuilt as
  sigma(x) * (sqrt(rho) u_group + sqrt(1 - rho) e_scene), with rho in
  {0.3, observed, 0.75}, e drawn from the standardized residuals, and sigma(x)
  following the observed |residual| slope on log ||g_h||_1.

Blocks written to sample_size.json:
1. size: rejection rate under the null at one-sided 0.025 for WCR, CR1-t(G-1),
   CR2-t(IK dof), and naive OLS, for M1 and M2, by world and G.
2. coverage: WCU 95% interval coverage under the null.
3. power: M1 and M2 in their own world and in each other's world, by G.
4. assurance: power averaged over the uncertainty in the effect and the nuisance.
5. required_G: smallest G with power >= 0.80 for each effect size.

Run: python3 docs/checkpoint-2.2-draft/sample_size.py  (about 20-40 min on the M4 CPU)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hdl_stats as hs  # noqa: E402
import mechanistic_predictor as mp  # noqa: E402

SEED = 20261109
ALPHA = 0.025
M1 = "M1 primary: f(P_h) | M, log L1, f(S_h)"
M2 = "M2 secondary: A | M, log L1"


def t_quantile(q: float, df: float) -> float:
    """Upper t quantile via Cornish-Fisher for df >= 3, exact enough for planning."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(q)
    df = max(df, 1.0)
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    return z + g1 / df + g2 / df**2 + g3 / df**3


class Dev:
    """Development groups: predictors for M1 and M2, null fits, residuals."""

    def __init__(self, c: float):
        rows = [r for r in mp.load() if r["M_n"] > 0 and r["M_h"] > 0]
        d = mp.designs(rows, c)
        self.X1 = np.column_stack(d[M1][0])
        self.col1 = d[M1][1]
        self.X2 = np.column_stack(d[M2][0])
        self.col2 = d[M2][1]
        y = mp.arr(rows, "dLh_pgd3")
        # null coefficients: M1's other columns explain y; the tested column has 0
        X1n = np.delete(self.X1, self.col1, axis=1)
        self.gamma1 = np.linalg.lstsq(X1n, y, rcond=None)[0]
        self.base = X1n @ self.gamma1
        # full-model residuals (orthogonal to f(P_h) and to A, so the null is exact)
        Xboth = np.column_stack([self.X1, self.X2[:, self.col2]])
        self.resid = y - Xboth @ np.linalg.lstsq(Xboth, y, rcond=None)[0]
        self.fP = self.X1[:, self.col1]
        self.A = self.X2[:, self.col2]
        self.loglh = np.log(mp.arr(rows, "L1_h"))
        gid = mp.groups(rows)
        self.members = [np.flatnonzero(gid == j) for j in range(int(gid.max()) + 1)]
        e = self.resid
        self.hetero = float(np.polyfit(self.loglh, np.abs(e), 1)[0])
        sig = np.clip(np.polyval(np.polyfit(self.loglh, np.abs(e), 1), self.loglh), 0.2, None)
        z = e / sig
        self.z = (z - z.mean()) / z.std()
        self.sigma_x = sig / np.sqrt(np.mean(sig**2)) * float(np.std(e))
        pairs = np.array([e[m] for m in self.members if len(m) == 2])
        self.rho_obs = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1])
        q = (self.loglh - self.loglh.mean()) ** 2
        self.quad = (q - q.mean()) / q.std()  # nonlinear magnitude term, no alignment
        P = 0.75 * hs.EPS * mp.arr(rows, "Q")
        self.fP_c2 = hs.link(P, 2.0)  # the same mechanism with a different link scale

    def draw(self, rng, G):
        pick = rng.integers(0, len(self.members), G)
        idx = np.concatenate([self.members[j] for j in pick])
        g = np.concatenate([np.full(len(self.members[j]), k) for k, j in enumerate(pick)])
        return idx, g


def errors(dev: Dev, rng, idx, g, world: str, rho: float | None):
    if world == "resampled":
        return dev.resid[idx]
    G = int(g.max()) + 1
    u = rng.standard_normal(G)[g]
    e = rng.choice(dev.z, len(idx))
    return dev.sigma_x[idx] * (math.sqrt(rho) * u + math.sqrt(1 - rho) * e)


def one_rep(dev: Dev, rng, G, world, rho, beta_q, beta_a, boot, want=("M1", "M2"), cr2=True, quad=0.0, beta_q_c2=0.0):
    idx, g = dev.draw(rng, G)
    y = (dev.base[idx] + beta_q * dev.fP[idx] + beta_a * dev.A[idx] + quad * dev.quad[idx]
         + beta_q_c2 * dev.fP_c2[idx] + errors(dev, rng, idx, g, world, rho))
    res = {}
    for name, X, col in (("M1", dev.X1[idx], dev.col1), ("M2", dev.X2[idx], dev.col2)):
        if name not in want:
            continue
        b, se, t = hs.cr1(X, y, g, col)
        out = {"cr1": t > t_quantile(1 - ALPHA, G - 1)}
        n, k = X.shape
        _, e, inv = hs.ols(X, y)
        se_naive = math.sqrt(float(e @ e) / (n - k) * inv[col, col])
        out["naive"] = b / se_naive > t_quantile(1 - ALPHA, n - k)
        if cr2:
            _, se2, t2, dof = hs.cr2(X, y, g, col)
            out["cr2_ik"] = t2 > t_quantile(1 - ALPHA, dof)
            out["ik_dof"] = dof
        if boot:
            out["wcr"] = hs.wcr_p(X, y, g, col, boot, rng) < ALPHA
        res[name] = out
    return res


def rate(dev, rng, G, world, rho, beta_q, beta_a, reps, boot, cr2=True, want=("M1", "M2"), quad=0.0, beta_q_c2=0.0):
    tallies: dict = {}
    for _ in range(reps):
        for name, out in one_rep(dev, rng, G, world, rho, beta_q, beta_a, boot, want, cr2, quad, beta_q_c2).items():
            for k, v in out.items():
                tallies.setdefault(name, {}).setdefault(k, []).append(float(v))
    return {name: {k: round(float(np.mean(v)), 4) if k != "ik_dof" else round(float(np.median(v)), 1)
                   for k, v in d.items()} for name, d in tallies.items()}


def coverage(dev, rng, G, reps, boot):
    hits = {"M1": 0, "M2": 0}
    for _ in range(reps):
        idx, g = dev.draw(rng, G)
        y = dev.base[idx] + errors(dev, rng, idx, g, "synthetic", dev.rho_obs)
        for name, X, col in (("M1", dev.X1[idx], dev.col1), ("M2", dev.X2[idx], dev.col2)):
            lo, hi = hs.wcu_interval(X, y, g, col, boot, rng)
            hits[name] += lo <= 0.0 <= hi
    return {k: round(v / reps, 3) for k, v in hits.items()}


def main() -> None:
    rng = np.random.default_rng(SEED)
    mech = json.loads((HERE / "mechanistic_predictor.json").read_text())
    c = mech["c_calibration"]["chosen_c"]
    b_src = mech["c_calibration"]["grid"][str(c)]["slope_b"]
    dev = Dev(c)
    sd_A = float(np.std(dev.A))
    t0 = time.time()
    out = {
        "schema": "checkpoint-2.2-draft.sample_size.v1", "seed": SEED, "planning_only": True,
        "c": c, "source_side_slope_b": b_src,
        "effect_anchor": (
            "beta_Q = r * b, where b is the source-side slope of dL_n on f(0.75 eps ||g_n||_1). "
            "r = 1 means the helpful loss responds to its own first-order prediction as strongly as the "
            "neutral loss responds to its prediction under the attack that targets it ('full transfer')."
        ),
        "observed_residual_sibling_correlation": dev.rho_obs,
        "observed_abs_residual_slope_on_log_L1_h": dev.hetero,
        "sd_A_development": sd_A,
    }

    # 1. Size
    size = []
    for world, rho in (("resampled", None), ("synthetic", 0.3), ("synthetic", round(dev.rho_obs, 2)), ("synthetic", 0.75)):
        for G in (48, 128, 256):
            r = rate(dev, rng, G, world, rho, 0.0, 0.0, reps=2000, boot=399)
            size.append({"world": world, "rho": rho, "G": G, **r})
            print("size", world, rho, G, r, f"{time.time() - t0:.0f}s", flush=True)
    # Misspecified magnitude: a quadratic effect of log ||g_h||_1 (1 nat per SD), no alignment effect
    for G in (128, 256):
        r = rate(dev, rng, G, "synthetic", dev.rho_obs, 0.0, 0.0, reps=2000, boot=399, quad=1.0)
        size.append({"world": "synthetic + nonlinear magnitude (1 nat/SD), no alignment", "rho": round(dev.rho_obs, 2), "G": G, **r})
        print("size quad", G, r, flush=True)
    out["size"] = size
    # Robustness: the truth uses c = 2 while the frozen model uses the calibrated c; and rho extremes
    out["robustness"] = []
    b_c2 = mech["c_calibration"]["grid"]["2"]["slope_b"]
    for G in (256, 384):
        r = rate(dev, rng, G, "synthetic", dev.rho_obs, 0.0, 0.0, reps=1000, boot=399, cr2=False, beta_q_c2=b_c2)
        out["robustness"].append({"world": "truth f with c = 2, r = 1", "G": G, **r})
        print("robust c2", G, r, flush=True)
        for rho in (0.3, 0.75):
            r = rate(dev, rng, G, "synthetic", rho, b_src, 0.0, reps=1000, boot=399, cr2=False)
            out["robustness"].append({"world": f"M1 world r = 1, rho = {rho}", "G": G, **r})
            print("robust rho", rho, G, r, flush=True)

    # 2. Coverage
    out["coverage_wcu95"] = []
    for G in (128, 256):
        cov = coverage(dev, rng, G, reps=1000, boot=399)
        out["coverage_wcu95"].append({"world": "synthetic, observed rho", "G": G, **cov})
        print("coverage", G, cov, flush=True)

    # 3. Power, own and cross worlds (WCR decides; CR1 alongside)
    power = []
    Gs = (128, 192, 256, 320, 384, 512)
    for r_eff in (0.5, 1.0, 1.5, 2.0):
        for G in Gs:
            p = rate(dev, rng, G, "synthetic", dev.rho_obs, r_eff * b_src, 0.0, reps=1000, boot=399, cr2=False)
            power.append({"world": "saturating first-order (M1 true)", "r": r_eff, "beta_Q": r_eff * b_src, "G": G, **p})
            print("power M1-world", r_eff, G, p, flush=True)
    for nats_per_sd in (0.25, 0.5, 0.75):
        for G in Gs:
            p = rate(dev, rng, G, "synthetic", dev.rho_obs, 0.0, nats_per_sd / sd_A, reps=1000, boot=399, cr2=False)
            power.append({"world": "additive in A (M2 true)", "nats_per_sd_A": nats_per_sd, "G": G, **p})
            print("power M2-world", nats_per_sd, G, p, flush=True)
    out["power"] = power

    # partial R^2 implied by each effect (large-sample, from the development design)
    def partial_r2(X, col, beta):
        o = np.delete(X, col, axis=1)
        rx = X[:, col] - o @ np.linalg.lstsq(o, X[:, col], rcond=None)[0]
        v = beta**2 * np.var(rx)
        return v / (v + np.var(dev.resid))
    out["implied_partial_r2"] = {
        **{f"M1 r={r}": round(partial_r2(dev.X1, dev.col1, r * b_src), 4) for r in (0.5, 1.0, 1.5, 2.0)},
        **{f"M2 {s} nats/SD A": round(partial_r2(dev.X2, dev.col2, s / sd_A), 4) for s in (0.25, 0.5, 0.75)},
    }

    # 4. Assurance over effect and nuisance uncertainty (CR1 for speed; WCR/CR1 ratio from block 3)
    assurance = []
    for G in Gs:
        hits = {"M1": 0, "M2": 0}
        reps = 3000
        for _ in range(reps):
            r_eff = math.exp(rng.normal(0.0, 0.5))           # effect: median 1x full transfer, 95% in 0.38-2.7x
            rho = float(np.clip(rng.normal(dev.rho_obs, 0.12), 0.05, 0.9))
            scale = math.exp(rng.normal(0.0, 0.15))           # residual SD uncertainty on v5
            idx, g = dev.draw(rng, G)
            e = errors(dev, rng, idx, g, "synthetic", rho) * scale
            y = dev.base[idx] + r_eff * b_src * dev.fP[idx] + e
            for name, X, col in (("M1", dev.X1[idx], dev.col1), ("M2", dev.X2[idx], dev.col2)):
                _, _, t = hs.cr1(X, y, g, col)
                hits[name] += t > t_quantile(1 - ALPHA, G - 1)
        assurance.append({"G": G, "prior": "r ~ LogNormal(0, 0.5); rho ~ N(obs, 0.12); residual SD x LogNormal(0, 0.15)",
                          "world": "saturating first-order", "test": "CR1",
                          **{k: round(v / reps, 3) for k, v in hits.items()}})
        print("assurance", G, assurance[-1], flush=True)
    out["assurance"] = assurance

    # 5. Required G (WCR power >= 0.80), linear interpolation on the grid
    req = {}
    for row_key, label in (("r", "M1-world r"), ("nats_per_sd_A", "M2-world nats/SD A")):
        for model in ("M1", "M2"):
            for eff in sorted({p[row_key] for p in power if row_key in p}):
                curve = [(p["G"], p[model].get("wcr")) for p in power if p.get(row_key) == eff]
                need = None
                for (g0, p0), (g1, p1) in zip(curve, curve[1:]):
                    if p0 >= 0.8:
                        need = g0
                        break
                    if p0 < 0.8 <= p1:
                        need = round(g0 + (0.8 - p0) * (g1 - g0) / (p1 - p0))
                        break
                req[f"{model} test, {label} = {eff}"] = need if need is not None else f"> {Gs[-1]}"
    out["required_G_for_80pct_wcr"] = req
    out["seconds"] = round(time.time() - t0)
    (HERE / "sample_size.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("implied_partial_r2", "required_G_for_80pct_wcr")}, indent=1))


if __name__ == "__main__":
    main()
