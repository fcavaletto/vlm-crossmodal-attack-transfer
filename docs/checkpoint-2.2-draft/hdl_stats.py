"""Shared statistics for the Checkpoint 2.2 H-dL planning scripts. PLANNING ONLY.

The same functions are used by the development analysis (mechanistic_predictor.py)
and the simulations (sample_size.py), so the procedure that is validated is the
procedure that would be run.

- OLS with cluster-robust CR1 and CR2 (Bell-McCaffrey) standard errors.
- Wild cluster restricted bootstrap (WCR) p-values, Webb six-point weights.
- Wild cluster unrestricted percentile-t (WCU) intervals.
- Leave-one-group-out cross-validated R^2.
"""

from __future__ import annotations

import math

import numpy as np

WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
EPS = 1.0 / 255.0


def link(p: np.ndarray, c: float) -> np.ndarray:
    """f(P) = c * asinh(P / c): P near 0, c * log(2P / c) for large P, odd."""
    if math.isinf(c):
        return np.asarray(p, dtype=float)
    return c * np.arcsinh(np.asarray(p, dtype=float) / c)


def onehot(g: np.ndarray, G: int) -> np.ndarray:
    m = np.zeros((G, len(g)))
    m[g, np.arange(len(g))] = 1.0
    return m


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ X.T @ y
    return b, y - X @ b, inv


def cr1(X, y, g, col):
    """Coefficient, CR1 SE, t. Small-sample factor G/(G-1) * (n-1)/(n-k)."""
    G = int(g.max()) + 1
    b, e, inv = ols(X, y)
    h = X @ inv[col]
    s = np.bincount(g, weights=h * e, minlength=G)
    n, k = X.shape
    se = math.sqrt((G / (G - 1)) * ((n - 1) / (n - k)) * float(np.sum(s**2)))
    return float(b[col]), se, float(b[col]) / se if se > 0 else float("nan")


def cr2(X, y, g, col):
    """Bell-McCaffrey CR2 SE and the Imbens-Kolesar degrees of freedom."""
    G = int(g.max()) + 1
    b, e, inv = ols(X, y)
    n = len(y)
    Hfull = X @ inv @ X.T
    a_vecs = []
    meat_terms = []
    rows = [np.flatnonzero(g == j) for j in range(G)]
    adj = []
    for idx in rows:
        Hg = Hfull[np.ix_(idx, idx)]
        I = np.eye(len(idx))
        w, V = np.linalg.eigh(I - Hg)
        w = np.clip(w, 1e-12, None)
        Ag = V @ np.diag(w**-0.5) @ V.T
        adj.append(Ag)
        meat_terms.append((X[idx].T @ Ag @ e[idx]))
    v = inv[col]
    var = float(sum((v @ m) ** 2 for m in meat_terms))
    se = math.sqrt(var)
    # Imbens-Kolesar dof: G-matrix built from the CR2 adjustment
    M = np.eye(n) - Hfull
    cols = []
    for j, idx in enumerate(rows):
        gj = np.zeros(n)
        gj[idx] = adj[j] @ (X[idx] @ v)
        cols.append(M.T @ gj)  # (I-H)' * component
    Gm = np.array(cols)  # (G, n)
    lam = np.linalg.eigvalsh(Gm @ Gm.T)
    dof = float(lam.sum() ** 2 / np.sum(lam**2)) if np.sum(lam**2) > 0 else float(G - 1)
    return float(b[col]), se, float(b[col]) / se, dof


def _t_stats(X, Y, g, col, G):
    """Vectorized CR1 t for many outcome columns Y (n, R)."""
    inv = np.linalg.pinv(X.T @ X)
    B = inv @ X.T @ Y
    E = Y - X @ B
    h = X @ inv[col]
    S = onehot(g, G) @ (h[:, None] * E)
    n, k = X.shape
    se = np.sqrt((G / (G - 1)) * ((n - 1) / (n - k)) * np.sum(S**2, axis=0))
    return B[col], se


def wcr_p(X, y, g, col, reps, rng, chunk=2000):
    """One-sided WCR p-value for b[col] > 0, null imposed, Webb weights."""
    G = int(g.max()) + 1
    _, _, t_obs = cr1(X, y, g, col)
    Xr = np.delete(X, col, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit, res = Xr @ br, y - Xr @ br
    hits, done = 0, 0
    while done < reps:
        r = min(chunk, reps - done)
        w = WEBB[rng.integers(0, 6, (G, r))][g]
        bstar, se = _t_stats(X, fit[:, None] + res[:, None] * w, g, col, G)
        hits += int(np.sum(bstar / se >= t_obs))
        done += r
    return (hits + 1) / (reps + 1)


def wcu_interval(X, y, g, col, reps, rng, level=0.95, chunk=2000):
    """Percentile-t interval from the wild cluster unrestricted bootstrap."""
    G = int(g.max()) + 1
    b, se, _ = cr1(X, y, g, col)
    bh, e, _ = ols(X, y)
    fit = X @ bh
    ts = []
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        w = WEBB[rng.integers(0, 6, (G, r))][g]
        bstar, sestar = _t_stats(X, fit[:, None] + e[:, None] * w, g, col, G)
        ts.append((bstar - b) / sestar)
        done += r
    t = np.concatenate(ts)
    lo_q, hi_q = np.quantile(t, [(1 + level) / 2, (1 - level) / 2])
    return b - lo_q * se, b - hi_q * se


def logo_cv_r2(X, y, g):
    """Leave-one-group-out predictive R^2 (1 - PRESS / SS_total)."""
    pred = np.empty_like(y, dtype=float)
    for j in np.unique(g):
        test = g == j
        b = np.linalg.lstsq(X[~test], y[~test], rcond=None)[0]
        pred[test] = X[test] @ b
    return 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def r2(X, y):
    _, e, _ = ols(X, y)
    return 1.0 - float(e @ e / np.sum((y - y.mean()) ** 2))
