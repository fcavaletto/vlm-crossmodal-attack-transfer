"""Does flexible magnitude adjustment fix M1's false positives? PLANNING ONLY.

sample_size.py found that a nonlinear magnitude effect (quadratic in
log ||g_h||_1, no alignment effect) makes M1's WCR test reject about 3x too
often. This compares M1 with M1s = M1 plus restricted cubic splines (4 knots, at
the 5/35/65/95 percentiles of the predictor pool) in log ||g_h||_1 and
log ||g_n||_1, in four worlds:

- null, linear magnitude (size);
- null + quadratic magnitude, 1 nat per SD (size under misspecification);
- null + a sharper magnitude effect: 1 nat x tanh of the standardized log norm (size);
- the M1 world at r = 1 (power).

Synthetic errors with the observed sibling correlation, as in sample_size.py.
No development outcome is related to any predictor here.

Run: python3 docs/checkpoint-2.2-draft/spline_check.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hdl_stats as hs  # noqa: E402
import sample_size as ss  # noqa: E402

SEED = 20261110


def rcs(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), k - 2 nonlinear columns."""
    k = knots
    scale = (k[-1] - k[0]) ** 2
    def p(u):
        return np.clip(u, 0, None) ** 3
    cols = []
    for j in range(len(k) - 2):
        cols.append((p(x - k[j]) - p(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + p(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / scale)
    return np.column_stack(cols)


def main() -> None:
    rng = np.random.default_rng(SEED)
    mech = json.loads((HERE / "mechanistic_predictor.json").read_text())
    c = mech["c_calibration"]["chosen_c"]
    b_src = mech["c_calibration"]["grid"][str(c)]["slope_b"]
    dev = ss.Dev(c)
    logn = dev.X1[:, 5]  # log L1_n column in M1's design order: 1, M_n, M_h, logL1n, logL1h, fS, fP
    logh = dev.X1[:, 4]
    # column order check: M1 = base0(3) + logs1(2) + [fS] + [fP]
    assert dev.col1 == 6
    logn, logh = dev.X1[:, 3], dev.X1[:, 4]
    q = [0.05, 0.35, 0.65, 0.95]
    S = np.column_stack([rcs(logh, np.quantile(logh, q)), rcs(logn, np.quantile(logn, q))])
    X1s = np.column_stack([dev.X1[:, :6], S, dev.X1[:, 6]])
    col1s = X1s.shape[1] - 1
    z = (logh - logh.mean()) / logh.std()
    sharp = np.tanh(2 * z)
    sharp = (sharp - sharp.mean()) / sharp.std()

    worlds = {
        "null, linear magnitude": dict(extra=0.0 * z, beta=0.0),
        "null + quadratic magnitude (1 nat/SD)": dict(extra=dev.quad, beta=0.0),
        "null + tanh magnitude (1 nat/SD)": dict(extra=sharp, beta=0.0),
        "M1 world, r = 1": dict(extra=0.0 * z, beta=b_src),
    }
    out = {"schema": "checkpoint-2.2-draft.spline_check.v1", "seed": SEED, "c": c, "results": []}
    for name, w in worlds.items():
        for G in (256, 384):
            reps = 1500 if w["beta"] == 0 else 800
            hits = {"M1": [], "M1s": []}
            dofs = {"M1": [], "M1s": []}
            for _ in range(reps):
                idx, g = dev.draw(rng, G)
                y = dev.base[idx] + w["beta"] * dev.fP[idx] + w["extra"][idx] + ss.errors(dev, rng, idx, g, "synthetic", dev.rho_obs)
                for label, X, col in (("M1", dev.X1[idx], dev.col1), ("M1s", X1s[idx], col1s)):
                    hits[label].append(hs.wcr_p(X, y, g, col, 399, rng) < ss.ALPHA)
            row = {"world": name, "G": G, "reps": reps,
                   **{f"{k}_wcr_rejection": round(float(np.mean(v)), 4) for k, v in hits.items()}}
            out["results"].append(row)
            print(row, flush=True)
    (HERE / "spline_check.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
