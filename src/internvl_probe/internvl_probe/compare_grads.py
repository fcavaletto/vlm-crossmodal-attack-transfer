"""bf16 against fp32 clean gradients on the same scenes (from the saved Stage 0 tensors). Model-free."""

from __future__ import annotations

import numpy as np

from .common import OUT_RELATIVE, repo_root, write_json


def main(tiling: str = "single") -> None:
    root = repo_root()
    d = root / OUT_RELATIVE / "tensors"
    a = np.load(d / f"clean_grad_bf16_{tiling}.npz")
    b = np.load(d / f"clean_grad_fp32_{tiling}.npz")
    rows = {}
    for sid in sorted(set(a.files) & set(b.files)):
        x, y = a[sid].ravel().astype(np.float64), b[sid].ravel().astype(np.float64)
        k = int(0.3 * x.size)
        tx, ty = np.argpartition(-np.abs(x), k)[:k], np.argpartition(-np.abs(y), k)[:k]
        rows[sid] = {"cosine": float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)),
                     "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
                     "top30_overlap": float(len(np.intersect1d(tx, ty)) / k), "top30_overlap_chance": 0.3,
                     "norm_ratio_bf16_over_fp32": float(np.linalg.norm(x) / np.linalg.norm(y))}
    out = {"label": "EXPLORATORY, development scenes", "tiling": tiling, "scenes": rows,
           "mean": {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}}
    print(write_json(root, f"grad_bf16_vs_fp32_{tiling}.json", out)); print(out["mean"])


if __name__ == "__main__":
    main()
