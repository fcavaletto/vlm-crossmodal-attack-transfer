"""Model-free: clean-gradient similarity between precisions (PLAN.md section 4), from saved tensors.

    python -m qwen_precision.compare_grads <run_dir> [--mps-dir docs/checkpoint-2.3-exploratory/paths/tensors]

Per scene and gradient (L = NLL, M = margin; both contexts): cosine, sign agreement, norm ratio,
and overlap of the top-30% |gradient| coordinates (chance 0.30). bf16 vs fp32 on CUDA, and,
when the MPS tensors are present, CUDA bf16 vs MPS bf16 (same scenes, cross-device).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("L_neutral", "M_neutral", "L_helpful_nonrevealing", "M_helpful_nonrevealing")


def sim(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    k = int(0.3 * x.size)
    tx, ty = np.argpartition(-np.abs(x), k)[:k], np.argpartition(-np.abs(y), k)[:k]
    both = (x != 0) & (y != 0)
    return {"cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
            "sign_agreement": float(np.mean(np.sign(x[both]) == np.sign(y[both]))),
            "norm_ratio_a_over_b": float(np.linalg.norm(x) / np.linalg.norm(y)),
            "top30_overlap": float(np.intersect1d(tx, ty).size / k)}


def summarize(rows: dict) -> dict:
    out = {}
    for key in KEYS:
        vals = [r[key] for r in rows.values() if key in r]
        if vals:
            out[key] = {m: {"mean": float(np.mean([v[m] for v in vals])), "min": float(np.min([v[m] for v in vals])),
                            "max": float(np.max([v[m] for v in vals]))} for m in vals[0]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mps-dir")
    a = ap.parse_args()
    run = Path(a.run_dir)
    pairs, cross = {}, {}
    for f in sorted((run / "bf16" / "tensors").glob("*.npz")):
        sid = f.stem
        g = run / "fp32" / "tensors" / f.name
        if not g.is_file():
            continue
        with np.load(f) as b16, np.load(g) as f32:
            pairs[sid] = {k: sim(b16[f"grad_{k}"], f32[f"grad_{k}"]) for k in KEYS}
            if a.mps_dir:
                m = Path(a.mps_dir) / f"{sid}__B__grads.npz"
                if m.is_file():
                    with np.load(m) as mps:
                        cross[sid] = {k: sim(b16[f"grad_{k}"], mps[k]) for k in KEYS}
    out = {"label": "EXPLORATORY, development scenes", "n_scenes": len(pairs),
           "bf16_vs_fp32_cuda": {"summary": summarize(pairs), "scenes": pairs},
           "cuda_bf16_vs_mps_bf16": {"summary": summarize(cross), "scenes": cross} if cross else None}
    dest = run / "gradient_similarity.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(dest, json.dumps(out["bf16_vs_fp32_cuda"]["summary"].get("L_neutral", {})))


if __name__ == "__main__":
    main()
