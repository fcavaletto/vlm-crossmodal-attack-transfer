"""Model-free Stage 0 gate: preprocessing geometry, surrogate-vs-official gap, perturbation survival.

    python -m smolvlm_probe.pixel_stage
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .common import development_scenes, git_head, load_processor, repo_root, scene_image, write_json
from .pins import IMAGE_MEAN, IMAGE_STD, TILINGS, TOKENS_PER_TILE
from .pixels import levels, official_pixels, surrogate_pixels, tile_grid

SEED = 20261110


def _gap(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = levels(a - b).abs()
    return {"max_levels": float(d.max()), "mean_levels": float(d.mean()),
            "frac_over_half_level": float((d > 0.5).float().mean()), "frac_over_1_level": float((d > 1.0 + 1e-4).float().mean())}


def main() -> None:
    root = repo_root()
    processor = load_processor()
    rows = development_scenes(root)
    gen = torch.Generator().manual_seed(SEED)
    out = {"schema": "checkpoint-6.stage0.pixels.v1", "label": "MODEL-FREE, development images only",
           "git_head": git_head(root), "n_images": len(rows), "tilings": {}}
    t0 = time.time()
    for tiling in TILINGS:
        geom, gaps, gaps_round, surv = [], [], [], []
        for row in rows:
            img = scene_image(root, row)
            off = official_pixels(processor, img, tiling)
            sur = surrogate_pixels(img, tiling)
            cols, r, n = tile_grid(*img.shape[-2:], tiling)
            geom.append((tuple(off.shape) == tuple(sur.shape)) and off.shape[0] == n)
            gaps.append(_gap(sur, off))
            # rounding the surrogate in 0..255 space before normalising, i.e. a float model of the uint8 kernel
            std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1); mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
            t255 = ((sur * std + mean) * 255).round()
            gaps_round.append(_gap((t255 / 255 - mean) / std, off))
            # survival of one-level RGB perturbations through the official (uint8) path
            sign = torch.randint(0, 2, img.shape, generator=gen).float().mul(2).sub(1)
            sparse = (torch.rand(img.shape, generator=gen) < 0.30).float()
            row_s = {}
            for name, delta in (("dense_1level", sign / 255), ("sparse30_1level", sign * sparse / 255)):
                pert = (img + delta).clamp(0, 1)
                off_p = official_pixels(processor, pert, tiling)
                sur_p = surrogate_pixels(pert, tiling)
                true = levels(sur_p - sur)
                kept = levels(off_p - off)
                moved = true.abs() > 1e-6
                row_s[name] = {
                    "frac_inputs_changed_official": float((kept != 0)[moved].float().mean()),
                    "kept_l1_over_float_l1": float(kept.abs().sum() / true.abs().sum()),
                    "cos_official_vs_float": float((kept * true).sum() / (kept.norm() * true.norm() + 1e-12)),
                    "median_float_change_levels": float(true.abs()[moved].median()),
                    "frac_changed_by_bf16_cast_official": float(((off_p.bfloat16() != off.bfloat16()) & (kept != 0)).float().sum() / max(int((kept != 0).sum()), 1)),
                }
            surv.append(row_s)
        def agg(items, key=None):
            keys = items[0].keys()
            return {k: {"mean": float(np.mean([x[k] for x in items])), "max": float(np.max([x[k] for x in items])),
                        "min": float(np.min([x[k] for x in items]))} for k in keys}
        out["tilings"][tiling] = {
            "tiles": tile_grid(384, 512, tiling)[2],
            "image_tokens": tile_grid(384, 512, tiling)[2] * TOKENS_PER_TILE,
            "geometry_matches_processor": all(geom),
            "surrogate_vs_official": agg(gaps),
            "rounded_surrogate_vs_official": agg(gaps_round),
            "survival": {name: agg([s[name] for s in surv]) for name in surv[0]},
        }
        print(tiling, "done", round(time.time() - t0, 1), "s")
    path = write_json(root, "pixels.json", out)
    print(path)


if __name__ == "__main__":
    main()
