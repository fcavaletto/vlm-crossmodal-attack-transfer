"""The frozen QwenBackend, observed but not changed.

``RecordingBackend`` subclasses path_probe.backend.QwenBackend and only adds bookkeeping:
it keeps every NLL gradient (so PGD's step-2 and step-3 gradients are available for the
alignment diagnostics) and, for the FGSM and PGD-3 images, scores the same 8-bit image on
the differentiable float path as well. The frozen scene_block calls the official scorer in a
fixed order: 2 clean calls, then per source 11 images x 2 contexts (FGSM, PGD-3, top-k,
8 random). Positions are checked by the runner after every scene.
"""

from __future__ import annotations

from typing import Any

import torch
from path_probe.backend import QwenBackend

N_CLEAN_CALLS = 2
CALLS_PER_SOURCE = 22
FLOAT_KINDS = ("fgsm", "fgsm", "pgd3", "pgd3")  # the first 4 official calls of a source block


class RecordingBackend(QwenBackend):
    def __init__(self, *args, float_check: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.float_check = float_check
        self.nll_grads: list[tuple[str, torch.Tensor]] = []
        self.official_ctx: list[str] = []
        self.float_scores: list[dict[str, Any]] = []

    def grad_nll(self, image: torch.Tensor, ctx: str) -> torch.Tensor:
        g = super().grad_nll(image, ctx)
        self.nll_grads.append((ctx, g))
        return g

    def score_official(self, image: torch.Tensor, ctx: str) -> dict[str, Any]:
        from qwen_rgb.constraints import round_trip_uint8

        out = super().score_official(image, ctx)
        i = len(self.official_ctx)
        self.official_ctx.append(ctx)
        j = i - N_CLEAN_CALLS
        if self.float_check and j >= 0 and j % CALLS_PER_SOURCE < len(FLOAT_KINDS):
            f = super().score_float(round_trip_uint8(image), ctx)
            self.float_scores.append({"call": i, "source_block": j // CALLS_PER_SOURCE,
                                      "kind": FLOAT_KINDS[j % CALLS_PER_SOURCE], "context": ctx,
                                      "official": {k: out[k] for k in ("nll", "margin", "chosen")},
                                      "float": {k: f[k] for k in ("nll", "margin", "chosen")}})
        return out
