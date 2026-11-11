"""Precision instruments.

- ``Fp32Projection``: a forward hook on ``lm_head`` that keeps the last
  position's final hidden state. The labeled-color NLL and margin are then
  recomputed with an fp32 matmul against the same weights. This removes the
  bf16 rounding of the logits only. Everything upstream stays as it ran.
- ``bf16_input_survival``: model-free. How much of an RGB perturbation survives
  the cast of normalized pixel values to bf16.
- ``VisionPrecision``: converts only the bf16 floating tensors of the vision
  tower to fp32 and restores exactly those tensors to bf16 afterwards. fp32 is an
  exact superset of bf16, so the restore is lossless.
"""

from __future__ import annotations

from typing import Any

import torch

CHUNK_ROWS = 16384


class Fp32Projection:
    def __init__(self, lm_head: torch.nn.Module, token_ids: dict[str, tuple[int, ...]]):
        self.lm_head = lm_head
        self.colors = tuple(token_ids)
        self.ids = [int(token_ids[c][0]) for c in self.colors]
        self.hidden: torch.Tensor | None = None
        self.handle = lm_head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        h = inputs[0]
        self.hidden = h[0, -1].detach().to(torch.float32).clone()

    def take(self) -> torch.Tensor:
        if self.hidden is None:
            raise RuntimeError("no lm_head forward was captured")
        h, self.hidden = self.hidden, None
        return h

    def score(self, label: str) -> dict[str, Any]:
        """fp32 labeled-color summed NLL (one token, so -log softmax) and margin."""
        h = self.take()
        weight = self.lm_head.weight
        with torch.no_grad():
            hd = h.to(weight.device)
            parts = []
            for start in range(0, weight.shape[0], CHUNK_ROWS):
                block = weight[start : start + CHUNK_ROWS].to(torch.float32)
                parts.append(block @ hd)
            logits = torch.cat(parts)
            log_z = float(torch.logsumexp(logits, dim=0).cpu())
            z = {c: float(logits[i].cpu()) for c, i in zip(self.colors, self.ids)}
            del logits, parts
        nll = {c: log_z - z[c] for c in self.colors}
        best_other = max(z[c] for c in self.colors if c != label)
        return {"nll32": nll[label], "margin32": z[label] - best_other, "summed_nll32": nll}

    def clear(self) -> None:
        self.hidden = None

    def remove(self) -> None:
        self.handle.remove()


def bf16_input_survival(
    pixel_fn,
    image: torch.Tensor,
    epsilon: float,
    fractions: tuple[float, ...],
    seed: int,
) -> list[dict[str, Any]]:
    """For x + t*eps*s with a fixed Rademacher s: what the bf16 cast keeps.

    ``pixel_fn(image) -> pixel_values`` is the differentiable processor path.
    """
    gen = torch.Generator().manual_seed(int(seed))
    s = torch.randint(0, 2, tuple(image.shape), generator=gen).to(torch.float32).mul(2).sub(1)
    p0 = pixel_fn(image)
    q0 = p0.to(torch.bfloat16)
    rows = []
    for t in fractions:
        pt = pixel_fn(image + float(t) * float(epsilon) * s)
        qt = pt.to(torch.bfloat16)
        true = (pt - p0).double()
        kept = (qt.double() - q0.double())
        moved = true.abs() > 0
        n = int(moved.sum())
        rows.append({
            "t": float(t),
            "n_coordinates": int(true.numel()),
            "n_moved_fp32": n,
            "frac_changed_bf16_among_moved": float(((qt != q0) & moved).sum()) / max(n, 1),
            "relative_l2_error_bf16": float((kept - true).norm() / true.norm()) if n else None,
            "kept_l1_over_true_l1": float(kept.abs().sum() / true.abs().sum()) if n else None,
            "true_step_normalized_median": float(true.abs()[moved].median()) if n else None,
        })
    return rows


class VisionPrecision:
    def __init__(self, visual: torch.nn.Module):
        self.visual = visual
        self.converted: list[tuple[str, str]] = []

    def fp32_bytes(self) -> int:
        total = 0
        for t in list(self.visual.parameters()) + list(self.visual.buffers()):
            if t.dtype == torch.bfloat16:
                total += t.numel() * 4
        return total

    def to_fp32(self) -> int:
        if self.converted:
            raise RuntimeError("the vision tower is already converted")
        count = 0
        for kind, items in (("param", self.visual.named_parameters()), ("buffer", self.visual.named_buffers())):
            for name, t in items:
                if t.dtype == torch.bfloat16:
                    t.data = t.data.to(torch.float32)
                    self.converted.append((kind, name))
                    count += t.numel()
        return count

    def restore(self) -> int:
        lookup = {("param", n): t for n, t in self.visual.named_parameters()}
        lookup.update({("buffer", n): t for n, t in self.visual.named_buffers()})
        count = 0
        for key in self.converted:
            t = lookup[key]
            t.data = t.data.to(torch.bfloat16)
            count += t.numel()
        self.converted = []
        return count
