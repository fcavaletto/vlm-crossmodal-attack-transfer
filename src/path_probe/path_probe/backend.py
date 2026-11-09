"""The Qwen backend. Gradients and scores go through the frozen functions; this
module adds the fp32-projection readout and a margin gradient."""

from __future__ import annotations

from typing import Any

import torch
from attack_feasibility.score import backward_at_image, is_memory_error, release_cache, score_differentiable, score_official
from qwen_rgb.measure import move_to_device
from qwen_rgb.rgb_path import differentiable_model_pixels
from qwen_rgb_loss.measure import MemoryStop
from vlm_iface.devices import synchronize

from .precision import Fp32Projection


class QwenBackend:
    def __init__(self, *, model, processor, image_processor, prompts, texts, token_ids, label, device, projection: Fp32Projection):
        self.model = model
        self.processor = processor
        self.image_processor = image_processor
        self.prompts = prompts
        self.texts = texts
        self.token_ids = token_ids
        self.label = label
        self.device = device
        self.projection = projection

    def _with_projection(self, score: dict[str, Any]) -> dict[str, Any]:
        if score.get("loss_identity_agreed") is not True:
            raise RuntimeError("the labeled-color NLL formulas disagreed")
        out = {"nll": float(score["loss"]), "margin": float(score["margin"]), "chosen": score["chosen"],
               "summed_nll": dict(score["summed_nll"])}
        out.update(self.projection.score(self.label))
        return out

    def score_float(self, image: torch.Tensor, ctx: str) -> dict[str, Any]:
        self.projection.clear()
        score = score_differentiable(model=self.model, image_processor=self.image_processor, prompt=self.prompts[ctx],
                                     image=image, token_ids=self.token_ids, label=self.label, device=self.device)
        return self._with_projection(score)

    def score_official(self, image: torch.Tensor, ctx: str) -> dict[str, Any]:
        self.projection.clear()
        score = score_official(model=self.model, processor=self.processor, image=image, user_text=self.texts[ctx],
                               token_ids=self.token_ids, label=self.label, device=self.device)
        release_cache(self.device)
        return self._with_projection(score)

    def grad_nll(self, image: torch.Tensor, ctx: str) -> torch.Tensor:
        public, gradient = backward_at_image(model=self.model, image_processor=self.image_processor, prompt=self.prompts[ctx],
                                             image=image, token_ids=self.token_ids, label=self.label, device=self.device)
        self.projection.clear()
        if public.get("loss_identity_agreed") is not True or gradient is None:
            raise RuntimeError("the labeled-color NLL formulas disagreed during a gradient")
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("a gradient was not finite")
        return gradient

    def grad_margin(self, image: torch.Tensor, ctx: str, comp: str) -> torch.Tensor:
        """RGB gradient of z_label - z_comp on the same differentiable path as the frozen gradient."""
        a = int(self.token_ids[self.label][0])
        b = int(self.token_ids[comp][0])
        leaf = image.detach().to(dtype=torch.float32, device="cpu").requires_grad_(True)
        try:
            pixels, grid = differentiable_model_pixels(leaf, self.image_processor)
            if not torch.equal(grid.cpu(), self.prompts[ctx]["image_grid_thw"].cpu()):
                raise RuntimeError("a step changed the patch grid")
            moved = move_to_device(self.prompts[ctx], self.device)
            moved["pixel_values"] = pixels.to(self.device)
            outputs = self.model(**moved, use_cache=False)
            row = outputs.logits[0, -1]
            margin = row[a].float() - row[b].float()
            margin.backward()
            synchronize(self.device)
            gradient = leaf.grad.detach().to(dtype=torch.float32, device="cpu").clone()
            del outputs, moved, pixels, row, margin
            return gradient
        except RuntimeError as exc:
            if is_memory_error(exc):
                raise MemoryStop(str(exc)) from exc
            raise
        finally:
            leaf.grad = None
            self.projection.clear()
            release_cache(self.device)
