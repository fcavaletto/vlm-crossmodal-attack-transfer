"""SmolVLM2-2.2B forward, scores and RGB gradients.

Scores: the official processor path (8-bit image -> uint8 Lanczos resizes -> tiles) or the float
surrogate (``pixels.surrogate_pixels``). Gradients always go through the surrogate.
The readout is the first answer-token row, as in every earlier checkpoint:
summed NLL of the four one-token colours over the full vocabulary, and
margin = best other colour's NLL minus the label's NLL (a logit difference).
Note: SmolVLM drops an image whose normalized pixels are all exactly 0 (a 127.5 gray); the
uniform image used for the image-dependence check is 128/255, so it is kept.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

from .common import Stop, snapshot_dir
from .pins import ANSWER_PREFIX, EXPECTED_TOKEN_IDS, IMAGE_TOKEN, TOKENS_PER_TILE
from .pixels import official_pixels, surrogate_pixels, tile_grid

COLORS = ("red", "blue", "green", "yellow")
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def release(device) -> None:
    gc.collect()
    if getattr(device, "type", None) == "mps":
        torch.mps.empty_cache()


def color_token_ids(processor) -> dict[str, int]:
    """One token per colour after the answer cue ("Assistant:" then " red"), checked jointly."""
    tok = processor.tokenizer
    ids = {}
    for c in COLORS:
        enc = tok.encode(" " + c, add_special_tokens=False)
        if len(enc) != 1:
            raise Stop(f"{c!r} is not one token: {enc}")
        joint = tok.encode(ANSWER_PREFIX + " " + c, add_special_tokens=False)
        alone = tok.encode(ANSWER_PREFIX, add_special_tokens=False)
        if joint != alone + enc:
            raise Stop(f"{c!r} does not tokenize the same after the answer cue")
        if tok.decode(enc).strip() != c:
            raise Stop(f"{c!r} decodes to {tok.decode(enc)!r}")
        ids[c] = int(enc[0])
    if ids != EXPECTED_TOKEN_IDS:
        raise Stop(f"colour ids {ids} differ from the locked {EXPECTED_TOKEN_IDS}")
    return ids


class SmolVLM:
    def __init__(self, dtype: str, tiling: str, device: str = "mps", attn: str = "eager"):
        if dtype not in DTYPES:
            raise Stop(f"unknown dtype {dtype!r}")
        self.dtype_name, self.dtype, self.tiling = dtype, DTYPES[dtype], tiling
        self.device = torch.device(device)
        self.attn = attn
        self.model = None
        self.processor = None

    def load(self) -> dict[str, Any]:
        from transformers import AutoModelForImageTextToText

        from .common import load_processor

        self.processor = load_processor()
        self.token_ids = color_token_ids(self.processor)
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot_dir()), dtype=self.dtype, attn_implementation=self.attn)
        model.requires_grad_(False)
        model.eval()
        self.model = model.to(self.device)
        dtypes = sorted({str(p.dtype) for p in self.model.parameters()})
        return {"param_dtypes": dtypes, "n_params": int(sum(p.numel() for p in self.model.parameters())),
                "attn_implementation": self.model.config._attn_implementation, "token_ids": self.token_ids,
                "model_class": type(self.model).__name__}

    # prompts ------------------------------------------------------------------------------------
    def prompt(self, image01: torch.Tensor, user_text: str) -> dict[str, torch.Tensor]:
        from qwen_rgb.constraints import round_trip_uint8
        from vlm_iface.images import image_01_to_pil

        messages = [{"role": "user", "content": [{"type": "image", "image": image_01_to_pil(round_trip_uint8(image01))},
                                                 {"type": "text", "text": user_text}]}]
        enc = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                 return_tensors="pt", images_kwargs={"do_image_splitting": self.tiling == "split"})
        n = tile_grid(*image01.shape[-2:], self.tiling)[2]
        img_id = self.processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if int((enc["input_ids"] == img_id).sum()) != n * TOKENS_PER_TILE or enc["pixel_values"].shape[1] != n:
            raise Stop("image token count or tile count is not the expected one")
        tail = self.processor.tokenizer.decode(enc["input_ids"][0][-6:])
        if not tail.endswith(ANSWER_PREFIX):
            raise Stop(f"the prompt does not end with the answer cue: {tail!r}")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "official_pixels": enc["pixel_values"][0]}

    # forward ------------------------------------------------------------------------------------
    def _row(self, prompt: dict[str, torch.Tensor], pixels: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=prompt["input_ids"].to(self.device), attention_mask=prompt["attention_mask"].to(self.device),
                         pixel_values=pixels.unsqueeze(0).to(self.device, self.dtype), use_cache=False, logits_to_keep=1)
        return out.logits[0, -1]

    def _readout(self, row: torch.Tensor, label: str) -> dict[str, Any]:
        logp = torch.log_softmax(row.detach().float().cpu(), dim=-1)
        nll = {c: float(-logp[self.token_ids[c]]) for c in COLORS}
        chosen = min(COLORS, key=lambda c: (nll[c], c))
        margin = min(nll[c] for c in COLORS if c != label) - nll[label]
        return {"nll": nll, "chosen": chosen, "margin": float(margin), "label_nll": nll[label],
                "top_token": int(row.argmax())}

    @torch.no_grad()
    def score(self, image01: torch.Tensor, prompt: dict[str, torch.Tensor], label: str, path: str = "official") -> dict[str, Any]:
        if path == "official":
            pix = official_pixels(self.processor, image01, self.tiling)
        elif path == "surrogate":
            pix = surrogate_pixels(image01, self.tiling)
        else:
            raise Stop(f"unknown path {path!r}")
        try:
            return self._readout(self._row(prompt, pix), label)
        finally:
            release(self.device)

    def objective(self, row: torch.Tensor, label: str, kind: str) -> torch.Tensor:
        logp = torch.log_softmax(row.float(), dim=-1)
        if kind == "nll":
            return -logp[self.token_ids[label]]
        if kind == "margin":
            others = torch.stack([logp[self.token_ids[c]] for c in COLORS if c != label])
            return logp[self.token_ids[label]] - others.max()
        raise Stop(f"unknown objective {kind!r}")

    @torch.no_grad()
    def objective_value(self, image01: torch.Tensor, prompt, label: str, kind: str = "nll") -> float:
        try:
            return float(self.objective(self._row(prompt, surrogate_pixels(image01, self.tiling)), label, kind))
        finally:
            release(self.device)

    def gradient(self, image01: torch.Tensor, prompt, label: str, kind: str = "nll") -> tuple[float, torch.Tensor]:
        """Value and RGB gradient of the objective on the surrogate path (float32, CPU)."""
        leaf = image01.detach().to(torch.float32).clone().requires_grad_(True)
        try:
            value = self.objective(self._row(prompt, surrogate_pixels(leaf, self.tiling)), label, kind)
            value.backward()
            if self.device.type == "mps":
                torch.mps.synchronize()
            g = leaf.grad.detach().clone()
            if not bool(torch.isfinite(g).all()):
                raise Stop("a gradient was not finite")
            return float(value.detach()), g
        finally:
            leaf.grad = None
            release(self.device)
