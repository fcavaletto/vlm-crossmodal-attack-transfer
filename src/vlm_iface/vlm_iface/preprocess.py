"""One image preprocessor for scoring, gradients, and generation.

The official SmolVLM processor resizes uint8 pixels with Lanczos and then
normalizes. That uint8 resize does not accept autograd. The differentiable path
feeds float32 RGB in ``[0, 1]`` with ``do_rescale=False``, on CPU, so Lanczos
does not fall back to bicubic. Float32 Lanczos is not the same function as
uint8 Lanczos. ``compare_preprocess`` measures that gap.

``answer_logprob``, ``image_loss_gradient``, and ``generate`` all consume the
float32 path. They match each other. They do not match a PIL chat-template
forward. The chat template is used only to build text token ids; its pixel
tensor is discarded after a shape check.
"""

from __future__ import annotations

from typing import Any

import torch

from .images import image_01_to_pil


def differentiable_pixel_values(
    image_processor,
    image_01: torch.Tensor,
    *,
    do_image_splitting: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the official processor on a float CHW ``[0, 1]`` tensor and keep the graph.

    ``do_rescale`` is false because the tensor is already in ``[0, 1]``. Leaving
    it true divides the image by 255 a second time.
    """
    _require_cpu_chw(image_01)
    outputs = image_processor(
        images=image_01,
        return_tensors="pt",
        do_image_splitting=do_image_splitting,
        do_rescale=False,
    )
    return outputs["pixel_values"], outputs["pixel_attention_mask"]


def uint8_pil_pixel_values(
    image_processor,
    image_01: torch.Tensor,
    *,
    do_image_splitting: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official non-differentiable path: quantize to uint8 PIL, then default rescale."""
    _require_cpu_chw(image_01)
    pil = image_01_to_pil(image_01)
    outputs = image_processor(
        images=pil,
        return_tensors="pt",
        do_image_splitting=do_image_splitting,
    )
    return outputs["pixel_values"], outputs["pixel_attention_mask"]


def uint8_tensor_pixel_values(
    image_processor,
    image_01: torch.Tensor,
    *,
    do_image_splitting: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same uint8 values as the PIL path, passed as a tensor instead of a file."""
    _require_cpu_chw(image_01)
    quantized = image_01.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    outputs = image_processor(
        images=quantized,
        return_tensors="pt",
        do_image_splitting=do_image_splitting,
    )
    return outputs["pixel_values"], outputs["pixel_attention_mask"]


def compare_preprocess(
    image_processor,
    image_01: torch.Tensor,
    *,
    do_image_splitting: bool,
) -> dict[str, Any]:
    """Measure float32 pixels against the uint8 processor path. Does not build a graph."""
    with torch.no_grad():
        float_pixels, float_mask = differentiable_pixel_values(
            image_processor, image_01, do_image_splitting=do_image_splitting
        )
        pil_pixels, pil_mask = uint8_pil_pixel_values(
            image_processor, image_01, do_image_splitting=do_image_splitting
        )
        tensor_pixels, tensor_mask = uint8_tensor_pixel_values(
            image_processor, image_01, do_image_splitting=do_image_splitting
        )
        float_gap = (float_pixels - pil_pixels).abs()
        uint8_gap = (tensor_pixels - pil_pixels).abs()
    return {
        "do_image_splitting": do_image_splitting,
        "do_rescale_on_float_path": False,
        "float_pixel_shape": list(float_pixels.shape),
        "pil_pixel_shape": list(pil_pixels.shape),
        "shapes_match": tuple(float_pixels.shape) == tuple(pil_pixels.shape),
        "float_mask_matches_pil": torch.equal(float_mask, pil_mask),
        "uint8_tensor_matches_pil_mask": torch.equal(tensor_mask, pil_mask),
        "max_abs_diff_float_vs_pil": float(float_gap.max()),
        "mean_abs_diff_float_vs_pil": float(float_gap.mean()),
        "max_abs_diff_uint8_tensor_vs_pil": float(uint8_gap.max()),
        "mean_abs_diff_uint8_tensor_vs_pil": float(uint8_gap.mean()),
        "note": (
            "Float32 Lanczos resize is not uint8 Lanczos resize. "
            "Scoring and generation in this package share the float32 path. "
            "The PIL path is the processor default and is not differentiated."
        ),
    }


def reference_uint8_image(*, seed: int, height: int, width: int) -> torch.Tensor:
    """Deterministic RGB uint8 image converted to float CHW ``[0, 1]``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator)
    return raw.permute(2, 0, 1).to(dtype=torch.float32) / 255.0


def _require_cpu_chw(image_01: torch.Tensor) -> None:
    if image_01.ndim != 3 or image_01.shape[0] != 3:
        raise ValueError(f"image_01 must be CHW with 3 channels, got {tuple(image_01.shape)}")
    if image_01.device.type != "cpu":
        raise ValueError(
            "Image preprocessing must stay on CPU so Lanczos is not replaced by a GPU kernel."
        )
