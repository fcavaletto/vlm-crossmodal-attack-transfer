"""Compare the differentiable RGB path with the official processor.

Every returned gap is a measured distance on one image. A forced resize probe
reports the uint8 rounding gap of the bicubic kernel. That probe is not the
tensor sent to the model unless ``smart_resize`` actually changes the image.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF

from .pins import OFFICIAL_MATCH_MAX_ABS
from .rgb_path import differentiable_model_pixels, target_hw, temporal_copies_match


def official_inputs(processor, image, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the public processor call and return pixels and the grid."""
    outputs = processor(images=image, return_tensors="pt", **kwargs)
    return outputs["pixel_values"], outputs["image_grid_thw"]


def distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Max and mean absolute difference, or a shape mismatch."""
    if tuple(left.shape) != tuple(right.shape):
        return {
            "shapes_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "max_abs": None,
            "mean_abs": None,
        }
    gap = (left.detach().float() - right.detach().float()).abs()
    return {
        "shapes_match": True,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "max_abs": float(gap.max()),
        "mean_abs": float(gap.mean()),
    }


def compare_image(image_01: torch.Tensor, processor) -> dict[str, Any]:
    """Compare one CHW ``[0, 1]`` image with the official processor paths.

    The image is the canonical float tensor. The PIL and uint8 paths quantize
    with the same half-to-even round the study uses for stored images.
    """
    image_01 = image_01.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if image_01.ndim != 3 or image_01.shape[0] != 3:
        raise ValueError("image_01 must be CHW")
    height, width = int(image_01.shape[1]), int(image_01.shape[2])
    resized_h, resized_w = target_hw(height, width, processor)
    ours, our_grid = differentiable_model_pixels(image_01, processor)
    uint8 = image_01.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    pil = Image.fromarray(uint8.permute(1, 2, 0).contiguous().numpy().copy(), mode="RGB")
    pil_pixels, pil_grid = official_inputs(processor, pil)
    uint8_pixels, uint8_grid = official_inputs(processor, uint8)
    float_pixels, float_grid = official_inputs(processor, uint8.float())
    unit_pixels, unit_grid = official_inputs(processor, image_01, do_rescale=False)
    pil_distance = distance(ours, pil_pixels)
    bf16 = pil_pixels.detach().to(dtype=torch.bfloat16).float()
    bf16_gap = (pil_pixels.detach().float() - bf16).abs()
    within = bool(
        pil_distance["shapes_match"]
        and pil_distance["max_abs"] is not None
        and pil_distance["max_abs"] <= OFFICIAL_MATCH_MAX_ABS
        and torch.equal(our_grid, pil_grid.cpu())
    )
    return {
        "input_hw": [height, width],
        "smart_resize_hw": [resized_h, resized_w],
        "resize_is_identity": (resized_h, resized_w) == (height, width),
        "our_shape": list(ours.shape),
        "our_grid": our_grid.tolist(),
        "temporal_copies_equal": bool(temporal_copies_match(ours, processor)),
        "versus_pil": pil_distance,
        "versus_uint8_tensor": distance(ours, uint8_pixels),
        "versus_float_0_255": distance(ours, float_pixels),
        "versus_unit_interval_do_rescale_false": distance(ours, unit_pixels),
        "grid_matches_pil": torch.equal(our_grid, pil_grid.cpu()),
        "grid_matches_uint8_tensor": torch.equal(our_grid, uint8_grid.cpu()),
        "grid_matches_float_0_255": torch.equal(our_grid, float_grid.cpu()),
        "grid_matches_unit_interval": torch.equal(our_grid, unit_grid.cpu()),
        "bf16_cast_of_official_pixels": {
            "max_abs": float(bf16_gap.max()),
            "mean_abs": float(bf16_gap.mean()),
            "note": (
                "The vision tower casts pixel_values to its parameter dtype. "
                "This is the gap of that cast on the official float32 pixels, "
                "measured without loading the model."
            ),
        },
        "official_match_max_abs": OFFICIAL_MATCH_MAX_ABS,
        "within_official_match": within,
        "resize_rounding_probe": resize_rounding_probe(uint8, processor),
    }


def resize_rounding_probe(uint8_chw: torch.Tensor, processor) -> dict[str, Any]:
    """Bicubic uint8 versus bicubic float on a smaller legal canvas.

    The probe size is fixed at 256 by 320, which is divisible by
    ``patch_size * merge_size``. It is not the model input for an image whose
    official resize is the identity.
    """
    out_h, out_w = 256, 320
    if uint8_chw.dtype != torch.uint8 or uint8_chw.ndim != 3:
        raise ValueError("the rounding probe starts from a uint8 CHW image")
    uint8_resized = tvF.resize(
        uint8_chw,
        [out_h, out_w],
        interpolation=tvF.InterpolationMode.BICUBIC,
        antialias=True,
    )
    float_resized = tvF.resize(
        uint8_chw.float(),
        [out_h, out_w],
        interpolation=tvF.InterpolationMode.BICUBIC,
        antialias=True,
    )
    uint8_pixels, _ = differentiable_model_pixels(uint8_resized.float() / 255.0, processor)
    float_pixels, _ = differentiable_model_pixels(float_resized / 255.0, processor)
    # The call above runs smart_resize again. 256x320 is already legal, so the
    # second resize is an identity and the only difference is the first kernel.
    gap = (uint8_pixels.detach().float() - float_pixels.detach().float()).abs()
    return {
        "used_as_model_input": False,
        "probe_hw": [out_h, out_w],
        "uint8_resize_dtype": str(uint8_resized.dtype),
        "max_abs_normalized_patches": float(gap.max()),
        "mean_abs_normalized_patches": float(gap.mean()),
        "note": (
            "Torchvision's uint8 bicubic resize rounds. The float resize does not. "
            "This probe forces a size change. It is not the model input when "
            "smart_resize keeps the original height and width."
        ),
    }


def allows_model_load(comparison: dict[str, Any]) -> bool:
    """True only when the differentiable pixels match the official PIL pixels."""
    return bool(comparison["within_official_match"] and comparison["grid_matches_pil"])
