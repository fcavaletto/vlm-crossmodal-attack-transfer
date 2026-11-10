"""RGB image -> SmolVLM2 pixel tensor.

The official processor resizes **uint8** images with Lanczos and rounds after every resize
(1536-px upscale, then tiles or a 384-px resize). So the official model input is a piecewise-
constant function of the RGB image. ``surrogate_pixels`` follows the same geometry in float32,
with the same saturation at 0 and 255 but without the rounding; gradients go through it.
torchvision's uint8 antialiased resize is separable: a horizontal pass, saturated to 0..255, then a
vertical pass. The surrogate follows that order (checked on a development image on 2026-11-10: the
largest gap to the uint8 upscale fell from 4.3 to 1.1 levels against a single 2-D pass). ``official_pixels`` is the processor itself. The gap is measured, not assumed away.
"""

from __future__ import annotations

import math

import torch
import torchvision.transforms.v2.functional as F

from .pins import IMAGE_MEAN, IMAGE_STD, LONGEST_EDGE, TILE


def upscaled_size(height: int, width: int) -> tuple[int, int]:
    """The processor's longest-edge resize (``_resize_output_size_rescale_to_max_len``), then tile multiples."""
    aspect = width / height
    if width >= height:
        w = LONGEST_EDGE; h = int(w / aspect); h += h % 2
    else:
        h = LONGEST_EDGE; w = int(h * aspect); w += w % 2
    return h, w


def split_size(height: int, width: int) -> tuple[int, int]:
    """``resize_for_vision_encoder``: round both sides up to multiples of the tile."""
    h, w = upscaled_size(height, width)
    aspect = w / h
    if w >= h:
        w2 = math.ceil(w / TILE) * TILE; h2 = math.ceil(int(w2 / aspect) / TILE) * TILE
    else:
        h2 = math.ceil(h / TILE) * TILE; w2 = math.ceil(int(h2 * aspect) / TILE) * TILE
    return h2, w2


def tile_grid(height: int, width: int, tiling: str) -> tuple[int, int, int]:
    """(rows, columns, number of images including the global one)."""
    if tiling == "single":
        return 0, 0, 1
    if tiling != "split":
        raise ValueError(f"unknown tiling {tiling!r}")
    h, w = split_size(height, width)
    rows, cols = h // TILE, w // TILE
    return rows, cols, rows * cols + 1


def _resize(x: torch.Tensor, h: int, w: int, clamp: bool) -> torch.Tensor:
    if tuple(x.shape[-2:]) == (h, w):
        return x  # torchvision returns a same-size uint8 resize unchanged
    lanczos = F.InterpolationMode.LANCZOS
    sat = (lambda t: t.clamp(0.0, 255.0)) if clamp else (lambda t: t)
    y = sat(F.resize(x, [x.shape[-2], w], interpolation=lanczos, antialias=True))  # horizontal pass
    return sat(F.resize(y, [h, w], interpolation=lanczos, antialias=True))  # vertical pass


def _normalize(t255: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGE_MEAN, dtype=t255.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=t255.dtype).view(1, 3, 1, 1)
    return (t255 / 255.0 - mean) / std


def surrogate_pixels(image01: torch.Tensor, tiling: str, *, clamp: bool = True) -> torch.Tensor:
    """Differentiable float32 path: (3, H, W) in [0, 1] -> (n_images, 3, 384, 384), normalized."""
    if image01.ndim != 3 or image01.shape[0] != 3:
        raise ValueError("expected a (3, H, W) image")
    x = image01.to(torch.float32).unsqueeze(0) * 255.0
    h, w = image01.shape[-2:]
    big = _resize(x, *upscaled_size(h, w), clamp)
    if tiling == "single":
        t = _resize(big, TILE, TILE, clamp)
    else:
        big = _resize(big, *split_size(h, w), clamp)
        rows, cols, _ = tile_grid(h, w, tiling)
        parts = [big[..., r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(rows) for c in range(cols)]
        parts.append(_resize(big, TILE, TILE, clamp))
        t = torch.cat(parts, dim=0)
    return _normalize(t)


def official_pixels(processor, image01: torch.Tensor, tiling: str) -> torch.Tensor:
    """The processor's own tensor for the 8-bit rounding of ``image01``: (n_images, 3, 384, 384)."""
    from qwen_rgb.constraints import round_trip_uint8
    from vlm_iface.images import image_01_to_pil

    pil = image_01_to_pil(round_trip_uint8(image01.detach()))
    out = processor.image_processor(images=[[pil]], do_image_splitting=(tiling == "split"), return_tensors="pt")
    pv = out["pixel_values"][0].to(torch.float32)
    if "pixel_attention_mask" in out and not bool(out["pixel_attention_mask"].all()):
        raise ValueError("unexpected padding in the official pixel tensor")
    return pv


def levels(normalized_diff: torch.Tensor) -> torch.Tensor:
    """A normalized-pixel difference expressed in 8-bit levels, per channel."""
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return normalized_diff * std * 255.0
