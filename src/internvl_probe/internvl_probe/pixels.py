"""RGB image -> InternVL pixel tensor.

The official processor resizes **uint8** images (bicubic, antialiased) and so rounds after the
resize. That makes the official model input a piecewise-constant function of the RGB image.
``surrogate_pixels`` is the same geometry in float32 without the rounding; it is the path that
gradients are taken through. ``official_pixels`` is the processor itself. The gap between the
two is measured, not assumed away.
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as F

from .pins import IMAGE_MEAN, IMAGE_STD, MAX_PATCHES, MIN_PATCHES, TILE


def tile_grid(height: int, width: int, tiling: str) -> tuple[int, int, int]:
    """(columns, rows, number of tiles including the thumbnail)."""
    if tiling == "single":
        return 1, 1, 1
    if tiling != "dynamic":
        raise ValueError(f"unknown tiling {tiling!r}")
    from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

    cols, rows = get_optimal_tiled_canvas((height, width), (TILE, TILE), MIN_PATCHES, MAX_PATCHES)
    n = cols * rows
    return cols, rows, n + (1 if n != 1 else 0)


def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.resize(x, [h, w], interpolation=F.InterpolationMode.BICUBIC, antialias=True)


def _normalize(t255: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGE_MEAN, dtype=t255.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=t255.dtype).view(1, 3, 1, 1)
    return (t255 / 255.0 - mean) / std


def surrogate_pixels(image01: torch.Tensor, tiling: str, *, clamp: bool = True) -> torch.Tensor:
    """Differentiable float32 path: (3, H, W) in [0, 1] -> (n_tiles, 3, 448, 448), normalized."""
    if image01.ndim != 3 or image01.shape[0] != 3:
        raise ValueError("expected a (3, H, W) image")
    x = image01.to(torch.float32).unsqueeze(0) * 255.0
    h, w = image01.shape[-2:]
    cols, rows, n = tile_grid(h, w, tiling)
    if n == 1:
        parts = [_resize(x, TILE, TILE)]
    else:
        big = _resize(x, rows * TILE, cols * TILE)
        parts = [big[..., r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] for r in range(rows) for c in range(cols)]
        parts.append(_resize(x, TILE, TILE))
    t = torch.cat(parts, dim=0)
    if clamp:
        t = t.clamp(0.0, 255.0)
    return _normalize(t)


def official_pixels(processor, image01: torch.Tensor, tiling: str) -> torch.Tensor:
    """The processor's own tensor for the 8-bit rounding of ``image01``."""
    from qwen_rgb.constraints import round_trip_uint8
    from vlm_iface.images import image_01_to_pil

    pil = image_01_to_pil(round_trip_uint8(image01.detach()))
    out = processor.image_processor(images=pil, crop_to_patches=(tiling == "dynamic"), return_tensors="pt")
    return out["pixel_values"].to(torch.float32)


def levels(normalized_diff: torch.Tensor) -> torch.Tensor:
    """A normalized-pixel difference expressed in 8-bit levels, per channel."""
    std = torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return normalized_diff * std * 255.0
