"""Differentiable path from canonical RGB to Qwen ``pixel_values``.

The official torchvision processor resizes an 8-bit image and then fuses the
1/255 rescale into the normalization constants. This module multiplies a
float32 CHW image by 255, without rounding, and calls that processor's
``resize``, ``rescale_and_normalize``, and ``patchify``. The graph stays
attached to the original RGB tensor.

``smart_resize`` can leave the height and width unchanged. Torchvision then
returns the same tensor. That identity is still the official resize, and
gradients pass through it because the later arithmetic depends on the tensor.
"""

from __future__ import annotations

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)

from .pins import (
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_PIXELS,
    MERGE_SIZE,
    MIN_PIXELS,
    PATCH_SIZE,
    RESAMPLE_BICUBIC,
    TEMPORAL_PATCH_SIZE,
)


def build_pinned_processor() -> Qwen2VLImageProcessor:
    """Construct the pinned image processor without reading model weights."""
    processor = Qwen2VLImageProcessor(
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        merge_size=MERGE_SIZE,
        image_mean=list(IMAGE_MEAN),
        image_std=list(IMAGE_STD),
        resample=RESAMPLE_BICUBIC,
    )
    assert_pinned_processor(processor)
    return processor


def assert_pinned_processor(processor) -> None:
    """Refuse a processor whose geometry is not the pinned Qwen3-VL-2B config."""
    if int(processor.patch_size) != PATCH_SIZE:
        raise RuntimeError(f"patch_size {processor.patch_size} is not {PATCH_SIZE}")
    if int(processor.temporal_patch_size) != TEMPORAL_PATCH_SIZE:
        raise RuntimeError("temporal_patch_size does not match the pin")
    if int(processor.merge_size) != MERGE_SIZE:
        raise RuntimeError("merge_size does not match the pin")
    if tuple(processor.image_mean) != IMAGE_MEAN or tuple(processor.image_std) != IMAGE_STD:
        raise RuntimeError("image mean or std does not match the pin")
    if int(processor.size.shortest_edge) != MIN_PIXELS or int(processor.size.longest_edge) != MAX_PIXELS:
        raise RuntimeError("min or max pixels do not match the pin")
    if int(processor.resample) != RESAMPLE_BICUBIC:
        raise RuntimeError(f"resample {processor.resample} is not bicubic")
    if not processor.do_rescale or not processor.do_normalize:
        raise RuntimeError("the pinned processor rescales and normalizes")
    if float(processor.rescale_factor) != 1 / 255:
        raise RuntimeError("rescale factor is not 1/255")


def target_hw(height: int, width: int, processor) -> tuple[int, int]:
    """Height and width after the official ``smart_resize``."""
    assert_pinned_processor(processor)
    return smart_resize(
        int(height),
        int(width),
        factor=int(processor.patch_size) * int(processor.merge_size),
        min_pixels=int(processor.size.shortest_edge),
        max_pixels=int(processor.size.longest_edge),
    )


def differentiable_model_pixels(
    image_chw: torch.Tensor,
    processor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``pixel_values`` of shape ``[patches, dim]`` and ``image_grid_thw`` ``[1, 3]``.

    ``image_chw`` is float32, CPU, channel-first. Values may leave [0, 1]: the
    finite-difference probe does that on purpose. The canonical scene image is
    checked to lie in [0, 1] before this function is asked for a gradient.
    Clipping is a future constraint and is not applied on this path.
    """
    _require_cpu_float_chw(image_chw)
    assert_pinned_processor(processor)
    scaled = image_chw.unsqueeze(0) * 255.0
    resized = processor.resize(
        scaled,
        size=processor.size,
        resample=processor.resample,
        factor=int(processor.patch_size) * int(processor.merge_size),
    )
    normalized = processor.rescale_and_normalize(
        resized,
        do_rescale=True,
        rescale_factor=float(processor.rescale_factor),
        do_normalize=True,
        image_mean=processor.image_mean,
        image_std=processor.image_std,
    )
    patches, grid_h, grid_w = processor.patchify(
        normalized,
        patch_size=int(processor.patch_size),
        merge_size=int(processor.merge_size),
        temporal_patch_size=int(processor.temporal_patch_size),
    )
    if patches.ndim != 3 or patches.shape[0] != 1:
        raise RuntimeError(f"expected one image of patches, got {tuple(patches.shape)}")
    grid = torch.tensor([[1, int(grid_h), int(grid_w)]], dtype=torch.long)
    return patches[0], grid


def temporal_copies_match(pixel_values: torch.Tensor, processor) -> bool:
    """The single frame is repeated along the temporal patch axis."""
    channels = 3
    temporal = int(processor.temporal_patch_size)
    patch = int(processor.patch_size)
    expected = channels * temporal * patch * patch
    if pixel_values.ndim != 2 or pixel_values.shape[1] != expected:
        raise ValueError(f"pixel_values must be [patches, {expected}], got {tuple(pixel_values.shape)}")
    view = pixel_values.reshape(-1, channels, temporal, patch, patch)
    first = view[:, :, 0]
    return all(torch.equal(first, view[:, :, index]) for index in range(1, temporal))


def _require_cpu_float_chw(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a tensor")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be CHW with 3 channels, got {tuple(image.shape)}")
    if image.dtype != torch.float32:
        raise TypeError(f"differentiable RGB must be float32, got {image.dtype}")
    if image.device.type != "cpu":
        raise ValueError("RGB preprocessing stays on CPU so it uses the same resize as the processor")
    if not torch.isfinite(image).all():
        raise ValueError("image contains a non-finite value")
