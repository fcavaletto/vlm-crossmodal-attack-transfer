"""Canonical RGB images in float32 CHW, values in ``[0, 1]``.

Integer images are divided by 255. Floating images must already lie in
``[0, 1]``; values outside that interval are rejected rather than rescaled, so
a 0–255 float array cannot be mistaken for a probability image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def to_image_01(image: torch.Tensor | np.ndarray | Image.Image | str | Path) -> torch.Tensor:
    """Return a detached CPU float32 tensor of shape ``(3, height, width)``."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            array = np.asarray(handle.convert("RGB"))
        tensor = _from_numpy(array, already_rgb=True)
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
        tensor = _from_numpy(array, already_rgb=True)
    elif isinstance(image, np.ndarray):
        tensor = _from_numpy(image, already_rgb=False)
    elif isinstance(image, torch.Tensor):
        tensor = _from_torch(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW with 3 channels, got shape {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Image contains a non-finite value.")
    if tensor.min() < -1e-4 or tensor.max() > 1 + 1e-4:
        raise ValueError(
            "Float RGB images must be in [0, 1]. "
            f"Got min {float(tensor.min())} and max {float(tensor.max())}."
        )
    return tensor.clamp(0, 1).to(dtype=torch.float32, device="cpu").contiguous()


def image_01_to_pil(image_01: torch.Tensor) -> Image.Image:
    """Quantize CHW ``[0, 1]`` to an RGB PIL image. This cast is not differentiated."""
    if image_01.ndim != 3 or image_01.shape[0] != 3:
        raise ValueError("image_01 must be CHW with 3 channels")
    array = (
        image_01.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _from_numpy(array: np.ndarray, *, already_rgb: bool) -> torch.Tensor:
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {array.shape}")
    tensor = torch.from_numpy(np.array(array, copy=True))
    tensor = _channels_first(tensor, already_rgb=already_rgb)
    return _unit_interval(tensor)


def _from_torch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        raise ValueError("Batched images are not supported at this checkpoint. Pass one HWC or CHW image.")
    tensor = _channels_first(tensor.detach(), already_rgb=False)
    return _unit_interval(tensor)


def _channels_first(tensor: torch.Tensor, *, already_rgb: bool) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected HWC or CHW, got shape {tuple(tensor.shape)}")
    if already_rgb:
        return tensor.permute(2, 0, 1)
    channels_first = tensor.shape[0] == 3
    channels_last = tensor.shape[-1] == 3
    if channels_first and channels_last:
        raise ValueError(
            f"Ambiguous image shape {tuple(tensor.shape)}. Pass CHW only when the width is not 3."
        )
    if channels_first:
        return tensor
    if channels_last:
        return tensor.permute(2, 0, 1)
    raise ValueError(f"Could not find a channel axis of size 3 in shape {tuple(tensor.shape)}")


def _unit_interval(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        return tensor.to(dtype=torch.float32) / 255.0
    if not tensor.dtype.is_floating_point:
        raise TypeError(f"Unsupported image dtype {tensor.dtype}")
    return tensor.to(dtype=torch.float32)
