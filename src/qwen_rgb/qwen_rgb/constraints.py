"""Future RGB feasible set. These functions are the constraint, not an attack.

A later attack would have to land in this set. This checkpoint does not take a
signed step, and it does not iterate.
"""

from __future__ import annotations

import torch


def clip_unit_interval(image: torch.Tensor) -> torch.Tensor:
    """Clip every channel to the closed interval [0, 1]."""
    _require_float_image(image)
    return image.clamp(0, 1)


def round_trip_uint8(image: torch.Tensor) -> torch.Tensor:
    """Map float RGB through the 8-bit storage used by ``image_01_to_pil``.

    PyTorch ``round`` is half-to-even. The result is a multiple of 1/255 in
    [0, 1]. The rounding is not differentiated: the returned tensor is detached.
    """
    _require_float_image(image)
    quantized = image.detach().clamp(0, 1).mul(255).round().to(dtype=torch.float32)
    return quantized / 255.0


def linf_distance(image: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """||image - clean||_∞ as a 0-dimensional tensor."""
    _require_same_shape(image, clean)
    return (image - clean).abs().max()


def project_linf(image: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project onto the L-infinity ball of radius ``epsilon`` and then onto [0, 1].

    ``clean`` is the unattacked canonical image. ``epsilon`` is in RGB units on
    the [0, 1] scale. This checkpoint does not choose a study value of epsilon.
    """
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real number")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    _require_same_shape(image, clean)
    clipped_delta = (image - clean).clamp(-float(epsilon), float(epsilon))
    return clip_unit_interval(clean + clipped_delta)


def store_feasible(image: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project onto the future feasible set and then apply the 8-bit round trip.

    The round trip can move a pixel by at most half a level, 0.5/255, so the
    stored image can sit slightly outside the continuous L-infinity ball.
    """
    return round_trip_uint8(project_linf(image, clean, epsilon))


def _require_float_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor) or not image.dtype.is_floating_point:
        raise TypeError("image must be a floating-point tensor")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be CHW with 3 channels, got {tuple(image.shape)}")


def _require_same_shape(image: torch.Tensor, clean: torch.Tensor) -> None:
    _require_float_image(image)
    _require_float_image(clean)
    if tuple(image.shape) != tuple(clean.shape):
        raise ValueError(f"image shape {tuple(image.shape)} does not match clean {tuple(clean.shape)}")
