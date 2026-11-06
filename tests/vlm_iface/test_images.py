"""Canonical image conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vlm_iface.images import image_01_to_pil, to_image_01


def test_uint8_hwc_is_scaled_to_unit_interval():
    array = np.zeros((4, 5, 3), dtype=np.uint8)
    array[0, 0] = (255, 0, 128)
    image = to_image_01(array)
    assert image.shape == (3, 4, 5)
    assert image.dtype == torch.float32
    assert torch.allclose(image[:, 0, 0], torch.tensor([1.0, 0.0, 128 / 255]))


def test_float_chw_in_unit_interval_is_kept():
    source = torch.rand(3, 6, 7)
    image = to_image_01(source)
    assert torch.allclose(image, source)


def test_float_outside_unit_interval_is_rejected():
    source = torch.randint(0, 255, (3, 6, 7)).float()
    with pytest.raises(ValueError, match="0, 1"):
        to_image_01(source)


def test_ambiguous_shape_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        to_image_01(torch.zeros(3, 5, 3))


def test_pil_and_path_roundtrip(tmp_path: Path):
    image = Image.new("RGB", (5, 4), (10, 20, 30))
    path = tmp_path / "pixel.png"
    image.save(path)
    from_pil = to_image_01(image)
    from_path = to_image_01(path)
    assert torch.equal(from_pil, from_path)
    restored = image_01_to_pil(from_pil)
    assert restored.size == (5, 4)
    assert restored.getpixel((0, 0)) == (10, 20, 30)
