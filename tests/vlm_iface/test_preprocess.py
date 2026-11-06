"""Float preprocess stays on the autograd path and is measured against uint8 PIL."""

from __future__ import annotations

import pytest
import torch

from vlm_iface.preprocess import (
    compare_preprocess,
    differentiable_pixel_values,
    reference_uint8_image,
)


def test_float_preprocess_keeps_a_finite_image_gradient(processor):
    image = reference_uint8_image(seed=0, height=96, width=128).requires_grad_(True)
    pixels, mask = differentiable_pixel_values(
        processor.image_processor, image, do_image_splitting=False
    )
    assert pixels.grad_fn is not None
    pixels.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0
    assert mask.shape[0] == pixels.shape[0]


def test_preprocess_is_deterministic(processor):
    image = reference_uint8_image(seed=1, height=64, width=80)
    first, _ = differentiable_pixel_values(processor.image_processor, image, do_image_splitting=False)
    second, _ = differentiable_pixel_values(processor.image_processor, image, do_image_splitting=False)
    assert torch.equal(first, second)


def test_uint8_matches_pil_and_float_gap_stays_moderate(processor):
    image = reference_uint8_image(seed=0, height=96, width=128)
    report = compare_preprocess(processor.image_processor, image, do_image_splitting=False)
    assert report["shapes_match"] is True
    assert report["float_mask_matches_pil"] is True
    assert report["max_abs_diff_uint8_tensor_vs_pil"] == 0.0
    # A second rescale would push the mean absolute gap near 1. The known
    # Lanczos float-versus-uint8 gap is much smaller than that.
    assert report["mean_abs_diff_float_vs_pil"] < 0.05
    assert report["max_abs_diff_float_vs_pil"] > 0.0


def test_different_input_sizes_share_the_no_split_layout(processor):
    small = reference_uint8_image(seed=2, height=32, width=32)
    wide = reference_uint8_image(seed=2, height=40, width=90)
    small_pixels, _ = differentiable_pixel_values(
        processor.image_processor, small, do_image_splitting=False
    )
    wide_pixels, _ = differentiable_pixel_values(
        processor.image_processor, wide, do_image_splitting=False
    )
    assert small_pixels.shape == wide_pixels.shape
    assert small_pixels.shape[-2:] == (512, 512)


def test_preprocess_refuses_a_non_cpu_image(processor):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available on this interpreter")
    image = reference_uint8_image(seed=3, height=16, width=16).to("mps")
    with pytest.raises(ValueError, match="CPU"):
        differentiable_pixel_values(processor.image_processor, image, do_image_splitting=False)
