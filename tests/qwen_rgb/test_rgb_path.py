"""RGB path against the official Qwen image processor. No model weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from qwen_rgb.pins import REQUIRED_HF_HOME, PINNED_REVISION
from qwen_rgb.rgb_path import build_pinned_processor, differentiable_model_pixels, target_hw
from qwen_rgb.compare import compare_image
from vlm_iface.images import to_image_01

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "fixtures" / "development_v2" / "images" / "dev2-attribute-00-a.png"
SNAPSHOT = (
    Path(REQUIRED_HF_HOME)
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
    / PINNED_REVISION
    / "preprocessor_config.json"
)


@pytest.fixture(scope="module")
def image_processor():
    return build_pinned_processor()


def test_pinned_config_matches_the_local_snapshot():
    if not SNAPSHOT.is_file():
        pytest.skip("local Qwen snapshot is not on this machine")
    config = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert config["patch_size"] == 16
    assert config["temporal_patch_size"] == 2
    assert config["merge_size"] == 2
    assert config["image_mean"] == [0.5, 0.5, 0.5]
    assert config["image_std"] == [0.5, 0.5, 0.5]
    assert config["size"]["shortest_edge"] == 65536
    assert config["size"]["longest_edge"] == 16777216


def test_development_scene_matches_the_official_pil_processor(image_processor):
    image = to_image_01(SCENE)
    report = compare_image(image, image_processor)
    assert report["resize_is_identity"] is True
    assert report["input_hw"] == [384, 512]
    assert report["our_shape"] == [768, 1536]
    assert report["our_grid"] == [[1, 24, 32]]
    assert report["temporal_copies_equal"] is True
    assert report["versus_pil"]["shapes_match"] is True
    assert report["versus_pil"]["max_abs"] == 0.0
    assert report["versus_uint8_tensor"]["max_abs"] == 0.0
    assert report["versus_float_0_255"]["max_abs"] == 0.0
    assert report["grid_matches_pil"] is True
    assert report["within_official_match"] is True
    assert report["bf16_cast_of_official_pixels"]["max_abs"] > 0.0
    assert report["resize_rounding_probe"]["used_as_model_input"] is False
    assert report["resize_rounding_probe"]["max_abs_normalized_patches"] > 1e-3
    unit = report["versus_unit_interval_do_rescale_false"]["max_abs"]
    assert unit is not None and unit < 1e-5


def test_changed_size_matches_float_processor_and_not_uint8(image_processor):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    raw = torch.randint(0, 256, (3, 40, 70), generator=generator, dtype=torch.uint8)
    image_01 = raw.float() / 255.0
    ours, grid = differentiable_model_pixels(image_01, image_processor)
    float_pixels = image_processor(images=raw.float(), return_tensors="pt")["pixel_values"]
    pil = Image.fromarray(raw.permute(1, 2, 0).contiguous().numpy().copy(), mode="RGB")
    pil_pixels = image_processor(images=pil, return_tensors="pt")["pixel_values"]
    assert tuple(ours.shape) == tuple(float_pixels.shape) == tuple(pil_pixels.shape)
    assert torch.allclose(ours, float_pixels, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(ours, pil_pixels, atol=1e-3, rtol=1e-3)
    assert grid.shape == (1, 3)
    height, width = target_hw(40, 70, image_processor)
    assert (height, width) != (40, 70)
    assert int(grid[0, 1]) * 16 == height
    assert int(grid[0, 2]) * 16 == width


def test_gradient_reaches_the_original_rgb_on_the_development_image(image_processor):
    image = to_image_01(SCENE).requires_grad_(True)
    pixels, _ = differentiable_model_pixels(image, image_processor)
    assert pixels.grad_fn is not None
    pixels.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) == image.grad.numel()
    base = image.detach()
    step = 1e-3
    plus = base.clone()
    minus = base.clone()
    plus[1, 20, 30] += step
    minus[1, 20, 30] -= step
    with torch.no_grad():
        higher, _ = differentiable_model_pixels(plus, image_processor)
        lower, _ = differentiable_model_pixels(minus, image_processor)
    numerical = (higher - lower).sum() / (2 * step)
    assert torch.allclose(image.grad[1, 20, 30], numerical, atol=1e-3, rtol=1e-3)


def test_non_identity_resize_keeps_a_finite_nonzero_gradient(image_processor):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    image = torch.rand(3, 40, 70, generator=generator, dtype=torch.float32).requires_grad_(True)
    pixels, _ = differentiable_model_pixels(image, image_processor)
    assert pixels.grad_fn is not None
    pixels.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert torch.count_nonzero(image.grad) > 0
