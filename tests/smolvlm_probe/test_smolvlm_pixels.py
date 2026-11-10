"""Model-free checks of the SmolVLM2 pixel paths. Skip when the pinned processor files are not cached."""

from pathlib import Path

import pytest
import torch

from smolvlm_probe.pixels import levels, official_pixels, split_size, surrogate_pixels, tile_grid, upscaled_size

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def processor():
    from smolvlm_probe.common import Stop, load_processor

    try:
        return load_processor()
    except (Stop, OSError) as exc:
        pytest.skip(f"pinned SmolVLM2 processor files are not in the local Hub cache: {type(exc).__name__}")


def test_geometry_for_study_images():
    assert upscaled_size(384, 512) == (1152, 1536)
    assert split_size(384, 512) == (1152, 1536)
    assert tile_grid(384, 512, "split") == (3, 4, 13)
    assert tile_grid(384, 512, "single") == (0, 0, 1)


@pytest.mark.parametrize("tiling", ["single", "split"])
def test_surrogate_is_differentiable_and_shaped(tiling):
    x = torch.rand(3, 384, 512, requires_grad=True)
    y = surrogate_pixels(x, tiling)
    assert y.shape == ((1 if tiling == "single" else 13), 3, 384, 384)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad != 0).any()


@pytest.mark.parametrize("tiling", ["single", "split"])
def test_surrogate_matches_processor_geometry_and_gap(processor, tiling):
    from attack_calibration.scene import _development_image
    from dev_pilot.fixtures import load_manifest

    for row in load_manifest(ROOT)[:3]:
        img = _development_image(ROOT, row)
        off = official_pixels(processor, img, tiling)
        sur = surrogate_pixels(img, tiling)
        assert off.shape == sur.shape
        # measured on all 96 development images: max 1.23 levels (uint8 rounding plus kernel differences)
        assert float(levels(sur - off).abs().max()) < 1.3


def test_colour_tokens(processor):
    from smolvlm_probe.backend import color_token_ids

    assert color_token_ids(processor) == {"red": 2382, "blue": 4461, "green": 2654, "yellow": 5724}
