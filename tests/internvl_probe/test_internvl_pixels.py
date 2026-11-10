"""Model-free checks of the InternVL pixel paths. Skip when the pinned processor files are not cached."""

from pathlib import Path

import pytest
import torch

from internvl_probe.pixels import levels, official_pixels, surrogate_pixels, tile_grid

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def processor():
    from internvl_probe.common import Stop, load_processor

    try:
        return load_processor()
    except (Stop, OSError) as exc:
        pytest.skip(f"pinned InternVL processor files are not in the local Hub cache: {type(exc).__name__}")


def test_tile_grid_for_study_images():
    assert tile_grid(384, 512, "dynamic") == (4, 3, 13)
    assert tile_grid(384, 512, "single") == (1, 1, 1)


def test_surrogate_is_differentiable_and_shaped():
    x = torch.rand(3, 384, 512, requires_grad=True)
    y = surrogate_pixels(x, "single")
    assert y.shape == (1, 3, 448, 448)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad != 0).any()


@pytest.mark.parametrize("tiling", ["single", "dynamic"])
def test_surrogate_matches_processor_geometry_and_within_one_level(processor, tiling):
    from attack_calibration.scene import _development_image
    from dev_pilot.fixtures import load_manifest

    for row in load_manifest(ROOT)[:3]:
        img = _development_image(ROOT, row)
        off = official_pixels(processor, img, tiling)
        sur = surrogate_pixels(img, tiling)
        assert off.shape == sur.shape
        # measured on all 96 development images: max 1.06 levels (uint8 rounding plus kernel differences)
        assert float(levels(sur - off).abs().max()) < 1.1


def test_colour_tokens(processor):
    from internvl_probe.backend import color_token_ids

    assert color_token_ids(processor) == {"red": 1151, "blue": 12203, "green": 13250, "yellow": 27869}
