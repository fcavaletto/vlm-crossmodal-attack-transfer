"""Future RGB constraint. No signed step and no iteration."""

from __future__ import annotations

import torch

from qwen_rgb.constraints import (
    clip_unit_interval,
    linf_distance,
    project_linf,
    round_trip_uint8,
    store_feasible,
)


def test_projection_respects_the_ball_and_the_unit_interval():
    clean = torch.tensor(
        [
            [[0.2, 0.8], [0.0, 1.0]],
            [[0.4, 0.6], [0.5, 0.5]],
            [[0.1, 0.9], [0.3, 0.7]],
        ]
    )
    proposed = clean + 0.2
    projected = project_linf(proposed, clean, epsilon=0.05)
    assert float(linf_distance(projected, clean)) <= 0.05 + 1e-6
    assert float(projected.min()) >= 0.0
    assert float(projected.max()) <= 1.0
    assert torch.equal(clip_unit_interval(projected), projected)


def test_eight_bit_round_trip_is_identity_on_stored_levels_and_bounded():
    levels = torch.arange(256, dtype=torch.float32) / 255.0
    image = levels.view(1, 16, 16).repeat(3, 1, 1)
    assert torch.equal(round_trip_uint8(image), image)
    fractional = image + (0.4 / 255.0)
    stored = round_trip_uint8(fractional)
    gap = (stored - fractional.clamp(0, 1)).abs().max()
    assert float(gap) <= 0.5 / 255.0 + 1e-6
    assert round_trip_uint8(image).requires_grad is False


def test_stored_feasible_image_stays_inside_the_cube():
    clean = torch.full((3, 8, 8), 0.5)
    proposed = torch.rand(3, 8, 8)
    stored = store_feasible(proposed, clean, epsilon=1 / 255)
    assert float(stored.min()) >= 0.0
    assert float(stored.max()) <= 1.0
    # Half a quantization level is the extra room the storage map can add.
    assert float(linf_distance(stored, clean)) <= (1 / 255) + (0.5 / 255) + 1e-6
