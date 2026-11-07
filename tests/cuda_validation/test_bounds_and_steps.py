"""Bounds are per device. The three steps call the frozen update sizes."""

from __future__ import annotations

from cuda_validation.bounds import check_bounds
from cuda_validation.hardware import driver_at_least_minimum, is_quoted_l4
from cuda_validation.pins import BOUND_ATOL, EPSILON, STEPS, STORED_HALF_LEVEL
from cuda_validation.trajectory import run_three_steps


def _bounds(**overrides):
    record = {
        "device_type": "cuda",
        "epsilon": "1/255",
        "epsilon_float": EPSILON,
        "steps": 3,
        "restarts": 0,
        "init": "clean_image",
        "step_size_label": "epsilon/4",
        "step_size_float": EPSILON / 4.0,
        "linf_float": EPSILON,
        "linf_stored_vs_clean": EPSILON + STORED_HALF_LEVEL,
        "pixels_in_unit_interval": True,
        "per_step_linf_float": [EPSILON / 4.0, EPSILON / 2.0, 3.0 * EPSILON / 4.0],
    }
    record.update(overrides)
    return record


def test_each_device_has_its_own_ball():
    inside = check_bounds(_bounds())
    assert inside["satisfied"] is True
    assert inside["cross_device_equality_required"] is False
    over = check_bounds(_bounds(linf_float=EPSILON + BOUND_ATOL + 1e-6))
    assert over["satisfied"] is False
    stored = check_bounds(_bounds(linf_stored_vs_clean=EPSILON + STORED_HALF_LEVEL + 1e-4))
    assert stored["satisfied"] is False
    pixels = check_bounds(_bounds(pixels_in_unit_interval=False))
    assert pixels["satisfied"] is False


def test_three_steps_use_the_frozen_step_size_and_do_not_restart():
    calls = []

    def update(current, clean, gradient, step_size, epsilon):
        calls.append((current, clean, gradient, step_size, epsilon))
        return ("image", len(calls))

    seen = []

    def gradient_at(image):
        seen.append(image)
        return ("gradient", image)

    result = run_three_steps("clean", gradient_at, update=update)
    assert result["steps"] == STEPS
    assert result["restarts"] == 0
    assert result["init"] == "clean_image"
    assert seen == ["clean", ("image", 1), ("image", 2)]
    assert len(calls) == 3
    assert all(call[3] == EPSILON / 4.0 and call[4] == EPSILON for call in calls)
    assert all(call[1] == "clean" for call in calls)


def test_l4_name_and_driver_floor():
    assert is_quoted_l4("NVIDIA L4") is True
    assert is_quoted_l4("NVIDIA L40") is False
    assert is_quoted_l4("NVIDIA L40S") is False
    assert driver_at_least_minimum("580.65.06") is True
    assert driver_at_least_minimum("579.99") is False


def test_frozen_pgd_update_stays_inside_the_ball():
    import pytest

    torch = pytest.importorskip("torch")
    from attack_feasibility.steps import pgd_update
    from qwen_rgb.constraints import linf_distance

    clean = torch.full((3, 2, 2), 0.5)
    gradient = torch.ones(3, 2, 2)
    result = run_three_steps(clean, lambda _image: gradient)
    assert result["steps"] == 3
    assert result["restarts"] == 0
    distance = float(linf_distance(result["images"][-1], clean))
    assert distance <= EPSILON + 1e-6
    del pgd_update
