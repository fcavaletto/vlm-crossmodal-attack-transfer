"""Three frozen PGD steps. The update function is the attack's own ``pgd_update``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .pins import EPSILON, INIT, RESTARTS, STEP_SIZE, STEPS


def run_three_steps(clean: Any, gradient_at: Callable[[Any], Any], update: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Start at ``clean``, ascend three times, and project with the frozen update.

    ``gradient_at`` is called on the image each step sees. Step zero sees the
    clean image. There is no second start.
    """
    if update is None:
        from attack_feasibility.steps import pgd_update

        update = pgd_update
    current = clean
    gradients = []
    images = []
    for _ in range(STEPS):
        gradient = gradient_at(current)
        gradients.append(gradient)
        current = update(current, clean, gradient, STEP_SIZE, EPSILON)
        images.append(current)
    return {
        "steps": STEPS,
        "restarts": RESTARTS,
        "init": INIT,
        "epsilon": EPSILON,
        "step_size": STEP_SIZE,
        "images": images,
        "gradients": gradients,
    }
