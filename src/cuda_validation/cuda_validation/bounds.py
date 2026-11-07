"""Per-device attack bounds. Cross-device equality is not one of them."""

from __future__ import annotations

from typing import Any

from .pins import (
    BOUND_ATOL,
    EPSILON,
    EPSILON_LABEL,
    INIT,
    RESTARTS,
    STEP_SIZE,
    STEP_SIZE_LABEL,
    STEPS,
    STORED_HALF_LEVEL,
)


def check_bounds(record: dict[str, Any]) -> dict[str, Any]:
    """Check one device's own image against the frozen ball.

    The continuous image must stay inside epsilon. The official 8-bit image
    may sit at most half a level outside that ball. The other device's image
    is not an input.
    """
    violations: list[str] = []
    if record.get("epsilon") != EPSILON_LABEL:
        violations.append("epsilon label is not 1/255")
    if abs(float(record["epsilon_float"]) - EPSILON) > 1e-12:
        violations.append("epsilon float is not 1/255")
    if int(record["steps"]) != STEPS:
        violations.append("step count is not 3")
    if int(record["restarts"]) != RESTARTS:
        violations.append("restarts are not 0")
    if record.get("init") != INIT:
        violations.append("the attack did not start from the clean image")
    if record.get("step_size_label") != STEP_SIZE_LABEL:
        violations.append("step size label is not epsilon/4")
    if abs(float(record["step_size_float"]) - STEP_SIZE) > 1e-12:
        violations.append("step size float is not epsilon/4")
    linf = float(record["linf_float"])
    if linf < 0:
        violations.append("continuous L-infinity distance is negative")
    if linf > EPSILON + BOUND_ATOL:
        violations.append("continuous L-infinity distance exceeds epsilon")
    stored = float(record["linf_stored_vs_clean"])
    if stored < 0:
        violations.append("stored L-infinity distance is negative")
    if stored > EPSILON + STORED_HALF_LEVEL + BOUND_ATOL:
        violations.append("stored image exceeds epsilon by more than half a level")
    if record.get("pixels_in_unit_interval") is not True:
        violations.append("a pixel is outside [0, 1]")
    for value in record.get("per_step_linf_float", []):
        if float(value) > EPSILON + BOUND_ATOL:
            violations.append("a step leaves the continuous L-infinity ball")
            break
    return {
        "device_type": record.get("device_type"),
        "satisfied": not violations,
        "violations": violations,
        "continuous_radius": EPSILON_LABEL,
        "stored_allowance": "0.5/255",
        "cross_device_equality_required": False,
    }
