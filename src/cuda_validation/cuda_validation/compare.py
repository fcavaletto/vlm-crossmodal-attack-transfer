"""Record MPS and CUDA differences. Numeric equality is not interchangeability."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any

from .bounds import check_bounds
from .hardware import missing_hardware_fields, precision_record_problems
from .pins import (
    ATTN_IMPLEMENTATION,
    DTYPE,
    EPSILON_LABEL,
    MUST_CHECK,
    REVISION,
    SCENE_ID,
    STEPS,
)

COLORS = ("red", "blue", "green", "yellow")


class IdenticalDevicesError(ValueError):
    """Raised when a caller asks to treat the two measurements as the same."""


def compare_records(
    mps: dict[str, Any],
    cuda: dict[str, Any],
    *,
    treat_as_identical: bool = False,
    mps_dir: Path | None = None,
    cuda_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare one MPS sidecar with one CUDA sidecar.

    ``treat_as_identical`` cannot be true. The report keeps
    ``devices_interchangeable`` false when every stored scalar matches.
    """
    if treat_as_identical:
        raise IdenticalDevicesError(
            "MPS and CUDA measurements are not interchangeable. Refusing to mark them identical."
        )
    if mps.get("device_type") != "mps" or cuda.get("device_type") != "cuda":
        raise ValueError("the left record must be MPS and the right record must be CUDA.")
    for record in (mps, cuda):
        if record.get("devices_identical") is True or record.get("devices_interchangeable") is True:
            raise IdenticalDevicesError("a sidecar claims the devices are interchangeable.")

    report: dict[str, Any] = {
        "schema": "cuda_validation.compare.v1",
        "label": "DEVELOPMENT ONLY",
        "devices_interchangeable": False,
        "mps_and_cuda_treated_as_identical": False,
        "numeric_equality_is_not_interchangeability": True,
        "cuda_measurement_is_an_mps_measurement": False,
        "must_check": list(MUST_CHECK),
        "memory_quantities_are_distinct": True,
        "cross_device_equality_required": False,
        "full_development_attack_run": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "guard_lowered": False,
    }
    blocked = _protocol_blocks(mps, cuda)
    mps_hardware = mps.get("hardware") or {}
    cuda_hardware = cuda.get("hardware") or {}
    hardware_missing = {
        "mps": missing_hardware_fields(mps_hardware),
        "cuda": missing_hardware_fields(cuda_hardware),
    }
    precision_problems = {
        "mps": precision_record_problems(mps_hardware),
        "cuda": precision_record_problems(cuda_hardware),
    }
    mps_bounds = check_bounds(_bound_view(mps))
    cuda_bounds = check_bounds(_bound_view(cuda))
    outputs = _output_differences(mps, cuda)
    losses = _loss_differences(mps, cuda)
    gradients = _gradient_differences(mps, cuda, mps_dir=mps_dir, cuda_dir=cuda_dir)
    report["protocol_blocks"] = blocked
    report["hardware_missing"] = hardware_missing
    report["precision_problems"] = precision_problems
    report["bounds"] = {"mps": mps_bounds, "cuda": cuda_bounds}
    report["outputs"] = outputs
    report["losses"] = losses
    report["gradients"] = gradients
    report["hardware_flags"] = {
        "mps_allow_tf32_matmul": mps_hardware.get("allow_tf32_matmul"),
        "cuda_allow_tf32_matmul": cuda_hardware.get("allow_tf32_matmul"),
        "mps_allow_tf32_cudnn": mps_hardware.get("allow_tf32_cudnn"),
        "cuda_allow_tf32_cudnn": cuda_hardware.get("allow_tf32_cudnn"),
        "cuda_driver_version": cuda_hardware.get("driver_version"),
        "cuda_compute_capability": cuda_hardware.get("compute_capability"),
        "mps_torch_version": mps_hardware.get("torch_version"),
        "cuda_torch_version": cuda_hardware.get("torch_version"),
    }
    report["memory"] = {
        "mps_os_available_bytes": (mps.get("memory") or {}).get("os_available_bytes"),
        "cuda_os_available_bytes": (cuda.get("memory") or {}).get("os_available_bytes"),
        "mps_accelerator": mps_hardware.get("accelerator_memory"),
        "cuda_accelerator": cuda_hardware.get("accelerator_memory"),
        "note": "OS-available bytes, MPS unified memory, and CUDA VRAM are three readings.",
    }
    report["numeric_values_equal"] = _numerics_equal(outputs, losses, gradients)
    report["status"] = _status(
        blocked,
        hardware_missing,
        precision_problems,
        mps_bounds,
        cuda_bounds,
        mps,
        cuda,
        gradients,
    )
    return report


def _protocol_blocks(mps: dict[str, Any], cuda: dict[str, Any]) -> list[str]:
    blocks = []
    pairs = (
        ("scene_id", SCENE_ID),
        ("revision", REVISION),
        ("dtype", DTYPE),
        ("attn_implementation", ATTN_IMPLEMENTATION),
        ("epsilon", EPSILON_LABEL),
        ("steps", STEPS),
    )
    for key, expected in pairs:
        if mps.get(key) != expected or cuda.get(key) != expected or mps.get(key) != cuda.get(key):
            blocks.append(key)
    if mps.get("scene_count") != 1 or cuda.get("scene_count") != 1:
        blocks.append("scene_count")
    left_trajectory = list((mps.get("losses") or {}).get("trajectory") or [])
    right_trajectory = list((cuda.get("losses") or {}).get("trajectory") or [])
    if len(left_trajectory) != STEPS + 1 or len(right_trajectory) != STEPS + 1:
        blocks.append("trajectory_length")
    if mps.get("loss_identity_agreed") is False or cuda.get("loss_identity_agreed") is False:
        blocks.append("loss_identity")
    return blocks


def _bound_view(record: dict[str, Any]) -> dict[str, Any]:
    bounds = dict(record.get("bounds") or {})
    bounds["device_type"] = record.get("device_type")
    for key in ("epsilon", "epsilon_float", "steps", "restarts", "init", "step_size_label", "step_size_float"):
        if key not in bounds and key in record:
            bounds[key] = record[key]
    return bounds


def _output_differences(mps: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    sections = {}
    for name in ("clean_official", "clean_differentiable", "final_official"):
        left = ((mps.get("outputs") or {}).get(name)) or {}
        right = ((cuda.get("outputs") or {}).get(name)) or {}
        summed = {}
        for color in COLORS:
            left_value = (left.get("summed_nll") or {}).get(color)
            right_value = (right.get("summed_nll") or {}).get(color)
            summed[color] = {
                "mps": left_value,
                "cuda": right_value,
                "abs_diff": _abs_diff(left_value, right_value),
            }
        sections[name] = {
            "summed_nll": summed,
            "mps_chosen": left.get("chosen"),
            "cuda_chosen": right.get("chosen"),
            "chosen_equal": left.get("chosen") == right.get("chosen"),
            "mps_loss": left.get("loss"),
            "cuda_loss": right.get("loss"),
            "loss_abs_diff": _abs_diff(left.get("loss"), right.get("loss")),
        }
    return sections


def _loss_differences(mps: dict[str, Any], cuda: dict[str, Any]) -> dict[str, Any]:
    left = list((mps.get("losses") or {}).get("trajectory") or [])
    right = list((cuda.get("losses") or {}).get("trajectory") or [])
    width = max(len(left), len(right))
    steps = []
    for index in range(width):
        left_value = left[index] if index < len(left) else None
        right_value = right[index] if index < len(right) else None
        steps.append(
            {
                "index": index,
                "mps": left_value,
                "cuda": right_value,
                "abs_diff": _abs_diff(left_value, right_value),
            }
        )
    return {
        "trajectory_length_mps": len(left),
        "trajectory_length_cuda": len(right),
        "expected_length": STEPS + 1,
        "steps": steps,
    }


def _gradient_differences(
    mps: dict[str, Any],
    cuda: dict[str, Any],
    *,
    mps_dir: Path | None,
    cuda_dir: Path | None,
) -> dict[str, Any]:
    left = mps.get("gradient") or {}
    right = cuda.get("gradient") or {}
    left_values = _values(left, mps_dir)
    right_values = _values(right, cuda_dir)
    report: dict[str, Any] = {
        "mps_sha256": left.get("sha256"),
        "cuda_sha256": right.get("sha256"),
        "sha256_equal": left.get("sha256") == right.get("sha256") and left.get("sha256") is not None,
        "cosine": None,
        "relative_l2": None,
        "sign_agreement": None,
        "max_abs": None,
        "identity_threshold_applied": False,
        "file_problems": [],
    }
    if left_values is None or right_values is None:
        report["reason"] = "gradient values were not both available, so cosine was not computed."
        return report
    if len(left_values) != len(right_values):
        report["file_problems"].append("gradient lengths differ")
        return report
    report.update(_vector_facts(left_values, right_values))
    return report


def _values(gradient: dict[str, Any], base: Path | None) -> list[float] | None:
    if gradient.get("values") is not None:
        return _flatten(gradient["values"])
    name = gradient.get("f32_file")
    if not name or base is None:
        return None
    path = (base / str(name)).resolve()
    if any(part.lower() in {"heldout", "held-out", "held_out", "sealed"} for part in path.parts):
        raise ValueError("refusing to read a gradient from a held-out path.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if gradient.get("sha256") and digest != gradient["sha256"]:
        raise ValueError("gradient file sha256 does not match the sidecar.")
    count = len(raw) // 4
    if count * 4 != len(raw):
        raise ValueError("gradient file is not a whole number of float32 values.")
    return list(struct.unpack(f"<{count}f", raw))


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list) and values and isinstance(values[0], list):
        flat: list[float] = []
        for item in values:
            flat.extend(_flatten(item))
        return flat
    return [float(item) for item in values]


def _vector_facts(left: list[float], right: list[float]) -> dict[str, Any]:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    gap_sq = 0.0
    max_abs = 0.0
    agreed = 0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_sq += a * a
        right_sq += b * b
        gap = a - b
        gap_sq += gap * gap
        max_abs = max(max_abs, abs(gap))
        if _sign(a) == _sign(b):
            agreed += 1
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    cosine = None
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = dot / (left_norm * right_norm)
    denominator = max(left_norm, right_norm, 1e-12)
    return {
        "cosine": cosine,
        "relative_l2": math.sqrt(gap_sq) / denominator,
        "sign_agreement": agreed / len(left),
        "max_abs": max_abs,
        "identity_threshold_applied": False,
        "reason": "cosine and relative L2 are recorded. They are not a pass line.",
    }


def _numerics_equal(outputs: dict[str, Any], losses: dict[str, Any], gradients: dict[str, Any]) -> bool:
    for section in outputs.values():
        if section["chosen_equal"] is not True or section["loss_abs_diff"] != 0.0:
            return False
        for color in section["summed_nll"].values():
            if color["abs_diff"] != 0.0:
                return False
    if losses["trajectory_length_mps"] != losses["trajectory_length_cuda"]:
        return False
    for step in losses["steps"]:
        if step["abs_diff"] != 0.0:
            return False
    values_equal = gradients.get("max_abs") == 0.0 and gradients.get("sign_agreement") == 1.0
    return values_equal or gradients.get("sha256_equal") is True


def _status(
    blocked: list[str],
    hardware_missing: dict[str, list[str]],
    precision_problems: dict[str, list[str]],
    mps_bounds: dict[str, Any],
    cuda_bounds: dict[str, Any],
    mps: dict[str, Any],
    cuda: dict[str, Any],
    gradients: dict[str, Any],
) -> str:
    if blocked:
        return "blocked_protocol_mismatch"
    if precision_problems["mps"] or precision_problems["cuda"]:
        return "blocked_precision_record"
    if mps_bounds["satisfied"] is not True or cuda_bounds["satisfied"] is not True:
        return "bound_violation"
    if hardware_missing["mps"] or hardware_missing["cuda"]:
        return "incomplete_hardware_record"
    if gradients.get("file_problems"):
        return "incomplete_gradient_file"
    if gradients.get("cosine") is None:
        return "incomplete_gradient_record"
    if mps.get("status") not in (None, "completed") or cuda.get("status") not in (None, "completed"):
        return "incomplete_hardware_record"
    return "differences_recorded"


def _abs_diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
