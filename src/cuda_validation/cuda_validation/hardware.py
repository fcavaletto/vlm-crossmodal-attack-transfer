"""Hardware fields the comparison has to carry. Flags are read, not flipped."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .pins import NVIDIA_DRIVER_MINIMUM

NULL_REASON = {
    "compute_capability": "compute_capability_null_reason",
    "cuda_runtime_version": "cuda_runtime_null_reason",
    "allow_tf32_matmul": "allow_tf32_matmul_null_reason",
    "allow_tf32_cudnn": "allow_tf32_cudnn_null_reason",
    "cudnn_benchmark": "cudnn_benchmark_null_reason",
    "cudnn_deterministic": "cudnn_deterministic_null_reason",
    "bf16_reduced_precision_reduction": "bf16_reduced_precision_reduction_null_reason",
    "driver_version": "driver_version_null_reason",
}

REQUIRED_FIELDS = (
    "device_type",
    "device_name",
    "compute_capability",
    "driver_version",
    "torch_version",
    "cuda_runtime_version",
    "parameter_dtype",
    "rgb_leaf_dtype",
    "attn_implementation",
    "allow_tf32_matmul",
    "allow_tf32_cudnn",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "float32_matmul_precision",
    "bf16_reduced_precision_reduction",
    "os_available_bytes",
    "accelerator_memory",
)


def missing_hardware_fields(hardware: dict[str, Any]) -> list[str]:
    """Names that are absent, or null without a reason that says why."""
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in hardware:
            missing.append(field)
            continue
        if hardware[field] is None:
            reason_key = NULL_REASON.get(field)
            reason = None if reason_key is None else hardware.get(reason_key)
            if not isinstance(reason, str) or not reason.strip():
                missing.append(field)
    return missing


def precision_record_problems(hardware: dict[str, Any]) -> list[str]:
    """Reject a record that pretends MPS has a CUDA switch."""
    problems = []
    device = hardware.get("device_type")
    if device == "mps":
        for field in ("allow_tf32_matmul", "allow_tf32_cudnn", "cudnn_benchmark", "cudnn_deterministic"):
            if hardware.get(field) is not None:
                problems.append(f"MPS {field} is not a boolean CUDA flag.")
        if hardware.get("cuda_runtime_version") is not None:
            problems.append("MPS cuda_runtime_version is not a CUDA runtime.")
        if hardware.get("compute_capability") is not None:
            problems.append("MPS compute_capability is not a CUDA capability.")
        memory = hardware.get("accelerator_memory") or {}
        if memory.get("kind") != "mps_unified":
            problems.append("MPS accelerator memory is the unified pool, not CUDA VRAM.")
    elif device == "cuda":
        if not isinstance(hardware.get("driver_version"), str) or not hardware.get("driver_version"):
            problems.append("the CUDA driver version was not read.")
        if not isinstance(hardware.get("compute_capability"), str):
            problems.append("the CUDA compute capability was not read.")
        if not isinstance(hardware.get("cuda_runtime_version"), str):
            problems.append("the CUDA runtime version was not read.")
        for field in ("allow_tf32_matmul", "allow_tf32_cudnn"):
            if not isinstance(hardware.get(field), bool):
                problems.append(f"CUDA {field} was not read as a boolean.")
        memory = hardware.get("accelerator_memory") or {}
        if memory.get("kind") != "cuda_vram":
            problems.append("CUDA accelerator memory is VRAM, not the MPS pool.")
    return problems


def driver_at_least_minimum(version: str, minimum: int = NVIDIA_DRIVER_MINIMUM) -> bool:
    """Compare the leading driver component with the recorded cu132 minimum."""
    head = version.strip().split(".", 1)[0]
    return head.isdigit() and int(head) >= minimum


def read_nvidia_driver() -> str | None:
    """Read ``nvidia-smi`` when it is on PATH. Absence is not driver 0."""
    if shutil.which("nvidia-smi") is None:
        return None
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None


def is_quoted_l4(name: str) -> bool:
    """True for an L4 name. L40 and L40S are different cards."""
    tokens = name.replace("-", " ").replace("_", " ").split()
    return "L4" in tokens


def collect_torch_hardware(device, *, driver_version: str | None, parameter_dtype: str, attn: str) -> dict[str, Any]:
    """Read device flags. This does not change them and does not load Qwen."""
    import torch

    if device.type not in {"mps", "cuda"}:
        raise RuntimeError("CPU hardware is not an MPS record and not a CUDA record.")
    hardware: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "parameter_dtype": parameter_dtype,
        "rgb_leaf_dtype": "torch.float32",
        "attn_implementation": attn,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "os_available_bytes": None,
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        properties = torch.cuda.get_device_properties(device)
        reduction = _bf16_reduction_flag(torch)
        hardware.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "compute_capability": f"{major}.{minor}",
                "driver_version": driver_version,
                "cuda_runtime_version": torch.version.cuda,
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "bf16_reduced_precision_reduction": reduction,
                "accelerator_memory": {
                    "kind": "cuda_vram",
                    "total_bytes": int(properties.total_memory),
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                },
            }
        )
        if reduction is None:
            hardware["bf16_reduced_precision_reduction_null_reason"] = (
                "This torch build does not expose allow_bf16_reduced_precision_reduction. "
                "Null is not a claim that the reduction is off."
            )
        return hardware
    hardware.update(
        {
            "device_name": "mps",
            "compute_capability": None,
            "compute_capability_null_reason": "MPS does not report a CUDA compute capability.",
            "driver_version": driver_version,
            "cuda_runtime_version": None,
            "cuda_runtime_null_reason": "MPS has no CUDA runtime. Null is not CUDA 0.",
            "allow_tf32_matmul": None,
            "allow_tf32_matmul_null_reason": "MPS has no TF32 switch. Null is not false.",
            "allow_tf32_cudnn": None,
            "allow_tf32_cudnn_null_reason": "MPS has no TF32 switch. Null is not false.",
            "cudnn_benchmark": None,
            "cudnn_benchmark_null_reason": "MPS has no cuDNN benchmark flag.",
            "cudnn_deterministic": None,
            "cudnn_deterministic_null_reason": "MPS has no cuDNN deterministic flag.",
            "bf16_reduced_precision_reduction": None,
            "bf16_reduced_precision_reduction_null_reason": "No CUDA reduction flag was read on MPS.",
            "accelerator_memory": {
                "kind": "mps_unified",
                "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
                "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            },
        }
    )
    return hardware


def _bf16_reduction_flag(torch) -> bool | None:
    matmul = getattr(torch.backends.cuda, "matmul", None)
    flag = getattr(matmul, "allow_bf16_reduced_precision_reduction", None)
    if flag is None:
        return None
    return bool(flag)
