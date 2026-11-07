"""Report whether CUDA is visible. Do not score an image in its place."""

from __future__ import annotations

import importlib.util
import shutil
from typing import Any


def probe() -> dict[str, Any]:
    """Read CUDA visibility when torch is installed. Never load Qwen."""
    report: dict[str, Any] = {
        "schema": "cuda_readiness.preflight.v1",
        "label": "DEVELOPMENT ONLY",
        "attacks_run": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "cuda_numerical_validation": "not_executed",
        "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None,
        "docker_on_path": shutil.which("docker") is not None,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "cuda_available": None,
        "cuda_device_count": None,
        "device_name": None,
        "torch_version": None,
    }
    if report["torch_installed"] is not True:
        report["note"] = (
            "torch is not installed in this interpreter. CUDA availability was not read. "
            "This is not a CUDA measurement, and a CPU score was not run in its place."
        )
        return report
    import torch

    available = bool(torch.cuda.is_available())
    report["torch_version"] = torch.__version__
    report["cuda_available"] = available
    report["cuda_device_count"] = int(torch.cuda.device_count()) if available else 0
    if available:
        report["device_name"] = torch.cuda.get_device_name(0)
        report["note"] = (
            "CUDA is visible. No weight was loaded and no image was scored. "
            "Visibility is not numerical validation of the Study 2 scoring path."
        )
    else:
        report["note"] = (
            "torch.cuda.is_available() is false. No CPU score was run in its place. "
            "This is not a CUDA measurement."
        )
    return report


def require_cuda(report: dict[str, Any]) -> None:
    """Exit condition for the on-GPU command. A missing device is not relabeled."""
    if report.get("cuda_available") is not True:
        raise SystemExit(2)
