"""Numerical controls: no TF32, no autocast, deterministic kernels, and audits that prove it."""

from __future__ import annotations

import os
from typing import Any

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def configure(device: torch.device) -> dict[str, Any]:
    """Call before any model is loaded. Returns the settings actually in force.

    On MPS nothing is changed: the local reproduction check must run the frozen 2.4 numerics,
    which never set these flags. On CUDA, TF32 and reduced-precision reductions are disabled
    and deterministic kernels are required, identically for both precisions.
    """
    if device.type != "cuda":
        return settings(device)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
    return settings(device)


def settings(device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"device": str(device), "torch": torch.__version__,
                           "float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "autocast_enabled": autocast_enabled(device)}
    if device.type == "cuda":
        out.update({
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "gpu": torch.cuda.get_device_name(device), "cuda_runtime": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(device)),
        })
    return out


def autocast_enabled(device: torch.device) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device.type))
    except TypeError:  # older signature
        return bool(torch.is_autocast_enabled())


def assert_no_autocast(device: torch.device) -> None:
    if autocast_enabled(device):
        raise RuntimeError("autocast is enabled; this study forbids it")


def fp32_matmul_probe(device: torch.device, n: int = 1024, seed: int = 0) -> dict[str, float | bool]:
    """fp32 matmul on ``device`` against an fp64 CPU reference. TF32 shows up as ~1e-3 relative error."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    b = torch.randn(n, n, generator=g, dtype=torch.float64)
    ref = a @ b
    got = (a.float().to(device) @ b.float().to(device)).cpu().double()  # MPS has no float64
    rel = float(((got - ref).abs().max() / ref.abs().max()))
    return {"max_rel_err_vs_fp64": rel, "passes_1e-5": rel < 1e-5}


class DtypeAudit:
    """Forward hooks on every module: the set of floating dtypes each module type outputs."""

    def __init__(self, model: torch.nn.Module):
        self.seen: dict[str, set[str]] = {}
        self.handles = [m.register_forward_hook(self._hook(name)) for name, m in model.named_modules() if name]

    def _hook(self, name):
        def fn(module, inputs, output):
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for o in outs:
                if torch.is_tensor(o) and o.is_floating_point():
                    self.seen.setdefault(type(module).__name__, set()).add(str(o.dtype).replace("torch.", ""))
        return fn

    def remove(self) -> dict[str, list[str]]:
        for h in self.handles:
            h.remove()
        return {k: sorted(v) for k, v in sorted(self.seen.items())}


def non_matching(audit: dict[str, list[str]], dtype: str) -> dict[str, list[str]]:
    want = "float32" if dtype == "fp32" else "bfloat16"
    return {k: v for k, v in audit.items() if v != [want]}
