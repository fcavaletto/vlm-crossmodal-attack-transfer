"""The validation plan. Building it does not load Qwen or start a pod."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cuda_readiness.preflight import probe

from .pins import (
    ATTN_IMPLEMENTATION,
    BF16_REQUIRED_AVAILABLE_BYTES,
    COMPUTE_CAPABILITY,
    DTYPE,
    EPSILON,
    EPSILON_LABEL,
    EVAL_CONTEXTS,
    FLOAT32_REQUIRED_AVAILABLE_BYTES,
    GPU_CLOUD,
    GPU_HOST_RAM,
    GPU_HOURLY_RATE_USD,
    GPU_NAME,
    GPU_PAGE_UPDATED,
    GPU_PRICE_CORROBORATION,
    GPU_PRICE_READ_ON,
    GPU_PRICE_SOURCE,
    GPU_VCPUS,
    GPU_VRAM,
    INIT,
    MIN_CUDA_VERSION,
    MUST_CHECK,
    NVIDIA_DRIVER_MINIMUM,
    OS_FLOOR_BYTES,
    RESTARTS,
    REVISION,
    SCENE_COUNT,
    SCENE_ID,
    SCENE_IMAGE_RELATIVE,
    SOURCE_CONTEXT,
    STEP_SIZE,
    STEP_SIZE_LABEL,
    STEPS,
)

SCIENCE_MANIFEST = Path("docs/checkpoint-2.0/cuda/manifest.json")
BUDGET_SCRIPT = Path("docs/checkpoint-1.5/run_model_feasibility.py")
POD_SPEC = Path("docker/runpod/pod-spec.json")


def build_protocol(root: Path) -> dict[str, Any]:
    """Describe the one-scene comparison and refuse a changed attack file."""
    attack = _attack_files(root)
    if attack["unchanged"] is not True:
        raise RuntimeError("A frozen attack file hash changed. The validation will not adopt the new file.")
    lines = _budget_lines(root)
    if lines["bf16_calibrated_projection"] != BF16_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError("The bfloat16 line is not 11,509,447,200. The guard was not replaced.")
    if lines["float32_calibrated_projection"] != FLOAT32_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError("The float32 line is not 20,871,410,752. The guard was not replaced.")
    if lines["os_floor"] != OS_FLOOR_BYTES:
        raise RuntimeError("The 6 GiB floor is not 6,442,450,944. The guard was not replaced.")
    pod = json.loads((root / POD_SPEC).read_text(encoding="utf-8"))
    if pod.get("launched") is not False or pod.get("billable_gpu_launched") is not False:
        raise RuntimeError("The pod spec says a GPU was launched. This protocol does not start one.")
    visibility = probe()
    return {
        "schema": "cuda_validation.protocol.v1",
        "label": "DEVELOPMENT ONLY",
        "executed_on_cuda": False,
        "executed_on_mps": False,
        "model_from_pretrained_called": False,
        "attacks_run": False,
        "full_development_attack_run": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "guard_lowered": False,
        "billable_gpu_launched": False,
        "runpod_api_called": False,
        "devices_interchangeable": False,
        "cuda_numerical_validation": "not_executed",
        "attack_files_unchanged": True,
        "scene": {
            "scene_id": SCENE_ID,
            "scene_count": SCENE_COUNT,
            "split": "development",
            "image_relative": SCENE_IMAGE_RELATIVE,
            "image_opened_by_this_command": False,
        },
        "attack": {
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "revision": REVISION,
            "dtype": DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "source_context": SOURCE_CONTEXT,
            "eval_contexts": list(EVAL_CONTEXTS),
            "epsilon": EPSILON_LABEL,
            "epsilon_float": EPSILON,
            "steps": STEPS,
            "step_size": STEP_SIZE_LABEL,
            "step_size_float": STEP_SIZE,
            "init": INIT,
            "restarts": RESTARTS,
            "definition_changed": False,
        },
        "checks": {
            "outputs": "summed NLL of red, blue, green, and yellow, clean official, clean differentiable, and final official",
            "losses": "labeled-color trajectory at the clean image and after each of the three steps",
            "gradients": "clean-image float32 RGB gradient: sha256, cosine, relative L2, and sign agreement",
            "bounds": "each device's own L-infinity ball, unit interval, and 8-bit round trip",
        },
        "must_check": list(MUST_CHECK),
        "gpu": {
            "name": GPU_NAME,
            "provider": "RunPod",
            "cloud": GPU_CLOUD,
            "vram": GPU_VRAM,
            "host_ram_listing": GPU_HOST_RAM,
            "vcpus_listing": GPU_VCPUS,
            "hourly_rate_usd": GPU_HOURLY_RATE_USD,
            "page_updated": GPU_PAGE_UPDATED,
            "price_read_on": GPU_PRICE_READ_ON,
            "source": GPU_PRICE_SOURCE,
            "corroboration": GPU_PRICE_CORROBORATION,
            "min_cuda_version": MIN_CUDA_VERSION,
            "nvidia_driver_minimum": NVIDIA_DRIVER_MINIMUM,
            "compute_capability": COMPUTE_CAPABILITY,
            "launched": False,
        },
        "memory_lines_bytes": lines,
        "preflight": visibility,
        "pod_spec": str(POD_SPEC),
    }


def repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "checkpoint-1.9" / "ATTACK_PROTOCOL.md").is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 1.9 protocol is not above {start}")


def _attack_files(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / SCIENCE_MANIFEST).read_text(encoding="utf-8"))
    expected = manifest["science_sha256"]
    actual = {relative: _sha256(root / relative) for relative in expected}
    return {"unchanged": actual == expected, "sha256": actual}


def _budget_lines(root: Path) -> dict[str, int]:
    module = _load_budget(root / BUDGET_SCRIPT)
    if 6 * 1024**3 != OS_FLOOR_BYTES:
        raise RuntimeError("The 6 GiB floor constant moved.")
    bf16 = module.required_available_bytes(
        module.QWEN_C0_BF16_BYTES,
        module.C1_DROP_PER_CURRENT,
        min_available_bytes=OS_FLOOR_BYTES,
    )
    float32 = module.required_available_bytes(
        module.QWEN_C0_FLOAT32_BYTES,
        module.C1_DROP_PER_CURRENT,
        min_available_bytes=OS_FLOOR_BYTES,
    )
    return {
        "bf16_calibrated_projection": int(bf16["calibrated_projection"]),
        "float32_calibrated_projection": int(float32["calibrated_projection"]),
        "os_floor": int(bf16["os_floor"]),
        "guard_lowered": False,
    }


def _load_budget(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("checkpoint15_budget_lines_cuda_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
