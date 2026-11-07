"""Refuse every run that is not the one development scene on MPS or CUDA.

Approval does not load a model and does not open an image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pins import (
    BANNED_PATH_PARTS,
    BF16_REQUIRED_AVAILABLE_BYTES,
    DTYPE,
    FLOAT32_REQUIRED_AVAILABLE_BYTES,
    FORBIDDEN_OUTPUT_PREFIXES,
    OS_FLOOR_BYTES,
    SCENE_COUNT,
    SCENE_ID,
    SCENE_IMAGE_RELATIVE,
)


def approve(request: dict[str, Any]) -> dict[str, Any]:
    """Return whether ``measure`` may continue. A refusal is not a CPU result."""
    problems: list[str] = []
    device = str(request.get("device", "")).lower()
    if device not in {"mps", "cuda"}:
        problems.append(
            f"device {device!r} is refused. A CPU result is not an MPS result and not a CUDA result."
        )
    if request.get("scene_id") != SCENE_ID:
        problems.append(f"the only validation scene is {SCENE_ID}. The 96-scene attack is refused.")
    scene_count = request.get("scene_count", SCENE_COUNT)
    if scene_count != SCENE_COUNT:
        problems.append("scene_count must be 1.")
    if request.get("all_scenes") or request.get("run_full_development"):
        problems.append("the full development attack is refused.")
    image_relative = str(request.get("image_relative", SCENE_IMAGE_RELATIVE))
    if image_relative != SCENE_IMAGE_RELATIVE:
        problems.append("the image path is not the one development fixture.")
    if _banned(image_relative):
        problems.append("a held-out path is refused before the file is opened.")
    output = request.get("output_path")
    if output is not None and (_banned(str(output)) or _forbidden_output(str(output))):
        problems.append("the output path would overwrite a frozen result or a held-out path.")
    available = request.get("os_available_bytes")
    lines_cleared = None
    if available is None:
        problems.append("os_available_bytes is required. The guard is not skipped.")
    else:
        available_int = int(available)
        if available_int < BF16_REQUIRED_AVAILABLE_BYTES:
            lines_cleared = False
            problems.append(
                f"OS-available bytes {available_int} are below {BF16_REQUIRED_AVAILABLE_BYTES}. "
                "The guard was not lowered."
            )
        elif available_int < OS_FLOOR_BYTES:
            lines_cleared = False
            problems.append("OS-available bytes are below the 6 GiB floor. The guard was not lowered.")
        else:
            lines_cleared = True
    return {
        "approved": not problems,
        "problems": problems,
        "guard_lowered": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "full_development_attack_run": False,
        "scene_id": SCENE_ID,
        "scene_count": SCENE_COUNT,
        "device": device,
        "dtype": DTYPE,
        "float32_not_selected": True,
        "os_lines_cleared": lines_cleared,
        "mps_budget_decision_is_not_a_cuda_permit": True,
        "bf16_required_available_bytes": BF16_REQUIRED_AVAILABLE_BYTES,
        "float32_required_available_bytes": FLOAT32_REQUIRED_AVAILABLE_BYTES,
        "os_floor_bytes": OS_FLOOR_BYTES,
    }


def _banned(path: str) -> bool:
    return any(part.lower() in BANNED_PATH_PARTS for part in Path(path).parts)


def _forbidden_output(path: str) -> bool:
    text = path.replace("\\", "/")
    return any(prefix in text for prefix in FORBIDDEN_OUTPUT_PREFIXES)
