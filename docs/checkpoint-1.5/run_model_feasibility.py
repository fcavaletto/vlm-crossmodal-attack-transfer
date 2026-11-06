"""Checkpoint 1.5 model-feasibility gates.

Surveys weight headers and decides whether ``Qwen/Qwen3-VL-2B-Instruct`` may be
loaded on this 24 GB Mac. The script does not call ``from_pretrained``, does
not download weight tensors, and does not run an attack.

Gates, in order, for the Qwen candidate:

1. Weight load and memory. A failing budget aborts before any parameter load.
2. Minimal forward. Started only after gate 1 permits a load.
3. Clean answers on a small development subset. Started only after gate 2.
4. Image gradients. Started only after gate 3.

SmolVLM2 task ability is taken from the Checkpoint 1 MPS pilot. This script
does not load that model again.

Run from the repository root:

    HF_HOME=/Users/fedecava/Development/models/huggingface \\
    PYTHONPATH=src/vlm_iface:src/pilot \\
    python docs/checkpoint-1.5/run_model_feasibility.py
"""

from __future__ import annotations

import ast
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

# Checkpoint 1 pilot report, section 8. The MB figures are the one-decimal
# values printed by vlm_iface.devices.memory_snapshot (bytes / 1024**2).
# They describe one float32 SmolVLM2 MPS load, not a Qwen load.
C1_AVAILABLE_BEFORE_MB = 8074.4
C1_AVAILABLE_AFTER_LOAD_MB = 3749.7
C1_MPS_CURRENT_ALLOCATED_MB = 1965.6
C1_DROP_PER_CURRENT = (C1_AVAILABLE_BEFORE_MB - C1_AVAILABLE_AFTER_LOAD_MB) / C1_MPS_CURRENT_ALLOCATED_MB

# Room that must remain available above the resident weight bytes before a
# from_pretrained call. The Checkpoint 1 load had about 6.00 GiB of this spare
# and then ran full-resolution gradients. Five GiB stays under that positive
# control and refuses a tighter candidate.
SPARE_REQUIRED_BYTES = 5 * 1024**3

# Floor on the memory projected to remain after applying the Checkpoint 1
# available-drop ratio. Two GiB is below the ~3.7 GiB the pilot still had
# after its load, so the positive control passes.
PROJECTED_REMAINING_FLOOR_BYTES = 2 * 1024**3

SMOLVLM_TENSOR_BYTES = 2_029_929_216
QWEN_C0_PARAMETERS = 2_127_532_032
QWEN_C0_FLOAT32_BYTES = 8_510_128_128
QWEN_C0_BF16_BYTES = QWEN_C0_PARAMETERS * 2

REPRODUCTION_COMMAND = (
    "HF_HOME=/Users/fedecava/Development/models/huggingface "
    "PYTHONPATH=src/vlm_iface:src/pilot "
    "python docs/checkpoint-1.5/run_model_feasibility.py"
)


def required_available_bytes(
    resident_bytes: int,
    ratio: float,
    *,
    min_available_bytes: int,
    spare_required_bytes: int = SPARE_REQUIRED_BYTES,
    projected_floor_bytes: int = PROJECTED_REMAINING_FLOOR_BYTES,
) -> dict[str, int]:
    """Bytes of OS-available memory each safety line demands before a load."""
    if resident_bytes < 0:
        raise ValueError("resident_bytes must be non-negative")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    lines = {
        "os_floor": int(min_available_bytes),
        "spare_above_weights": int(resident_bytes + spare_required_bytes),
        "calibrated_projection": int(math.ceil(resident_bytes * ratio) + projected_floor_bytes),
        "staging_double_copy": int(2 * resident_bytes),
    }
    return lines


def budget_decision(
    *,
    resident_bytes: int,
    available_bytes: int,
    ratio: float,
    min_available_bytes: int,
    mps_available: bool,
) -> dict[str, Any]:
    """Return whether a from_pretrained call is inside the unified-memory budget.

    ``available_bytes`` is what the OS reports as available, including
    reclaimable cache. A true result would be permission to start gate 1's
    load. It is not permission to adopt the model, and it is not an MPS timing.
    """
    lines = required_available_bytes(
        resident_bytes, ratio, min_available_bytes=min_available_bytes
    )
    failed = [name for name, need in lines.items() if available_bytes < need]
    binding = max(lines, key=lines.get)
    permitted = mps_available and not failed
    if not mps_available:
        reason = "MPS is not available. A CPU load is not an MPS result and was not started."
    elif failed:
        reason = (
            "OS-available memory is below the pre-load budget. "
            "from_pretrained must not be called. Failed lines: " + ", ".join(failed) + "."
        )
    else:
        reason = "OS-available memory clears every pre-load line."
    return {
        "resident_bytes": resident_bytes,
        "available_bytes": available_bytes,
        "required_available_bytes": lines[binding],
        "lines_bytes": lines,
        "binding_line": binding,
        "failed_lines": failed,
        "mps_available": mps_available,
        "permitted": permitted,
        "reason": reason,
    }


def assert_budget_fixtures(min_available_bytes: int) -> None:
    """Refuse to run if the budget would block the known-safe SmolVLM load or allow Qwen today."""
    pilot_available = int(round(C1_AVAILABLE_BEFORE_MB * 1024 * 1024))
    smol = budget_decision(
        resident_bytes=SMOLVLM_TENSOR_BYTES,
        available_bytes=pilot_available,
        ratio=C1_DROP_PER_CURRENT,
        min_available_bytes=min_available_bytes,
        mps_available=True,
    )
    if not smol["permitted"]:
        raise RuntimeError(
            "Budget fixture failed: the Checkpoint 1 SmolVLM2 load would have been refused. "
            f"{json.dumps(smol)}"
        )
    observed = 8_145_289_216
    for dtype, resident in (("float32", QWEN_C0_FLOAT32_BYTES), ("bfloat16", QWEN_C0_BF16_BYTES)):
        decision = budget_decision(
            resident_bytes=resident,
            available_bytes=observed,
            ratio=C1_DROP_PER_CURRENT,
            min_available_bytes=min_available_bytes,
            mps_available=True,
        )
        if decision["permitted"]:
            raise RuntimeError(f"Budget fixture failed: Qwen {dtype} was permitted at {observed} available bytes.")
    float32_lines = required_available_bytes(QWEN_C0_FLOAT32_BYTES, C1_DROP_PER_CURRENT, min_available_bytes=min_available_bytes)
    # 8,510,128,128 * (4324.7/1965.6) + 2 GiB is 19.44 GiB. A regression that
    # shrinks that line below 19 GiB would no longer match this calibration.
    if float32_lines["calibrated_projection"] <= 19 * 1024**3:
        raise RuntimeError("Budget fixture failed: float32 projection is unexpectedly small.")
    bf16_lines = required_available_bytes(QWEN_C0_BF16_BYTES, C1_DROP_PER_CURRENT, min_available_bytes=min_available_bytes)
    if bf16_lines["calibrated_projection"] <= 10 * 1024**3:
        raise RuntimeError("Budget fixture failed: bfloat16 projection is unexpectedly small.")


def assert_script_has_no_pretrained_call(path: Path) -> None:
    """The gate-1 script is not allowed to grow a weight-load call by accident."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "from_pretrained":
            raise RuntimeError("This script contains a from_pretrained call. Remove it.")


def main() -> None:
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    assert_script_has_no_pretrained_call(script_path)

    from vlm_iface.candidate import survey_weights
    from vlm_iface.pins import MIN_AVAILABLE_BYTES, MODEL_ID, PINNED_REVISION, QWEN_CANDIDATE_ID

    assert_budget_fixtures(MIN_AVAILABLE_BYTES)

    import psutil
    import torch

    memory = psutil.virtual_memory()
    available = int(memory.available)
    mps_available = bool(torch.backends.mps.is_available())
    print("Surveying weight headers. Tensor bytes are not downloaded.", flush=True)
    qwen_survey = _survey(survey_weights, QWEN_CANDIDATE_ID)
    smol_survey = _survey(survey_weights, MODEL_ID, revision=PINNED_REVISION)

    qwen_decisions = _qwen_decisions(
        qwen_survey,
        available=available,
        active_bytes=int(getattr(memory, "active", 0)),
        min_available_bytes=MIN_AVAILABLE_BYTES,
        mps_available=mps_available,
    )
    permitted = any(item["permitted"] for item in qwen_decisions.values())
    preprocessor = _preprocessor_note(qwen_survey)

    payload: dict[str, Any] = {
        "measurement_kind": "checkpoint15_model_feasibility",
        "label": "DEVELOPMENT ONLY",
        "checkpoint": "checkpoint-1.5",
        "attacks_run": False,
        "training_run": False,
        "cloud_gpu": False,
        "held_out_opened": False,
        "h_star_fitted": False,
        "from_pretrained_called": False,
        "weights_downloaded": False,
        "reproduction_command": REPRODUCTION_COMMAND,
        "machine": _machine(memory),
        "versions": _versions(),
        "mps_available": mps_available,
        "mps_model_measurement": None,
        "mps_model_measurement_note": (
            "No model was placed on MPS. There is no MPS load time, MPS allocation, "
            "or MPS forward time in this record. mps_available is only a capability flag."
        ),
        "calibration": {
            "source": "docs/checkpoint-1/PILOT_REPORT.md section 8",
            "available_before_load_mb": C1_AVAILABLE_BEFORE_MB,
            "available_after_load_mb": C1_AVAILABLE_AFTER_LOAD_MB,
            "mps_current_allocated_mb": C1_MPS_CURRENT_ALLOCATED_MB,
            "drop_per_current_allocation": C1_DROP_PER_CURRENT,
            "note": (
                "The ratio is an extrapolation from one SmolVLM2 float32 MPS load. "
                "It is a pre-load safety bound for a larger model, not a Qwen measurement. "
                "MB values are the published one-decimal figures."
            ),
        },
        "budget_constants_bytes": {
            "min_available": MIN_AVAILABLE_BYTES,
            "spare_required": SPARE_REQUIRED_BYTES,
            "projected_remaining_floor": PROJECTED_REMAINING_FLOOR_BYTES,
        },
        "os_memory_bytes": {
            "total": int(memory.total),
            "available": available,
            "used": int(memory.used),
            "free": int(memory.free),
            "active": int(getattr(memory, "active", 0)),
            "inactive": int(getattr(memory, "inactive", 0)),
            "wired": int(getattr(memory, "wired", 0)),
        },
        "qwen_candidate": qwen_survey,
        "qwen_dtype_decisions": qwen_decisions,
        "qwen_preprocessor": preprocessor,
        "qwen_local_weight_file_present": False,
        "smolvlm": {
            "model_id": MODEL_ID,
            "pinned_revision": PINNED_REVISION,
            "weights_loaded_this_checkpoint": False,
            "new_subset_ran": False,
            "reason": (
                "Task ability is the Checkpoint 1 24-scene MPS pilot. "
                "A second load was not required to answer that question and was not started."
            ),
            "task_ability_source": {
                "commit": "90c6ffe83d61514da217a20f8dfc2f202afbbe69",
                "report": "docs/checkpoint-1/PILOT_REPORT.md",
                "summary": "docs/checkpoint-1/summary.json",
                "predictions": "docs/checkpoint-1/predictions.jsonl",
                "label": "DEVELOPMENT ONLY",
                "device": "mps",
                "parameter_dtype": "torch.float32",
            },
            "header_survey": smol_survey,
        },
    }
    weight_present = _local_weight_present(qwen_survey)
    payload["qwen_local_weight_file_present"] = weight_present
    payload["weights_downloaded"] = False
    if weight_present:
        payload["local_weight_note"] = (
            "model.safetensors is in the local Hub snapshot. This script did not download it "
            "and did not load it."
        )
    if permitted:
        payload["gates"] = {
            "weight_load_and_memory": {
                "status": "budget_passed_load_not_implemented",
                "from_pretrained_called": False,
            },
            "minimal_forward": {"status": "not_started"},
            "clean_answers": {"status": "not_started"},
            "image_gradients": {"status": "not_started"},
        }
        payload["action"] = "stopped_before_load"
        payload["passed_gate_1"] = False
        payload["survey_wall_seconds"] = round(time.perf_counter() - started, 3)
        _write(payload)
        raise SystemExit(
            "The memory budget permitted a Qwen load, but this script has no weight-load path. "
            "No weights were loaded. Add an MPS load only in a later revision of this script, "
            "after re-checking the budget in the same process."
        )

    payload["gates"] = {
        "weight_load_and_memory": {
            "status": "aborted",
            "from_pretrained_called": False,
            "weights_loaded": False,
            "reason": "No dtype cleared the pre-load budget. See qwen_dtype_decisions.",
        },
        "minimal_forward": {
            "status": "not_started",
            "reason": "Gate 1 aborted before a load.",
        },
        "clean_answers": {
            "status": "not_started",
            "reason": "Gate 1 aborted before a load.",
        },
        "image_gradients": {
            "status": "not_started",
            "reason": "Gate 1 aborted before a load. Gradients are not run after a failed earlier gate.",
        },
    }
    payload["action"] = "load_aborted"
    payload["adopted"] = False
    payload["passed_gate_1"] = False
    payload["comparison_with_smolvlm"] = (
        "Not run. Qwen produced no scores and no generations, so there is no paired subset."
    )
    payload["survey_wall_seconds"] = round(time.perf_counter() - started, 3)
    _write(payload)
    print(json.dumps({
        "action": payload["action"],
        "passed_gate_1": False,
        "from_pretrained_called": False,
        "mps_available": mps_available,
        "available_bytes": available,
        "output": "docs/checkpoint-1.5/feasibility-record.json",
    }, indent=2))


def _qwen_decisions(
    survey: dict[str, Any],
    *,
    available: int,
    active_bytes: int,
    min_available_bytes: int,
    mps_available: bool,
) -> dict[str, Any]:
    if survey.get("survey_error"):
        refused = {
            "permitted": False,
            "from_pretrained_called": False,
            "reason": "Header survey failed. The safe action is to skip the load.",
            "survey_error": survey["survey_error"],
        }
        return {"float32": refused, "bfloat16": refused}
    parameter_count = int(survey["parameter_count"])
    float32_bytes = int(survey["float32_weight_bytes"])
    stored_counts = survey.get("dtype_counts") or {}
    decisions = {
        "float32": budget_decision(
            resident_bytes=float32_bytes,
            available_bytes=available,
            ratio=C1_DROP_PER_CURRENT,
            min_available_bytes=min_available_bytes,
            mps_available=mps_available,
        ),
        "bfloat16": budget_decision(
            resident_bytes=parameter_count * 2,
            available_bytes=available,
            ratio=C1_DROP_PER_CURRENT,
            min_available_bytes=min_available_bytes,
            mps_available=mps_available,
        ),
    }
    for name, decision in decisions.items():
        decision["dtype"] = name
        decision["from_pretrained_called"] = False
        decision["checkpoint0_float32_guard_bytes"] = 6 * 1024**3
        decision["checkpoint0_float32_guard_permits_load"] = float32_bytes <= 6 * 1024**3
    decisions["bfloat16"]["stored_dtype_counts"] = stored_counts
    decisions["float32"]["note"] = (
        "Resident size is the float32 expansion of the surveyed header, "
        "not the on-disk BF16 file."
    )
    decisions["bfloat16"]["note"] = (
        "Resident size is two bytes times the surveyed parameter count. "
        "The Checkpoint 0 file was BF16. This is the native-width load that a later "
        "checkpoint could attempt. It was not attempted here."
    )
    for decision in decisions.values():
        decision["available_plus_active_bytes"] = available + active_bytes
        decision["available_plus_active_would_clear"] = (
            available + active_bytes
        ) >= decision["required_available_bytes"]
    return decisions


def _survey(survey_weights, repo_id: str, revision: str | None = None) -> dict[str, Any]:
    try:
        surveyed = survey_weights(repo_id, revision=revision)
    except Exception as exc:
        return {
            "repo_id": repo_id,
            "revision": revision,
            "weights_loaded": False,
            "survey_error": f"{type(exc).__name__}: {exc}",
        }
    surveyed["weights_loaded"] = False
    surveyed["from_pretrained_called"] = False
    return surveyed


def _preprocessor_note(survey: dict[str, Any]) -> dict[str, Any]:
    """Read the processor config only. This does not load parameters."""
    if survey.get("survey_error") or not survey.get("revision"):
        return {"downloaded": False, "reason": "no revision from the weight survey"}
    from huggingface_hub import hf_hub_download

    repo_id = survey["repo_id"]
    revision = survey["revision"]
    try:
        config_path = hf_hub_download(repo_id, "config.json", revision=revision)
        processor_path = hf_hub_download(repo_id, "preprocessor_config.json", revision=revision)
    except Exception as exc:
        return {"downloaded": False, "error": f"{type(exc).__name__}: {exc}"}
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    with open(processor_path, encoding="utf-8") as handle:
        processor = json.load(handle)
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    patch = int(processor.get("patch_size") or vision.get("patch_size") or 0)
    merge = int(processor.get("merge_size") or vision.get("spatial_merge_size") or 0)
    size = processor.get("size") if isinstance(processor.get("size"), dict) else {}
    min_pixels = processor.get("min_pixels", size.get("shortest_edge"))
    max_pixels = processor.get("max_pixels", size.get("longest_edge"))
    note: dict[str, Any] = {
        "downloaded": True,
        "weights_included": False,
        "image_processor_type": processor.get("image_processor_type"),
        "patch_size": patch or None,
        "merge_size": merge or None,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "development_image_hw": [384, 512],
    }
    if patch and merge and isinstance(min_pixels, int) and isinstance(max_pixels, int):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        factor = patch * merge
        height, width = smart_resize(384, 512, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
        grid_h = height // patch
        grid_w = width // patch
        tokens = (grid_h * grid_w) // (merge * merge)
        note["estimated_resize_hw"] = [height, width]
        note["estimated_llm_image_tokens"] = tokens
        note["estimate_formula"] = (
            "smart_resize(384, 512, factor=patch*merge, min_pixels, max_pixels); "
            "tokens = (h/patch)*(w/patch)/(merge*merge). Not a measured forward."
        )
    return note


def _local_weight_present(survey: dict[str, Any]) -> bool:
    """True only when the weight file is already in the local Hub cache. Does not download it."""
    revision = survey.get("revision")
    repo_id = survey.get("repo_id")
    if not revision or not repo_id:
        return False
    import os

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    folder = "models--" + repo_id.replace("/", "--")
    path = hf_home / "hub" / folder / "snapshots" / str(revision) / "model.safetensors"
    return path.is_file()


def _machine(memory) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "memory_bytes": int(memory.total),
    }
    if platform.system() == "Darwin":
        import subprocess

        summary["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        summary["hw_memsize"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    return summary


def _versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _write(payload: dict[str, Any]) -> None:
    output = Path("docs/checkpoint-1.5/feasibility-record.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
