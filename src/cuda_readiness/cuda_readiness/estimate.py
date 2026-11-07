"""Planning estimate scaled from Checkpoint 1.9 MPS scene times.

The numbers are not CUDA measurements. No image is opened and no model is
loaded.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from .manifest import load_module_from_path
from .pins import (
    BF16_REQUIRED_AVAILABLE_BYTES,
    BF16_WEIGHT_BYTES,
    BUDGET_SCRIPT_RELATIVE,
    COMPARISON_FLOOR_BYTES,
    DEVELOPMENT_SCENE_COUNT,
    FLOAT32_REQUIRED_AVAILABLE_BYTES,
    GPU_HOST_MEMORY_GIB,
    GPU_HOURLY_RATE_USD,
    GPU_INSTANCE_TYPE,
    GPU_MEMORY_CLASS,
    GPU_NAME,
    GPU_REGION,
    GPU_VCPUS,
    HELD_OUT_PLANNING_SCENE_COUNT,
    OS_FLOOR_BYTES,
    PROGRESS_RELATIVE,
    SUMMARY_RELATIVE,
)

RATE = Decimal(GPU_HOURLY_RATE_USD)
PRIMARY_RATIO = Decimal("1")
SENSITIVITY_RATIOS = (Decimal("0.5"), Decimal("2"))
QUOTED_STUDIES = (
    ("development_v4", DEVELOPMENT_SCENE_COUNT),
    ("held_out_planning", HELD_OUT_PLANNING_SCENE_COUNT),
)


def build_estimate(root: Path) -> dict[str, Any]:
    """Wall time and on-demand cost for the measured Stage B grid."""
    progress = _load(root / PROGRESS_RELATIVE)
    summary = _load(root / SUMMARY_RELATIVE)
    stage_a = _scene_seconds(progress["stage_a_scenes"])
    stage_b = _scene_seconds(progress["stage_b_scenes"])
    if len(stage_a) != 8:
        raise RuntimeError(f"Stage A scene count is {len(stage_a)}, not 8.")
    if len(stage_b) != 16:
        raise RuntimeError(f"Stage B scene count is {len(stage_b)}, not 16.")
    stage_a_sum = sum(stage_a, Decimal("0"))
    stage_b_sum = sum(stage_b, Decimal("0"))
    load_seconds = progress["load_seconds"]
    if not isinstance(load_seconds, Decimal):
        raise RuntimeError("load_seconds was not read as a decimal.")
    per_scene = stage_b_sum / Decimal(len(stage_b))
    memory = _memory_estimate(root, summary)
    rows = []
    for name, count in QUOTED_STUDIES:
        for ratio in (PRIMARY_RATIO, *SENSITIVITY_RATIOS):
            rows.append(_row(name, count, per_scene, load_seconds, ratio))
    return {
        "schema": "cuda_readiness.planning_estimate.v1",
        "label": "planning estimate, not a CUDA measurement",
        "cuda_timing_measured": False,
        "cuda_numerical_validation": "not_executed",
        "attacks_run": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "guard_lowered": False,
        "source": {
            "progress": PROGRESS_RELATIVE,
            "device": "mps",
            "dtype": "bfloat16",
            "stage_a_scene_count": len(stage_a),
            "stage_a_seconds_sum": _dec(stage_a_sum),
            "stage_a_used_for_scaling": False,
            "stage_b_scene_count": len(stage_b),
            "stage_b_seconds_sum": _dec(stage_b_sum),
            "stage_b_seconds_per_scene": _dec(per_scene),
            "load_seconds": _dec(load_seconds),
            "finishing_process_wall_seconds": _dec(progress["wall_seconds"]),
            "finishing_process_wall_used_for_scaling": False,
        },
        "task": {
            "name": "checkpoint-1.9 stage B grid, one scene at a time",
            "source_context": "neutral",
            "eval_contexts": ["neutral", "helpful_nonrevealing"],
            "budgets": ["1/255", "2/255", "4/255"],
            "pgd_steps": 7,
            "step_size": "epsilon/4",
            "init": "clean image",
            "restarts": 0,
            "controls": ["one positive sign step", "one matched random-sign step"],
            "definition_changed": False,
        },
        "gpu": {
            "provider": "AWS EC2",
            "instance_type": GPU_INSTANCE_TYPE,
            "region": GPU_REGION,
            "operating_system": "Linux",
            "purchase": "on-demand",
            "gpu_count": 1,
            "gpu_name": GPU_NAME,
            "gpu_memory_class": GPU_MEMORY_CLASS,
            "vcpus": GPU_VCPUS,
            "host_memory_gib": GPU_HOST_MEMORY_GIB,
            "hourly_rate_usd": GPU_HOURLY_RATE_USD,
            "rate_excludes": ["EBS", "public IPv4", "data transfer", "tax"],
            "sources": [
                "https://calculator.holori.com/aws/ec2/g6.2xlarge",
                "https://instances.vantage.sh/aws/ec2/g6.2xlarge",
                "https://cloud-credits.org/prices/ec2/g6-2xlarge/us-east-1",
            ],
            "source_note": (
                "Holori lists $0.9776 per hour for Linux on-demand in us-east-1, "
                "updated 2026-11-06. Vantage and Cloud Credits list the same rate."
            ),
        },
        "memory": memory,
        "assumptions": [
            "Estimated CUDA scene time equals measured MPS scene time when assumed_cuda_seconds_per_mps_second is 1. No CUDA timing was measured, so 1 is a placeholder, not a benchmark.",
            "Scene time scales linearly. The runner scores one scene at a time.",
            "The measured MPS load of the finishing process is added once and is not scaled. It is not a CUDA load time.",
            "The scaler is the sum of the sixteen Stage B scene seconds. Stage A is a single-step grid and is not the scaler. The finishing process wall clock excludes an earlier process and is not the scaler.",
            "The 96-scene row is that same Stage B grid on every development_v4 scene. It is not the two-budget calibration grid.",
            "The held-out row uses 256 scenes, the sealed-set size written in docs/research/RESEARCH_PROPOSAL.md section 12. No held-out file was opened. A different sealed count scales by the same per-scene seconds.",
            "The hourly rate is the published on-demand Linux price. EBS, public IPv4, data transfer, and tax are not included.",
            "Ratios 0.5 and 2 are sensitivities. They are not measurements.",
        ],
        "rows": rows,
    }


def quoted_rows(estimate: dict[str, Any]) -> list[dict[str, Any]]:
    """The two prices this checkpoint quotes. Each carries the GPU class and the rate."""
    return [row for row in estimate["rows"] if row["quoted"] is True]


def _row(
    study: str,
    scene_count: int,
    per_scene: Decimal,
    load_seconds: Decimal,
    ratio: Decimal,
) -> dict[str, Any]:
    scene_seconds = per_scene * Decimal(scene_count) * ratio
    total = load_seconds + scene_seconds
    hours = total / Decimal(3600)
    cost = hours * RATE
    return {
        "study": study,
        "scene_count": scene_count,
        "assumed_cuda_seconds_per_mps_second": _dec(ratio),
        "quoted": ratio == PRIMARY_RATIO,
        "gpu_instance_type": GPU_INSTANCE_TYPE,
        "gpu_name": GPU_NAME,
        "region": GPU_REGION,
        "hourly_rate_usd": GPU_HOURLY_RATE_USD,
        "estimated_scene_seconds": _q(scene_seconds, "0.001"),
        "estimated_wall_seconds": _q(total, "0.001"),
        "estimated_wall_hours": _q(hours, "0.001"),
        "estimated_on_demand_usd": _q(cost, "0.01"),
        "cuda_timing_measured": False,
    }


def _memory_estimate(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    budget = load_module_from_path(root / BUDGET_SCRIPT_RELATIVE, "checkpoint15_budget_lines")
    if budget.QWEN_C0_BF16_BYTES != BF16_WEIGHT_BYTES:
        raise RuntimeError("The bfloat16 weight byte count moved. The estimate will not invent a new one.")
    if 6 * 1024**3 != OS_FLOOR_BYTES:
        raise RuntimeError("The 6 GiB floor constant moved.")
    bf16 = budget.required_available_bytes(
        budget.QWEN_C0_BF16_BYTES,
        budget.C1_DROP_PER_CURRENT,
        min_available_bytes=OS_FLOOR_BYTES,
    )
    float32 = budget.required_available_bytes(
        budget.QWEN_C0_FLOAT32_BYTES,
        budget.C1_DROP_PER_CURRENT,
        min_available_bytes=OS_FLOOR_BYTES,
    )
    if bf16["calibrated_projection"] != BF16_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError(
            "The bfloat16 calibrated projection is not 11,509,447,200. "
            "The guard was not replaced with the new number."
        )
    if float32["calibrated_projection"] != FLOAT32_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError("The float32 calibrated projection is not 20,871,410,752.")
    if bf16["os_floor"] != OS_FLOOR_BYTES:
        raise RuntimeError("The 6 GiB floor is not 6,442,450,944.")
    peak = summary["peak_memory"]
    driver_mb = peak["peak_mps_driver_allocated_mb"]
    current_mb = peak["peak_mps_current_allocated_mb"]
    driver_bytes = int((driver_mb * Decimal(1024**2)).to_integral_value(rounding=ROUND_HALF_EVEN))
    return {
        "kind": "estimate",
        "local_cuda_allocator_check": False,
        "cuda_numerical_validation": "not_executed",
        "guard_lowered": False,
        "bf16_weight_bytes": budget.QWEN_C0_BF16_BYTES,
        "unchanged_bf16_lines_bytes": {key: int(value) for key, value in bf16.items()},
        "unchanged_float32_required_available_bytes": int(float32["calibrated_projection"]),
        "observed_mps_peak_driver_allocated_mb": _dec(driver_mb),
        "observed_mps_peak_current_allocated_mb": _dec(current_mb),
        "observed_mps_peak_driver_bytes": driver_bytes,
        "comparison_floor_bytes": COMPARISON_FLOOR_BYTES,
        "comparison_floor_note": (
            "22 GiB, the smaller published L4 listing. AWS pages also say 24 GB. "
            "This floor was not read from a CUDA device."
        ),
        "weight_bytes_fit_under_comparison_floor": budget.QWEN_C0_BF16_BYTES < COMPARISON_FLOOR_BYTES,
        "calibrated_projection_fits_under_comparison_floor": (
            bf16["calibrated_projection"] < COMPARISON_FLOOR_BYTES
        ),
        "observed_driver_snapshot_fits_under_comparison_floor": driver_bytes < COMPARISON_FLOOR_BYTES,
        "host_memory_gib": GPU_HOST_MEMORY_GIB,
        "why_this_host_size": (
            "g6.2xlarge has 32 GiB of host memory, so the unchanged bfloat16 line of "
            "11,509,447,200 OS-available bytes can clear. g6.xlarge has 16 GiB of host "
            "memory and is not the quoted class. The line was not lowered and was not "
            "evaluated on either instance. The existing budget decision also requires "
            "MPS, and the Checkpoint 1.9 runner requests MPS. This estimate does not "
            "change either fact."
        ),
    }


def _scene_seconds(scenes: dict[str, Any]) -> list[Decimal]:
    seconds = []
    for row in scenes.values():
        value = row["seconds"]
        if not isinstance(value, Decimal):
            raise RuntimeError("scene seconds were not read as decimals")
        if row.get("finished") is not True:
            raise RuntimeError(f"{row.get('scene_id')} is not a finished scene")
        seconds.append(value)
    return seconds


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def _dec(value: Decimal) -> str:
    return format(value, "f")


def _q(value: Decimal, places: str) -> str:
    return format(value.quantize(Decimal(places), rounding=ROUND_HALF_EVEN), "f")
