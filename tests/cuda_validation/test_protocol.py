"""The plan matches the frozen attack and does not launch a pod."""

from __future__ import annotations

import json
from pathlib import Path

from attack_feasibility.protocol import PGD_STEPS, STAGE_B_SOURCE_CONTEXT, epsilon
from cuda_validation.pins import EPSILON, SCENE_ID, SOURCE_CONTEXT, STEPS
from cuda_validation.protocol import build_protocol

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "cuda_validation"


def test_protocol_keeps_the_frozen_attack_and_the_quoted_card():
    protocol = build_protocol(ROOT)
    assert protocol["billable_gpu_launched"] is False
    assert protocol["runpod_api_called"] is False
    assert protocol["model_from_pretrained_called"] is False
    assert protocol["attacks_run"] is False
    assert protocol["held_out_scenes_opened"] is False
    assert protocol["executed_on_cuda"] is False
    assert protocol["executed_on_mps"] is False
    assert protocol["devices_interchangeable"] is False
    assert protocol["cuda_numerical_validation"] == "not_executed"
    assert protocol["attack_files_unchanged"] is True
    assert protocol["guard_lowered"] is False
    assert protocol["scene"]["scene_id"] == SCENE_ID
    assert protocol["scene"]["scene_count"] == 1
    assert protocol["scene"]["image_opened_by_this_command"] is False
    assert protocol["attack"]["steps"] == 3
    assert protocol["attack"]["steps"] != PGD_STEPS
    assert protocol["attack"]["epsilon_float"] == epsilon(1)
    assert protocol["attack"]["epsilon_float"] == EPSILON
    assert protocol["attack"]["source_context"] == STAGE_B_SOURCE_CONTEXT
    assert SOURCE_CONTEXT == "neutral"
    assert protocol["attack"]["restarts"] == 0
    assert protocol["attack"]["definition_changed"] is False
    assert protocol["gpu"]["name"] == "NVIDIA L4"
    assert protocol["gpu"]["hourly_rate_usd"] == "0.59"
    assert protocol["gpu"]["cloud"] == "Secure Cloud"
    assert protocol["gpu"]["price_read_on"] == "2026-11-07"
    assert protocol["gpu"]["launched"] is False
    assert protocol["memory_lines_bytes"]["bf16_calibrated_projection"] == 11_509_447_200
    assert protocol["memory_lines_bytes"]["float32_calibrated_projection"] == 20_871_410_752
    assert protocol["memory_lines_bytes"]["os_floor"] == 6_442_450_944
    assert protocol["preflight"]["cuda_numerical_validation"] == "not_executed"
    assert protocol["preflight"]["attacks_run"] is False
    assert STEPS == 3
    text = (ROOT / "docs/checkpoint-2.1/CUDA_VALIDATION.md").read_text(encoding="utf-8")
    for piece in (
        "NVIDIA L4",
        "$0.59",
        "2026-11-07",
        "https://www.runpod.io/pricing",
        "11,509,447,200",
        "dev4-ref-00-a",
        "No billable GPU was launched.",
        "devices_interchangeable",
    ):
        assert piece in text


def test_pod_spec_was_not_posted_and_the_image_default_is_preflight():
    pod = json.loads((ROOT / "docker/runpod/pod-spec.json").read_text(encoding="utf-8"))
    assert pod["launched"] is False
    assert pod["billable_gpu_launched"] is False
    assert pod["doNotPost"] is True
    assert pod["apiCallMade"] is False
    assert pod["gpu"]["hourlyRateUsd"] == "0.59"
    assert pod["gpu"]["name"] == "NVIDIA L4"
    assert pod["gpu"]["minCudaVersion"] == "13.2"
    assert pod["experimentCommandExecuted"] is False
    assert "attack_feasibility" not in " ".join(pod["startCommand"])
    assert "measure" in pod["experimentCommand"]
    dockerfile = (ROOT / "docker/cuda/Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["preflight"]' in dockerfile
    assert "src/cuda_validation" in dockerfile
    assert "https://download.pytorch.org/whl/cu132" in dockerfile


def test_validation_package_does_not_redefine_the_attack():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    for banned in (
        ".from_pretrained(",
        "def pgd_update",
        "def project_linf",
        "def backward_at_image",
        "def differentiable_model_pixels",
        "select_dtype(",
        "prepare_environment(",
        "stage_b",
        "run_scene",
        "runpodctl",
        "api.runpod.io",
    ):
        assert banned not in text
