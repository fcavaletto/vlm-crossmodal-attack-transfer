"""The gate refuses CPU, a second scene, held-out paths, and a lowered guard."""

from __future__ import annotations

from pathlib import Path

from cuda_validation.gate import approve
from cuda_validation.measure import measure
from cuda_validation.pins import BF16_REQUIRED_AVAILABLE_BYTES, SCENE_ID

ROOT = Path(__file__).resolve().parents[2]


def _request(**overrides):
    request = {
        "device": "cuda",
        "scene_id": SCENE_ID,
        "scene_count": 1,
        "image_relative": "fixtures/development_v4/images/dev4-ref-00-a.png",
        "os_available_bytes": BF16_REQUIRED_AVAILABLE_BYTES,
    }
    request.update(overrides)
    return request


def test_approval_does_not_load_or_lower_the_guard():
    decision = approve(_request())
    assert decision["approved"] is True
    assert decision["model_from_pretrained_called"] is False
    assert decision["guard_lowered"] is False
    assert decision["float32_not_selected"] is True
    assert decision["mps_budget_decision_is_not_a_cuda_permit"] is True
    assert decision["os_lines_cleared"] is True


def test_cpu_and_the_full_attack_are_refused():
    cpu = approve(_request(device="cpu"))
    assert cpu["approved"] is False
    assert any("CPU" in problem for problem in cpu["problems"])
    many = approve(_request(scene_count=96, all_scenes=True))
    assert many["approved"] is False
    assert many["guard_lowered"] is False
    low = approve(_request(os_available_bytes=BF16_REQUIRED_AVAILABLE_BYTES - 1))
    assert low["approved"] is False
    assert low["os_lines_cleared"] is False
    assert any("not lowered" in problem for problem in low["problems"])


def test_held_out_paths_are_refused_before_a_measurement(tmp_path: Path):
    decision = approve(_request(output_path=str(tmp_path / "sealed" / "scene.json")))
    assert decision["approved"] is False
    assert decision["held_out_scenes_opened"] is False
    frozen = approve(_request(output_path="docs/checkpoint-2.0/cuda/scene.json"))
    assert frozen["approved"] is False
    result = measure(ROOT, device_name="cpu", os_available_bytes=BF16_REQUIRED_AVAILABLE_BYTES)
    assert result["status"] == "refused"
    assert result["attacks_run"] is False
    assert result["model_from_pretrained_called"] is False
    assert result["pod_launched_by_this_command"] is False
    assert result["runpod_api_called"] is False
