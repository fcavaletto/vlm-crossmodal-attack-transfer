"""Equal scalars are still not an interchangeable pair of devices."""

from __future__ import annotations

import json
from pathlib import Path

from cuda_validation.compare import IdenticalDevicesError, compare_records
from cuda_validation.pins import EPSILON, REVISION, STORED_HALF_LEVEL

NLL = {"red": 1.0, "blue": 2.0, "green": 3.0, "yellow": 4.0}


def _section(loss: float) -> dict:
    return {"summed_nll": dict(NLL), "chosen": "red", "loss": loss}


def _hardware(device: str) -> dict:
    if device == "mps":
        return {
            "device_type": "mps",
            "device_name": "mps",
            "compute_capability": None,
            "compute_capability_null_reason": "MPS does not report a CUDA compute capability.",
            "driver_version": "macOS",
            "torch_version": "2.15.0",
            "cuda_runtime_version": None,
            "cuda_runtime_null_reason": "MPS has no CUDA runtime. Null is not CUDA 0.",
            "parameter_dtype": "torch.bfloat16",
            "rgb_leaf_dtype": "torch.float32",
            "attn_implementation": "eager",
            "allow_tf32_matmul": None,
            "allow_tf32_matmul_null_reason": "MPS has no TF32 switch. Null is not false.",
            "allow_tf32_cudnn": None,
            "allow_tf32_cudnn_null_reason": "MPS has no TF32 switch. Null is not false.",
            "cudnn_benchmark": None,
            "cudnn_benchmark_null_reason": "MPS has no cuDNN benchmark flag.",
            "cudnn_deterministic": None,
            "cudnn_deterministic_null_reason": "MPS has no cuDNN deterministic flag.",
            "float32_matmul_precision": "highest",
            "bf16_reduced_precision_reduction": None,
            "bf16_reduced_precision_reduction_null_reason": "No CUDA reduction flag was read on MPS.",
            "os_available_bytes": 16_000_000_000,
            "accelerator_memory": {
                "kind": "mps_unified",
                "driver_allocated_bytes": 100,
                "current_allocated_bytes": 50,
            },
        }
    return {
        "device_type": "cuda",
        "device_name": "NVIDIA L4",
        "compute_capability": "8.9",
        "driver_version": "580.65.06",
        "torch_version": "2.15.0+cu132",
        "cuda_runtime_version": "13.2",
        "parameter_dtype": "torch.bfloat16",
        "rgb_leaf_dtype": "torch.float32",
        "attn_implementation": "eager",
        "allow_tf32_matmul": True,
        "allow_tf32_cudnn": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "float32_matmul_precision": "highest",
        "bf16_reduced_precision_reduction": True,
        "os_available_bytes": 40_000_000_000,
        "accelerator_memory": {
            "kind": "cuda_vram",
            "total_bytes": 23_000_000_000,
            "allocated_bytes": 100,
            "reserved_bytes": 200,
        },
    }


def _record(device: str, **overrides) -> dict:
    record = {
        "device_type": device,
        "status": "completed",
        "scene_id": "dev4-ref-00-a",
        "scene_count": 1,
        "revision": REVISION,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "epsilon": "1/255",
        "steps": 3,
        "loss_identity_agreed": True,
        "losses": {"trajectory": [1.0, 1.1, 1.2, 1.3]},
        "outputs": {
            "clean_official": _section(1.5),
            "clean_differentiable": _section(1.25),
            "final_official": _section(2.0),
        },
        "gradient": {"values": [1.0, -1.0, 0.0], "sha256": "same-bytes"},
        "bounds": {
            "epsilon": "1/255",
            "epsilon_float": EPSILON,
            "steps": 3,
            "restarts": 0,
            "init": "clean_image",
            "step_size_label": "epsilon/4",
            "step_size_float": EPSILON / 4.0,
            "linf_float": 3.0 * EPSILON / 4.0,
            "linf_stored_vs_clean": EPSILON,
            "pixels_in_unit_interval": True,
            "per_step_linf_float": [EPSILON / 4.0, EPSILON / 2.0, 3.0 * EPSILON / 4.0],
        },
        "hardware": _hardware(device),
        "memory": {"os_available_bytes": _hardware(device)["os_available_bytes"]},
    }
    record.update(overrides)
    return record


def test_matching_scalars_are_not_interchangeable():
    report = compare_records(_record("mps"), _record("cuda"))
    assert report["status"] == "differences_recorded"
    assert report["numeric_values_equal"] is True
    assert report["devices_interchangeable"] is False
    assert report["mps_and_cuda_treated_as_identical"] is False
    assert report["cuda_measurement_is_an_mps_measurement"] is False
    assert report["gradients"]["identity_threshold_applied"] is False
    assert report["gradients"]["cosine"] != 1.0
    assert report["gradients"]["cosine"] > 0.999
    assert report["memory_quantities_are_distinct"] is True
    assert "memory_delta_bytes" not in report
    assert any("TF32" in item for item in report["must_check"])
    assert report["hardware_flags"]["cuda_allow_tf32_matmul"] is True
    assert report["hardware_flags"]["mps_allow_tf32_matmul"] is None


def test_a_gradient_difference_is_recorded_and_not_passed():
    cuda = _record("cuda", gradient={"values": [1.0, 1.0, 0.0], "sha256": "other-bytes"})
    report = compare_records(_record("mps"), cuda)
    assert report["status"] == "differences_recorded"
    assert report["numeric_values_equal"] is False
    assert report["devices_interchangeable"] is False
    assert report["gradients"]["sign_agreement"] == 2.0 / 3.0
    assert report["gradients"]["cosine"] != 1.0
    assert report["gradients"]["identity_threshold_applied"] is False


def test_bounds_flags_and_protocol_stay_separate():
    broken = _record("cuda")
    broken["bounds"]["linf_float"] = 0.1
    report = compare_records(_record("mps"), broken)
    assert report["status"] == "bound_violation"
    assert report["bounds"]["cuda"]["satisfied"] is False
    assert report["bounds"]["mps"]["satisfied"] is True
    assert report["devices_interchangeable"] is False

    missing_flag = _record("cuda")
    missing_flag["hardware"].pop("allow_tf32_matmul")
    assert compare_records(_record("mps"), missing_flag)["status"] == "blocked_precision_record"
    missing_name = _record("cuda")
    missing_name["hardware"].pop("device_name")
    assert compare_records(_record("mps"), missing_name)["status"] == "incomplete_hardware_record"

    false_flag = _record("mps")
    false_flag["hardware"]["allow_tf32_matmul"] = False
    assert compare_records(false_flag, _record("cuda"))["status"] == "blocked_precision_record"

    longer = _record("cuda", steps=7)
    assert compare_records(_record("mps"), longer)["status"] == "blocked_protocol_mismatch"


def test_identical_treatment_is_refused():
    try:
        compare_records(_record("mps"), _record("cuda"), treat_as_identical=True)
    except IdenticalDevicesError:
        return
    raise AssertionError("identical treatment was accepted")


def test_gradient_file_is_checked_and_held_out_is_not_read(tmp_path: Path):
    mps = _record("mps")
    cuda = _record("cuda")
    raw = __import__("struct").pack("<3f", 1.0, -1.0, 0.0)
    digest = __import__("hashlib").sha256(raw).hexdigest()
    for record, folder in ((mps, tmp_path / "mps"), (cuda, tmp_path / "cuda")):
        folder.mkdir()
        (folder / "clean.f32").write_bytes(raw)
        record["gradient"] = {"f32_file": "clean.f32", "sha256": digest}
        record["gradient"].pop("values", None)
    report = compare_records(mps, cuda, mps_dir=tmp_path / "mps", cuda_dir=tmp_path / "cuda")
    assert report["status"] == "differences_recorded"
    assert report["gradients"]["cosine"] != 1.0
    assert report["gradients"]["cosine"] > 0.999
    assert report["devices_interchangeable"] is False
    banned = tmp_path / "sealed"
    banned.mkdir()
    (banned / "clean.f32").write_bytes(raw)
    cuda["gradient"]["f32_file"] = "../sealed/clean.f32"
    try:
        compare_records(mps, cuda, mps_dir=tmp_path / "mps", cuda_dir=tmp_path / "cuda")
    except ValueError as exc:
        assert "held-out" in str(exc)
    else:
        raise AssertionError("a sealed gradient was read")


def test_compare_command_exit_is_not_numeric_agreement(tmp_path: Path, capsys):
    from cuda_validation.__main__ import main

    mps_path = tmp_path / "mps.json"
    cuda_path = tmp_path / "cuda.json"
    mps_path.write_text(json.dumps(_record("mps")), encoding="utf-8")
    cuda_path.write_text(json.dumps(_record("cuda")), encoding="utf-8")
    main(["compare", str(mps_path), str(cuda_path)])
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "differences_recorded"
    assert report["devices_interchangeable"] is False
    assert STORED_HALF_LEVEL > 0
