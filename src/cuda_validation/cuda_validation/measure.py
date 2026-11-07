"""Run the frozen 3-step attack on one development scene.

The command does not call the RunPod API, does not walk the 96-scene manifest
for images, and does not call ``prepare_environment`` or ``select_dtype``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import struct
from pathlib import Path
from typing import Any

from .bounds import check_bounds
from .gate import approve
from .hardware import (
    collect_torch_hardware,
    driver_at_least_minimum,
    is_quoted_l4,
    read_nvidia_driver,
)
from .pins import (
    ATTN_IMPLEMENTATION,
    BF16_REQUIRED_AVAILABLE_BYTES,
    COMPUTE_CAPABILITY,
    DTYPE,
    EPSILON,
    EPSILON_LABEL,
    EVAL_CONTEXTS,
    FLOAT32_REQUIRED_AVAILABLE_BYTES,
    INIT,
    MODEL_ID,
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
from .trajectory import run_three_steps

PREDICTIONS = Path("docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl")
MANIFEST = Path("fixtures/development_v4/manifest.jsonl")
CHECKSUMS = Path("fixtures/development_v4/checksums.json")


def measure(
    root: Path,
    *,
    device_name: str,
    output: Path | None = None,
    os_available_bytes: int | None = None,
) -> dict[str, Any]:
    """Score ``dev4-ref-00-a`` on ``mps`` or ``cuda``, or write a refusal."""
    available = os_available_bytes
    if available is None and str(device_name).lower() in {"mps", "cuda"}:
        available = _read_available_bytes()
    decision = approve(
        {
            "device": device_name,
            "scene_id": SCENE_ID,
            "scene_count": SCENE_COUNT,
            "image_relative": SCENE_IMAGE_RELATIVE,
            "output_path": None if output is None else str(output),
            "os_available_bytes": available,
        }
    )
    if decision["approved"] is not True:
        return _emit(_base(status="refused", device=str(device_name).lower(), extra=decision), output)
    try:
        payload = _execute(root, device_name=str(device_name).lower(), available=int(available), output=output)
    except Exception as exc:
        payload = _base(
            status="error",
            device=str(device_name).lower(),
            extra={
                "error": f"{type(exc).__name__}: {exc}",
                "model_from_pretrained_called": None,
                "model_from_pretrained_called_null_reason": (
                    "The command failed after the refusal checks. The flag was not invented."
                ),
                "attacks_run": False,
            },
        )
    return _emit(payload, output)


def _execute(root: Path, *, device_name: str, available: int, output: Path | None) -> dict[str, Any]:
    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        return _stop("stopped_device", device_name, "CUDA was requested but torch.cuda.is_available() is false.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        return _stop("stopped_device", device_name, "MPS was requested but torch.backends.mps.is_available() is false.")

    driver = platform.platform()
    if device.type == "cuda":
        driver = read_nvidia_driver()
        if not isinstance(driver, str) or not driver_at_least_minimum(driver):
            return _stop(
                "stopped_driver",
                device_name,
                "The NVIDIA driver was not read as 580 or newer. CPU was not used instead.",
            )
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        capability = f"{major}.{minor}"
        if not is_quoted_l4(name) or capability != COMPUTE_CAPABILITY:
            return _stop(
                "stopped_device",
                device_name,
                f"The visible GPU is {name} at capability {capability}. The quoted card is an L4 at 8.9.",
            )

    budget = _budget_module(root)
    bf16, float32 = _lines(budget)
    if bf16["calibrated_projection"] != BF16_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError("The bfloat16 line moved. Refusing to adopt the new number.")
    if float32["calibrated_projection"] != FLOAT32_REQUIRED_AVAILABLE_BYTES:
        raise RuntimeError("The float32 line moved. Refusing to adopt the new number.")
    if any(available < need for need in bf16.values()):
        return _stop(
            "stopped_budget",
            device_name,
            f"OS-available bytes {available} miss a bfloat16 line. The guard was not lowered.",
            extra={"os_available_bytes": available, "lines": {key: int(value) for key, value in bf16.items()}},
        )

    weights = _weight_file()
    if weights is None or not weights.is_file():
        return _stop(
            "stopped_missing_weights",
            device_name,
            "The pinned safetensors file is not in HF_HOME. This command does not download it.",
        )
    from qwen_rgb.measure import read_local_header

    header = read_local_header(weights)
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS:
        return _stop("stopped_unexpected_header", device_name, "The header parameter count changed.")
    if header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
        return _stop("stopped_unexpected_header", device_name, "The header float32 size changed.")
    if header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        return _stop("stopped_unexpected_header", device_name, "The header is not exclusively BF16.")

    row = _scene_row(root)
    image_path = root / "fixtures" / "development_v4" / row["image_path"]
    if _sha256(image_path) != _expected_checksum(root, row["image_path"]):
        return _stop("stopped_checksum", device_name, "The development image checksum does not match.")
    texts = _user_texts(row)
    hashes = _prediction_hashes(root)
    for context, user_text in texts.items():
        if hashes.get(context) != hashlib.sha256(user_text.encode("utf-8")).hexdigest():
            return _stop("stopped_prompt", device_name, f"{context} prompt hash does not match Checkpoint 1.7.")

    from qwen_rgb.measure import bf16_probe, load_model_after_budget, load_processor
    from qwen_rgb.rgb_path import assert_pinned_processor
    from dev_pilot.scoring import assert_locked_tokenization

    processor, image_processor = load_processor()
    assert_pinned_processor(image_processor)
    identity = assert_locked_tokenization(processor, "qwen3-vl-2b")
    token_ids = {color: tuple(int(token) for token in ids) for color, ids in identity["token_ids"].items()}
    if any(len(ids) != 1 for ids in token_ids.values()):
        return _stop("stopped_tokenization", device_name, "A locked color continuation is not one token.")

    if device.type == "mps":
        mps_probe = bf16_probe(torch)
        cuda_probe = {"ran": False, "reason": "This is the MPS measurement. A CUDA matmul was not run."}
        if mps_probe.get("matmul_ok") is not True:
            return _stop("stopped_mps_probe", device_name, "The MPS BF16 probe did not run. from_pretrained was not called.")
    else:
        mps_probe = {"ran": False, "is_mps_probe": False, "reason": "The MPS probe was not run and was not relabeled."}
        cuda_probe = _cuda_bf16_probe(torch)
        if cuda_probe.get("ok") is not True:
            return _stop("stopped_cuda_probe", device_name, "The CUDA BF16 probe did not run. A CPU matmul was not run.")

    called = False
    try:
        called = True
        model = load_model_after_budget({"permitted": True}, DTYPE)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    except Exception as exc:
        return _stop(
            "error",
            device_name,
            f"{type(exc).__name__}: {exc}",
            extra={"model_from_pretrained_called": called},
        )
    parameter = next(model.parameters())
    commit = getattr(model.config, "_commit_hash", None)
    if parameter.device.type != device.type or str(parameter.dtype) != "torch.bfloat16":
        return _stop(
            "stopped_unexpected_load",
            device_name,
            "The loaded parameters are not bfloat16 on the requested device.",
            extra={"model_from_pretrained_called": True},
        )
    if commit not in (None, REVISION):
        return _stop(
            "stopped_revision",
            device_name,
            f"Loaded commit {commit} is not {REVISION}.",
            extra={"model_from_pretrained_called": True},
        )

    from attack_feasibility.score import (
        MemoryStop,
        backward_at_image,
        geometry,
        is_memory_error,
        prepare_prompt,
        release_cache,
        score_differentiable,
        score_official,
    )
    from qwen_rgb.constraints import linf_distance
    from vlm_iface.images import to_image_01

    hardware = collect_torch_hardware(
        device,
        driver_version=driver,
        parameter_dtype=str(parameter.dtype),
        attn=ATTN_IMPLEMENTATION,
    )
    hardware["os_available_bytes"] = available
    image = to_image_01(image_path)
    prompts = {
        context: prepare_prompt(processor, image_processor, image, texts[context]) for context in EVAL_CONTEXTS
    }
    label = str(row["answer"])
    try:
        clean_float, gradient = backward_at_image(
            model=model,
            image_processor=image_processor,
            prompt=prompts[SOURCE_CONTEXT],
            image=image,
            token_ids=token_ids,
            label=label,
            device=device,
        )
        if clean_float.get("loss_identity_agreed") is not True or gradient is None:
            return _stop(
                "stopped_loss_identity",
                device_name,
                "The labeled-color NLL formulas disagreed, so PGD was not started.",
                extra={"model_from_pretrained_called": True, "hardware": hardware},
            )
        if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) == 0.0:
            return _stop(
                "stopped_loss_identity",
                device_name,
                "The clean gradient is missing, non-finite, or zero.",
                extra={"model_from_pretrained_called": True, "hardware": hardware},
            )

        holder = {"index": 0, "trajectory": [clean_float["loss"]]}

        def gradient_at(current):
            if current is None:
                raise RuntimeError("a PGD step was given no image.")
            if holder["index"] == 0:
                holder["index"] = 1
                return gradient
            stepped, step_gradient = backward_at_image(
                model=model,
                image_processor=image_processor,
                prompt=prompts[SOURCE_CONTEXT],
                image=current,
                token_ids=token_ids,
                label=label,
                device=device,
            )
            if stepped.get("loss_identity_agreed") is not True or step_gradient is None:
                raise RuntimeError("The labeled-color NLL formulas disagreed during PGD.")
            holder["trajectory"].append(stepped["loss"])
            holder["index"] += 1
            return step_gradient

        result = run_three_steps(image, gradient_at)
        per_step = [float(linf_distance(step_image, image)) for step_image in result["images"]]
        final = result["images"][-1]
        final_float = score_differentiable(
            model=model,
            image_processor=image_processor,
            prompt=prompts[SOURCE_CONTEXT],
            image=final,
            token_ids=token_ids,
            label=label,
            device=device,
        )
        holder["trajectory"].append(final_float.get("loss"))
        geo, stored = geometry(image, gradient, final)
        clean_official = {
            context: score_official(
                model=model,
                processor=processor,
                image=image,
                user_text=texts[context],
                token_ids=token_ids,
                label=label,
                device=device,
            )
            for context in EVAL_CONTEXTS
        }
        final_official = score_official(
            model=model,
            processor=processor,
            image=stored,
            user_text=texts[SOURCE_CONTEXT],
            token_ids=token_ids,
            label=label,
            device=device,
        )
    except MemoryStop as exc:
        return _stop(
            "stopped_oom",
            device_name,
            "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
            extra={"model_from_pretrained_called": True, "detail": str(exc), "hardware": hardware},
        )
    except RuntimeError as exc:
        if is_memory_error(exc):
            return _stop(
                "stopped_oom",
                device_name,
                "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
                extra={"model_from_pretrained_called": True, "hardware": hardware},
            )
        if "NLL formulas disagreed" in str(exc):
            return _stop(
                "stopped_loss_identity",
                device_name,
                str(exc),
                extra={"model_from_pretrained_called": True, "hardware": hardware},
            )
        raise
    finally:
        release_cache(device)

    gradient_path = None if output is None else output.with_suffix(".gradient.f32")
    gradient_record = _gradient_record(gradient, gradient_path)
    projected_ok = bool(float(final.min()) >= -1e-6 and float(final.max()) <= 1 + 1e-6)
    stored_ok = bool(float(stored.min()) >= -1e-6 and float(stored.max()) <= 1 + 1e-6)
    payload = _base(status="completed", device=device_name)
    payload.update(
        {
            "model_from_pretrained_called": True,
            "attacks_run": True,
            "attack_steps_completed": STEPS,
            "scene_id": SCENE_ID,
            "revision": REVISION,
            "model_id": MODEL_ID,
            "dtype": DTYPE,
            "attn_implementation": ATTN_IMPLEMENTATION,
            "epsilon": EPSILON_LABEL,
            "epsilon_float": EPSILON,
            "steps": STEPS,
            "restarts": RESTARTS,
            "init": INIT,
            "step_size_label": STEP_SIZE_LABEL,
            "step_size_float": STEP_SIZE,
            "loss_identity_agreed": True,
            "hardware": hardware,
            "mps_probe": mps_probe,
            "cuda_probe": cuda_probe,
            "float32_budget_would_pass": all(available >= int(value) for value in float32.values()),
            "float32_loaded": False,
            "outputs": {
                "clean_official": _slice(clean_official[SOURCE_CONTEXT]),
                "clean_differentiable": _slice(clean_float),
                "final_official": _slice(final_official),
                "helpful_clean_official": _slice(clean_official["helpful_nonrevealing"]),
            },
            "losses": {"label": label, "trajectory": holder["trajectory"]},
            "gradient": gradient_record,
            "bounds": {
                "epsilon": EPSILON_LABEL,
                "epsilon_float": EPSILON,
                "steps": STEPS,
                "restarts": RESTARTS,
                "init": INIT,
                "step_size_label": STEP_SIZE_LABEL,
                "step_size_float": STEP_SIZE,
                "linf_float": geo["linf"],
                "linf_stored_vs_clean": geo["uint8_geometry"]["linf_vs_clean"],
                "linf_stored_vs_projected": geo["uint8_geometry"]["linf_vs_projected"],
                "per_step_linf_float": per_step,
                "pixels_in_unit_interval": projected_ok and stored_ok,
                "min_pixel": float(min(float(final.min()), float(stored.min()))),
                "max_pixel": float(max(float(final.max()), float(stored.max()))),
            },
            "memory": {
                "os_available_bytes": available,
                "bf16_required_available_bytes": BF16_REQUIRED_AVAILABLE_BYTES,
                "float32_required_available_bytes": FLOAT32_REQUIRED_AVAILABLE_BYTES,
                "os_floor_bytes": OS_FLOOR_BYTES,
                "accelerator": hardware["accelerator_memory"],
                "pools_are_distinct": True,
            },
        }
    )
    bound_view = dict(payload["bounds"])
    bound_view["device_type"] = device_name
    payload["bound_check"] = check_bounds(bound_view)
    return payload


def _cuda_bf16_probe(torch) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "cuda_bf16_matmul",
        "is_mps_probe": False,
        "is_model_forward": False,
        "ok": False,
    }
    if not torch.cuda.is_available():
        report["error"] = "CUDA is not available. A CPU matmul was not run."
        return report
    ones = torch.zeros(4, device="cuda", dtype=torch.bfloat16)
    added = (ones + torch.tensor(1, device="cuda", dtype=torch.bfloat16)).float().cpu()
    report["add_ok"] = bool(torch.isfinite(added).all()) and added.tolist() == [1.0, 1.0, 1.0, 1.0]
    matrix = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    report["ok"] = (
        report["add_ok"] is True
        and bool(torch.isfinite(product.float()).all())
        and str(product.dtype) == "torch.bfloat16"
    )
    del ones, added, matrix, product
    torch.cuda.empty_cache()
    return report


def _gradient_record(gradient, path: Path | None) -> dict[str, Any]:
    import torch

    array = gradient.detach().to(dtype=torch.float32, device="cpu").contiguous()
    try:
        raw = array.numpy().tobytes()
    except Exception:
        flat = array.reshape(-1).tolist()
        raw = struct.pack(f"<{len(flat)}f", *flat)
    record = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
        "dtype": "float32",
        "finite": bool(torch.isfinite(array).all()),
        "nonzero": bool(float(array.abs().sum()) != 0.0),
        "l1": float(array.abs().sum()),
        "l2": float(torch.linalg.vector_norm(array.reshape(-1).double())),
    }
    if path is not None:
        path.write_bytes(raw)
        record["f32_file"] = path.name
    return record


def _slice(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "summed_nll": score.get("summed_nll"),
        "chosen": score.get("chosen"),
        "loss": score.get("loss"),
        "loss_identity_agreed": score.get("loss_identity_agreed"),
        "loss_identity_max_abs": score.get("loss_identity_max_abs"),
    }


def _user_texts(row: dict[str, Any]) -> dict[str, str]:
    from choice_eval.prompt import build_prompt

    return {
        context: build_prompt(row["contexts"][context]["text"], row["question"]).user_text()
        for context in EVAL_CONTEXTS
    }


def _scene_row(root: Path) -> dict[str, Any]:
    match = None
    with (root / MANIFEST).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene_id") != SCENE_ID:
                continue
            if match is not None:
                raise RuntimeError(f"{SCENE_ID} appears more than once.")
            match = row
    if match is None:
        raise RuntimeError(f"{SCENE_ID} is not in the development manifest.")
    if match.get("split") != "development":
        raise RuntimeError("The pinned scene is not marked development.")
    if match.get("image_path") != "images/dev4-ref-00-a.png":
        raise RuntimeError("The pinned scene image path changed.")
    return match


def _prediction_hashes(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    with (root / PREDICTIONS).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene_id") != SCENE_ID or row.get("context") not in EVAL_CONTEXTS:
                continue
            found[str(row["context"])] = str(row["user_text_sha256"])
    if set(found) != set(EVAL_CONTEXTS):
        raise RuntimeError("Checkpoint 1.7 is missing a prompt hash for the validation scene.")
    return found


def _expected_checksum(root: Path, image_path: str) -> str:
    checksums = json.loads((root / CHECKSUMS).read_text(encoding="utf-8"))
    return str(checksums["files"][image_path]["sha256"])


def _weight_file() -> Path | None:
    import os

    home = os.environ.get("HF_HOME")
    if not home:
        return None
    return (
        Path(home)
        / "hub"
        / "models--Qwen--Qwen3-VL-2B-Instruct"
        / "snapshots"
        / REVISION
        / "model.safetensors"
    )


def _budget_module(root: Path):
    import importlib.util

    path = root / "docs" / "checkpoint-1.5" / "run_model_feasibility.py"
    spec = importlib.util.spec_from_file_location("checkpoint15_budget_lines_measure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lines(budget) -> tuple[dict[str, int], dict[str, int]]:
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
    return bf16, float32


def _read_available_bytes() -> int:
    from vlm_iface.devices import available_bytes

    return int(available_bytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop(status: str, device: str, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _base(status=status, device=device, extra={"reason": reason, "attacks_run": False})
    if extra:
        payload.update(extra)
    payload.setdefault("model_from_pretrained_called", False)
    return payload


def _base(status: str, device: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "cuda_validation.scene_record.v1",
        "label": "DEVELOPMENT ONLY",
        "status": status,
        "device_type": device,
        "scene_id": SCENE_ID,
        "scene_count": SCENE_COUNT,
        "runpod_api_called": False,
        "pod_launched_by_this_command": False,
        "guard_lowered": False,
        "held_out_scenes_opened": False,
        "full_development_attack_run": False,
        "h_star_fitted": False,
        "select_dtype_called": False,
        "prepare_environment_called": False,
        "model_from_pretrained_called": False,
        "attacks_run": False,
        "random_seed_used": False,
    }
    if extra:
        payload.update(extra)
    return payload


def _emit(payload: dict[str, Any], output: Path | None) -> dict[str, Any]:
    if output is None:
        return payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload
