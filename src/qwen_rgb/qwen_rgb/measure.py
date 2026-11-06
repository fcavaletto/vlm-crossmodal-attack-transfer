"""One-scene RGB gradient of the pinned Qwen3-VL-2B candidate.

The pre-load budget is the Checkpoint 1.5 function. A failing recheck returns
before ``Qwen3VLForConditionalGeneration.from_pretrained``. The run scores one
development scene and one context. It does not generate, and it does not take
an attack step.

Run from the worktree root:

    HF_HOME=/Users/fedecava/Development/models/huggingface \\
    PYTHONPATH=src/qwen_rgb:src/vlm_iface:src/pilot \\
    /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb.measure
"""

from __future__ import annotations

import gc
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

from .budget import dtype_decisions, load_budget_module, repo_root_from, select_dtype
from .compare import allows_model_load, compare_image
from .finite_difference import (
    DIRECTION_SEED,
    PRIMARY_DELTAS,
    analytic_directional,
    ceiling_for_dtype,
    central_difference,
    judge_samples,
    rademacher,
)
from .loss import chosen_answer_nll, extend_with_answer
from .pins import CONTEXT_KEY, MODEL_ID, PINNED_REVISION, REQUIRED_HF_HOME, SCENE_ID
from .rgb_path import assert_pinned_processor, differentiable_model_pixels

REPRODUCTION_COMMAND = (
    "HF_HOME=/Users/fedecava/Development/models/huggingface "
    "PYTHONPATH=src/qwen_rgb:src/vlm_iface:src/pilot "
    "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb.measure"
)
STARTED_FROM_COMMIT = "004783b5604c34aa066befbcd5356bfd7bd6e7b2"


def prepare_environment() -> dict[str, Any]:
    """Use the study Hub cache. Refuse a different HF_HOME. Do not set offline mode."""
    offline_was_set = "HF_HUB_OFFLINE" in os.environ
    if offline_was_set:
        os.environ.pop("HF_HUB_OFFLINE", None)
    home = os.environ.get("HF_HOME")
    assigned = False
    if home is None:
        os.environ["HF_HOME"] = REQUIRED_HF_HOME
        home = REQUIRED_HF_HOME
        assigned = True
    if Path(home).resolve() != Path(REQUIRED_HF_HOME).resolve():
        raise SystemExit(f"HF_HOME is {home}, expected {REQUIRED_HF_HOME}. Refusing to use another cache.")
    return {
        "HF_HOME": home,
        "HF_HOME_assigned_by_script": assigned,
        "HF_HUB_OFFLINE_was_set": offline_was_set,
        "HF_HUB_OFFLINE_set_by_script": False,
    }


def output_path(repo_root: Path) -> Path:
    return repo_root / "src" / "qwen_rgb" / "records" / "development_rgb_gradient.json"


def weight_file() -> Path:
    return (
        Path(os.environ["HF_HOME"])
        / "hub"
        / "models--Qwen--Qwen3-VL-2B-Instruct"
        / "snapshots"
        / PINNED_REVISION
        / "model.safetensors"
    )


def read_local_header(path: Path) -> dict[str, Any]:
    """Parse the safetensors header. The tensor bytes stay on disk."""
    from vlm_iface.candidate import parse_safetensors_prefix

    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise RuntimeError("The weight file is shorter than a safetensors header.")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or header_len > 10_000_000:
            raise RuntimeError(f"Refusing safetensors header length {header_len}.")
        rest = handle.read(header_len)
    if len(rest) != header_len:
        raise RuntimeError("The weight file ended inside the header. Tensor bytes were not read.")
    summary = parse_safetensors_prefix(prefix + rest)
    summary["approximate"] = False
    summary["header_source"] = "local_file_prefix"
    summary["weights_loaded"] = False
    summary["file_bytes"] = path.stat().st_size
    return summary


def load_scene(repo_root: Path) -> tuple[dict[str, Any], Path]:
    """Read the one pinned development scene. Other manifest rows are not opened."""
    manifest = repo_root / "fixtures" / "development_v2" / "manifest.jsonl"
    match: dict[str, Any] | None = None
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scene_id") != SCENE_ID:
                continue
            if match is not None:
                raise RuntimeError(f"{SCENE_ID} appears more than once")
            match = row
    if match is None:
        raise RuntimeError(f"{SCENE_ID} is not in the development manifest")
    if match.get("split") != "development":
        raise RuntimeError("The pinned scene is not marked development.")
    if match.get("task_family") != "attribute":
        raise RuntimeError("The pinned scene is not an attribute scene.")
    if match.get("answer") not in {"red", "blue", "green", "yellow"}:
        raise RuntimeError("The chosen answer is not one of the four color classes.")
    image_path = repo_root / "fixtures" / "development_v2" / match["image_path"]
    if not image_path.is_file():
        raise RuntimeError(f"Missing development image {image_path}")
    return match, image_path


def bf16_probe(torch) -> dict[str, Any]:
    """One small MPS BF16 matmul. This does not load Qwen."""
    probe: dict[str, Any] = {
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "add_ok": False,
        "matmul_ok": False,
        "is_model_forward": False,
    }
    if not probe["mps_available"]:
        probe["error"] = "MPS is not available. A CPU matmul was not run."
        return probe
    ones = torch.zeros(4, device="mps", dtype=torch.bfloat16)
    added = (ones + torch.tensor(1, device="mps", dtype=torch.bfloat16)).float().cpu()
    probe["add_ok"] = bool(torch.isfinite(added).all()) and added.tolist() == [1.0, 1.0, 1.0, 1.0]
    matrix = torch.randn(8, 8, device="mps", dtype=torch.bfloat16)
    product = matrix @ matrix
    probe["matmul_ok"] = bool(torch.isfinite(product.float()).all()) and str(product.dtype) == "torch.bfloat16"
    del ones, added, matrix, product
    torch.mps.empty_cache()
    return probe


def load_processor():
    """Load the processor config and tokenizer. This does not load model weights."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=PINNED_REVISION)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("The loaded processor has no image processor.")
    assert_pinned_processor(image_processor)
    return processor, image_processor


def load_model_after_budget(recheck: dict[str, Any], dtype_name: str):
    """The only model ``from_pretrained`` call. The budget recheck is the permit."""
    if not recheck.get("permitted"):
        raise RuntimeError("from_pretrained refused: the budget recheck did not permit a load.")
    if dtype_name not in {"float32", "bfloat16"}:
        raise RuntimeError("from_pretrained refused: no dtype was selected.")
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    torch_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    return Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        dtype=torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )


def encode_chat(processor, pil_image, prompt: str) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(encoded)


def move_to_device(inputs: dict[str, Any], device) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _preprocessor_sum_gradient(image, image_processor) -> dict[str, Any]:
    """Backward of the sum of ``pixel_values``. This is not the chosen-answer loss."""
    import torch

    leaf = image.detach().to(dtype=torch.float32, device="cpu").requires_grad_(True)
    pixels, _ = differentiable_model_pixels(leaf, image_processor)
    pixels.sum().backward()
    stats = gradient_stats(leaf.grad)
    stats["loss"] = "sum_of_pixel_values"
    stats["is_chosen_answer_loss"] = False
    return stats


def gradient_stats(gradient) -> dict[str, Any]:
    import torch

    if gradient is None:
        return {"computed": False, "finite": None, "nonzero": None}
    values = gradient.detach().float().cpu()
    finite = bool(torch.isfinite(values).all())
    return {
        "computed": True,
        "finite": finite,
        "nonzero": bool((values != 0).any()) if finite else None,
        "abs_mean": float(values.abs().mean()) if finite else None,
        "abs_max": float(values.abs().max()) if finite else None,
        "abs_sum": float(values.abs().sum()) if finite else None,
        "fraction_nonzero": float((values != 0).float().mean()) if finite else None,
        "shape": list(gradient.shape),
        "dtype": str(gradient.dtype),
    }


def _os_memory() -> dict[str, int]:
    import psutil

    memory = psutil.virtual_memory()
    return {
        "total": int(memory.total),
        "available": int(memory.available),
        "used": int(memory.used),
        "free": int(memory.free),
    }


def _machine() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    if platform.system() == "Darwin":
        summary["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        summary["hw_memsize"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    return summary


def _versions() -> dict[str, str]:
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _empty_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "measurement_kind": "checkpoint17_qwen_rgb_gradient",
        "label": "DEVELOPMENT ONLY",
        "checkpoint": "checkpoint-1.7",
        "started_from_commit": STARTED_FROM_COMMIT,
        "scene_id": SCENE_ID,
        "context": CONTEXT_KEY,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "attacks_run": False,
        "generation_run": False,
        "training_run": False,
        "cloud_gpu": False,
        "held_out_opened": False,
        "h_star_fitted": False,
        "model_from_pretrained_called": False,
        "model_load_started": False,
        "rgb_gradient_finite": None,
        "rgb_gradient_nonzero": None,
        "reproduction_command": REPRODUCTION_COMMAND,
        "repo_root": str(repo_root),
    }


def run(repo_root: Path | None = None) -> dict[str, Any]:
    """Execute the guarded measurement and write the JSON record."""
    root = repo_root if repo_root is not None else repo_root_from(Path(__file__).resolve())
    path = output_path(root)
    payload = _empty_payload(root)
    payload["environment"] = prepare_environment()
    payload["machine"] = _machine()
    try:
        _execute(root, payload)
    except Exception as exc:
        text = str(exc).lower()
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        if payload.get("model_load_started") and "from_pretrained refused" not in str(exc):
            payload["model_from_pretrained_called"] = True
        if "out of memory" in text:
            payload["status"] = "out_of_memory"
            payload["stopped_reason"] = (
                "The runtime reported out of memory. The budget was not lowered and the load was not retried."
            )
            _write(path, payload)
            return payload
        payload["status"] = "error"
        _write(path, payload)
        raise
    _write(path, payload)
    return payload


def _execute(root: Path, payload: dict[str, Any]) -> None:
    import torch
    from vlm_iface.devices import memory_snapshot, synchronize
    from vlm_iface.images import image_01_to_pil, to_image_01
    from vlm_iface.tokens import answer_token_ids
    from pilot.prompt import user_prompt

    budget = load_budget_module(root)
    payload["versions"] = _versions()
    payload["budget_constants"] = {
        "spare_required_bytes": budget.SPARE_REQUIRED_BYTES,
        "projected_remaining_floor_bytes": budget.PROJECTED_REMAINING_FLOOR_BYTES,
        "os_floor_bytes": 6 * 1024**3,
        "source": "docs/checkpoint-1.5/run_model_feasibility.py",
    }
    memory_at_start = _os_memory()
    payload["memory_at_start"] = memory_at_start
    probe = bf16_probe(torch)
    payload["bf16_probe"] = probe

    weights = weight_file()
    payload["weight_file_present"] = weights.is_file()
    if not weights.is_file():
        payload["status"] = "stopped_missing_weights"
        payload["stopped_reason"] = "The local safetensors file is absent. This script does not download it."
        payload["model_from_pretrained_called"] = False
        return
    header = read_local_header(weights)
    payload["header"] = {
        "parameter_count": header["parameter_count"],
        "dtype_counts": header["dtype_counts"],
        "float32_weight_bytes": header["float32_weight_bytes"],
        "file_bytes": header["file_bytes"],
        "header_bytes": header["header_bytes"],
        "approximate": header["approximate"],
        "weights_loaded": False,
    }
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS:
        payload["status"] = "stopped_unexpected_header"
        payload["stopped_reason"] = "The header parameter count is not the Checkpoint 1.5 Qwen count."
        return
    if header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
        payload["status"] = "stopped_unexpected_header"
        payload["stopped_reason"] = "The header float32 expansion is not the Checkpoint 1.5 Qwen size."
        return
    if header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        payload["status"] = "stopped_unexpected_header"
        payload["stopped_reason"] = "The header is not exclusively BF16. The BF16 resident size would be a guess."
        return

    decisions = dtype_decisions(
        budget,
        parameter_count=header["parameter_count"],
        float32_weight_bytes=header["float32_weight_bytes"],
        available_bytes=memory_at_start["available"],
        mps_available=bool(probe["mps_available"]),
    )
    selection = select_dtype(decisions, bf16_probe_ok=bool(probe.get("matmul_ok")))
    payload["dtype_decisions_at_start"] = decisions
    payload["dtype_selection_at_start"] = selection

    print("Loading the processor. Model weights stay on disk.", flush=True)
    processor, image_processor = load_processor()
    payload["processor_class"] = type(processor).__name__
    payload["image_processor_class"] = type(image_processor).__name__
    scene, image_path = load_scene(root)
    prompt = user_prompt(scene["contexts"][CONTEXT_KEY]["text"], scene["question"])
    image = to_image_01(image_path)
    if float(image.min()) < 0 or float(image.max()) > 1:
        raise RuntimeError("The canonical scene image is outside [0, 1].")
    comparison = compare_image(image, image_processor)
    payload["processor_comparison"] = comparison
    payload["preprocessor_only_sum_gradient"] = _preprocessor_sum_gradient(image, image_processor)
    payload["scene"] = {
        "scene_id": scene["scene_id"],
        "task_family": scene["task_family"],
        "answer": scene["answer"],
        "question": scene["question"],
        "context_key": CONTEXT_KEY,
        "context_text": scene["contexts"][CONTEXT_KEY]["text"],
        "prompt": prompt,
        "image_path": str(Path("fixtures") / "development_v2" / scene["image_path"]),
        "image_shape": list(image.shape),
    }
    tokenizer = processor.tokenizer
    answer_ids = answer_token_ids(tokenizer, scene["answer"], leading_space=True)
    bare_ids = answer_token_ids(tokenizer, scene["answer"], leading_space=False)
    payload["answer_tokens"] = {
        "leading_space": True,
        "token_ids": [int(token) for token in answer_ids.tolist()],
        "pieces": [tokenizer.decode([int(token)]) for token in answer_ids.tolist()],
        "bare_token_ids": [int(token) for token in bare_ids.tolist()],
        "bare_pieces": [tokenizer.decode([int(token)]) for token in bare_ids.tolist()],
        "reduction": "summed_nll",
    }
    if not allows_model_load(comparison):
        payload["status"] = "stopped_processor_gap"
        payload["stopped_reason"] = (
            "The differentiable pixels do not match the official PIL processor within "
            "the precommitted absolute tolerance, or the grids differ. The model was not loaded."
        )
        payload["model_from_pretrained_called"] = False
        return

    print("Rechecking the budget immediately before the model load.", flush=True)
    payload["memory_before_releasing_temporary_tensors"] = _os_memory()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    memory_before = _os_memory()
    payload["memory_immediately_before_load"] = memory_before
    recheck = dtype_decisions(
        budget,
        parameter_count=header["parameter_count"],
        float32_weight_bytes=header["float32_weight_bytes"],
        available_bytes=memory_before["available"],
        mps_available=bool(torch.backends.mps.is_available()),
    )
    reselection = select_dtype(recheck, bf16_probe_ok=bool(probe.get("matmul_ok")))
    payload["budget_recheck_immediately_before_load"] = recheck
    payload["dtype_selection_immediately_before_load"] = reselection
    if not reselection["permitted"] or reselection["dtype"] is None:
        payload["status"] = "stopped_budget"
        payload["stopped_reason"] = reselection["reason"]
        payload["model_from_pretrained_called"] = False
        payload["rgb_gradient_finite"] = None
        return

    dtype_name = str(reselection["dtype"])
    device = torch.device("mps")
    payload["model_load_started"] = True
    _write(output_path(root), payload)
    started = time.perf_counter()
    model = load_model_after_budget(recheck[dtype_name], dtype_name)
    payload["model_from_pretrained_called"] = True
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    synchronize(device)
    payload["load_seconds"] = round(time.perf_counter() - started, 3)
    parameter = next(model.parameters())
    dtype_counts: dict[str, int] = {}
    for item in model.parameters():
        dtype_counts[str(item.dtype)] = dtype_counts.get(str(item.dtype), 0) + item.numel()
    payload["loaded_model"] = {
        "device": parameter.device.type,
        "dtype": str(parameter.dtype),
        "dtype_counts": dtype_counts,
        "parameters_require_grad": any(item.requires_grad for item in model.parameters()),
        "memory_after_load": memory_snapshot(device),
    }
    if parameter.device.type != "mps" or str(parameter.dtype) != f"torch.{dtype_name}":
        payload["status"] = "stopped_unexpected_load"
        payload["stopped_reason"] = "The loaded parameters are not the permitted dtype on MPS."
        return

    _gradient_and_check(
        payload,
        model=model,
        processor=processor,
        image_processor=image_processor,
        image=image,
        prompt=prompt,
        answer_ids=answer_ids,
        device=device,
        dtype_name=dtype_name,
    )


def _gradient_and_check(
    payload,
    *,
    model,
    processor,
    image_processor,
    image,
    prompt,
    answer_ids,
    device,
    dtype_name: str,
) -> None:
    import torch
    from vlm_iface.devices import memory_snapshot, synchronize
    from vlm_iface.images import image_01_to_pil

    leaf = image.detach().to(dtype=torch.float32, device="cpu").requires_grad_(True)
    pixels, grid = differentiable_model_pixels(leaf, image_processor)
    pil = image_01_to_pil(leaf.detach())
    encoded = encode_chat(processor, pil, prompt)
    official_grid = encoded["image_grid_thw"].cpu()
    if not torch.equal(grid, official_grid):
        payload["status"] = "stopped_grid_mismatch"
        payload["stopped_reason"] = "The chat-template grid does not match the differentiable grid."
        return
    prompt_ids = encoded["input_ids"]
    encoded.pop("pixel_values", None)
    extended, prompt_length = extend_with_answer(encoded, answer_ids)
    moved = move_to_device(extended, device)
    pixels_mps = pixels.to(device)
    pixels_mps.retain_grad()
    moved["pixel_values"] = pixels_mps
    if not torch.equal(moved["image_grid_thw"].cpu(), grid):
        payload["status"] = "stopped_grid_mismatch"
        payload["stopped_reason"] = "Moving the grid changed its values."
        return

    synchronize(device)
    print("Running the chosen-answer backward on one scene.", flush=True)
    started = time.perf_counter()
    outputs = model(**moved, use_cache=False)
    loss = chosen_answer_nll(outputs.logits, moved["input_ids"], prompt_length)
    finite_loss = bool(torch.isfinite(loss.detach().float()).all())
    payload["chosen_answer_loss"] = {
        "value": None if not finite_loss else float(loss.detach().float().cpu()),
        "finite": finite_loss,
        "prompt_length": prompt_length,
        "answer_length": int(answer_ids.numel()),
        "reduction": "summed_nll",
        "sequence_length": int(moved["input_ids"].shape[-1]),
    }
    if not finite_loss:
        payload["status"] = "nonfinite_loss"
        payload["rgb_gradient_finite"] = None
        payload["stopped_reason"] = "The chosen-answer loss was not finite. Backward was not run."
        return
    loss.backward()
    synchronize(device)
    payload["backward_seconds"] = round(time.perf_counter() - started, 3)
    rgb_stats = gradient_stats(leaf.grad)
    pixel_stats = gradient_stats(pixels_mps.grad)
    payload["rgb_gradient"] = rgb_stats
    payload["pixel_values_gradient"] = pixel_stats
    payload["rgb_gradient_finite"] = rgb_stats["finite"]
    payload["rgb_gradient_nonzero"] = rgb_stats["nonzero"]
    payload["memory_after_backward"] = memory_snapshot(device)
    analytic_gradient = None if leaf.grad is None else leaf.grad.detach().cpu().clone()
    del outputs, loss, pixels_mps, moved
    leaf.grad = None
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    if analytic_gradient is None or not rgb_stats["finite"]:
        payload["status"] = "completed_gradient"
        payload["finite_difference"] = {"status": "not_run", "reason": "The RGB gradient was missing or not finite."}
        return
    _finite_difference(
        payload,
        model=model,
        image_processor=image_processor,
        extended_cpu=extended,
        leaf=leaf.detach(),
        analytic_gradient=analytic_gradient,
        prompt_length=prompt_length,
        device=device,
        dtype_name=dtype_name,
    )
    payload["status"] = "completed"
    payload["memory_at_end"] = memory_snapshot(device)


def _finite_difference(
    payload,
    *,
    model,
    image_processor,
    extended_cpu,
    leaf,
    analytic_gradient,
    prompt_length: int,
    device,
    dtype_name: str,
) -> None:
    import torch
    from vlm_iface.devices import synchronize

    direction = rademacher(tuple(leaf.shape), seed=DIRECTION_SEED)
    base = {key: value for key, value in extended_cpu.items() if key != "pixel_values"}

    def loss_at(image: torch.Tensor) -> float:
        with torch.inference_mode():
            pixels, grid = differentiable_model_pixels(image, image_processor)
            inputs = move_to_device(base, device)
            if not torch.equal(inputs["image_grid_thw"].cpu(), grid):
                raise RuntimeError("A finite-difference step changed the patch grid.")
            inputs["pixel_values"] = pixels.to(device)
            outputs = model(**inputs, use_cache=False)
            loss = chosen_answer_nll(outputs.logits, inputs["input_ids"], prompt_length)
            value = float(loss.detach().float().cpu())
            del outputs, loss, inputs, pixels
            return value

    samples: list[tuple[float, float]] = []
    outside = {}
    started = time.perf_counter()
    for delta in PRIMARY_DELTAS:
        estimate = central_difference(loss_at, leaf, direction, delta)
        samples.append((delta, estimate))
        stepped = leaf + delta * direction
        outside[str(delta)] = float(((stepped < 0) | (stepped > 1)).float().mean())
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    synchronize(device)
    analytic = analytic_directional(analytic_gradient, direction)
    judged = judge_samples(analytic, samples, ceiling=ceiling_for_dtype(dtype_name))
    judged["status"] = "completed"
    judged["direction"] = "rademacher_pm_one"
    judged["direction_seed"] = DIRECTION_SEED
    judged["deltas"] = list(PRIMARY_DELTAS)
    judged["fraction_outside_unit_interval"] = outside
    judged["seconds"] = round(time.perf_counter() - started, 3)
    judged["dtype"] = dtype_name
    judged["note"] = (
        "The direction has entries +1 and -1, so delta is the per-pixel RGB step. "
        "The differentiable path does not clip. A future L-infinity constraint would "
        "make this a different function. The BF16 ceiling is the straight-through "
        "check, not a claim of float32 accuracy."
    )
    payload["finite_difference"] = judged


def main() -> None:
    started = time.perf_counter()
    payload = run()
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    root = repo_root_from(Path(__file__).resolve())
    _write(output_path(root), payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "rgb_gradient_finite": payload.get("rgb_gradient_finite"),
                "rgb_gradient_nonzero": payload.get("rgb_gradient_nonzero"),
                "model_from_pretrained_called": payload.get("model_from_pretrained_called"),
                "stopped_reason": payload.get("stopped_reason"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
