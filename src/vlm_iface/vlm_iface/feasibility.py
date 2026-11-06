"""MPS feasibility measurement for the pinned SmolVLM2 checkpoint.

Run from the repository root:

    python -m vlm_iface.feasibility --device mps --output docs/models/feasibility-smolvlm2-mps.json

The device recorded in the JSON is the resolved torch device. A CPU run is not
an MPS result. This command does not load Qwen weights.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from .candidate import survey_weights
from .devices import available_bytes, memory_snapshot, resolve_device, synchronize
from .pins import MODEL_ID, PINNED_REVISION, QWEN_CANDIDATE_ID
from .preprocess import compare_preprocess, reference_uint8_image
from .session import VLMSession
from .tokens import answer_token_ids

PROMPT = "What color is the central square? Reply with one word."
ANSWER = "red"
MULTI_ANSWER = "red square"
REPRODUCTION_COMMAND = (
    "python -m vlm_iface.feasibility --device mps "
    "--output docs/models/feasibility-smolvlm2-mps.json"
)


def probe_image(kind: str) -> torch.Tensor:
    """Synthetic RGB image. Not drawn from a dataset."""
    image = torch.zeros(3, 128, 128, dtype=torch.float32)
    if kind == "red_square":
        image[:, 32:96, 32:96] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
        return image
    if kind == "blue_square":
        image[:, :, :] = 1.0
        image[:, 16:112, 16:112] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        return image
    raise ValueError(kind)


def gradient_stats(gradient: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(gradient.shape),
        "finite": bool(torch.isfinite(gradient).all()),
        "abs_mean": float(gradient.abs().mean()),
        "abs_max": float(gradient.abs().max()),
        "nonzero_fraction": float((gradient != 0).float().mean()),
        "nonzero": bool((gradient != 0).any()),
    }


def run_measurement(device_name: str) -> dict[str, Any]:
    """Execute the smoke measurement. Raises if the requested device is absent."""
    device = resolve_device(device_name)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "measurement_kind": "checkpoint0_gradient_feasibility",
        "device_requested": device_name,
        "device_resolved": device.type,
        "mps_validation": "not_applicable" if device.type != "mps" else "running",
        "cuda_measurement": False,
        "reproduction_command": REPRODUCTION_COMMAND,
        "model_id": MODEL_ID,
        "pinned_revision": PINNED_REVISION,
        "prompt": PROMPT,
        "answer": ANSWER,
        "multi_token_answer": MULTI_ANSWER,
        "machine": _machine(),
        "versions": _versions(),
        "os_available_bytes_before_load": available_bytes(),
    }
    print("Surveying weight files without downloading tensors.", flush=True)
    payload["qwen_candidate"] = _survey_qwen()
    payload["smolvlm_weights"] = survey_weights(MODEL_ID, revision=PINNED_REVISION)
    if not payload["smolvlm_weights"]["load_permitted_by_weight_guard"]:
        raise RuntimeError("Pinned SmolVLM2 failed the weight-memory guard. It was not loaded.")
    if payload["qwen_candidate"].get("weights_loaded"):
        raise RuntimeError("Qwen weights were loaded. That is not allowed at this checkpoint.")

    print(f"Loading {MODEL_ID} on {device.type}.", flush=True)
    load_started = time.perf_counter()
    session = VLMSession(device=device.type, do_image_splitting=False, max_new_tokens=16, leading_space=True)
    synchronize(device)
    payload["load_seconds"] = round(time.perf_counter() - load_started, 3)
    payload["memory_after_load"] = memory_snapshot(device)
    parameter = next(session.model.parameters())
    payload["model_parameter_device"] = parameter.device.type
    payload["model_parameter_dtype"] = str(parameter.dtype)
    payload["attn_implementation"] = getattr(session.model.config, "_attn_implementation", None)
    if parameter.device.type != device.type:
        raise RuntimeError(
            f"Requested {device.type} but parameters are on {parameter.device.type}."
        )

    gap_image = reference_uint8_image(seed=0, height=96, width=128)
    native_image = reference_uint8_image(seed=0, height=512, width=512)
    payload["preprocess_gap_96x128"] = compare_preprocess(
        session.processor.image_processor, gap_image, do_image_splitting=False
    )
    payload["preprocess_gap_512x512"] = compare_preprocess(
        session.processor.image_processor, native_image, do_image_splitting=False
    )

    details = session.score_details(probe_image("red_square"), PROMPT, MULTI_ANSWER)
    payload["multi_token_score"] = details
    bare = answer_token_ids(session.processor.tokenizer, ANSWER, leading_space=False)
    spaced = answer_token_ids(session.processor.tokenizer, ANSWER, leading_space=True)
    payload["tokenization"] = {
        "bare_ids": bare.tolist(),
        "bare_tokens": session.processor.tokenizer.convert_ids_to_tokens(bare.tolist()),
        "leading_space_ids": spaced.tolist(),
        "leading_space_tokens": session.processor.tokenizer.convert_ids_to_tokens(spaced.tolist()),
        "ids_differ": bare.tolist() != spaced.tolist(),
    }
    saved_flag = session.leading_space
    session.leading_space = False
    bare_logprob = session.answer_logprob(probe_image("red_square"), PROMPT, ANSWER)
    session.leading_space = True
    spaced_logprob = session.answer_logprob(probe_image("red_square"), PROMPT, ANSWER)
    session.leading_space = saved_flag
    payload["leading_space_comparison"] = {
        "image": "red_square",
        "answer": ANSWER,
        "bare_logprob": bare_logprob,
        "leading_space_logprob": spaced_logprob,
        "leading_space_higher": spaced_logprob > bare_logprob,
    }

    print("Computing input-image gradients.", flush=True)
    red = probe_image("red_square")
    blue = probe_image("blue_square")
    first = _timed_gradient(session, device, red, PROMPT, ANSWER)
    repeated = _timed_gradient(session, device, red, PROMPT, ANSWER)
    other = _timed_gradient(session, device, blue, PROMPT, ANSWER)
    payload["gradient_red_square"] = first
    payload["gradient_red_square_repeat"] = repeated
    payload["gradient_blue_square"] = other
    repeat_gap = _max_abs_diff(first["gradient"], repeated["gradient"])
    image_gap = _max_abs_diff(first["gradient"], other["gradient"])
    payload["gradient_comparison"] = {
        "repeat_max_abs_diff": repeat_gap,
        "image_max_abs_diff": image_gap,
    }
    payload["memory_after_gradients"] = memory_snapshot(device)

    generation = session.generate(red, PROMPT)
    payload["generation"] = generation
    payload["generation_is_accuracy_claim"] = False

    image_dependent = image_gap > 0 and image_gap > (repeat_gap * 10)
    checks = {
        "device_is_requested": payload["model_parameter_device"] == device.type,
        "revision_pinned": payload["smolvlm_weights"]["revision"] == PINNED_REVISION,
        "float32": payload["model_parameter_dtype"] == "torch.float32",
        "multi_token": details["n_tokens"] >= 2,
        "logprob_finite": details["logprob"] == details["logprob"],
        "library_mean_nll_matches_sum": details["mean_nll_abs_diff"] < 1e-3,
        "gradient_finite": first["stats"]["finite"] and other["stats"]["finite"],
        "gradient_nonzero": first["stats"]["nonzero"] and other["stats"]["nonzero"],
        "gradient_image_dependent": image_dependent,
        "qwen_weights_not_loaded": payload["qwen_candidate"].get("weights_loaded") is False,
        "preprocess_shapes_match": payload["preprocess_gap_96x128"]["shapes_match"],
    }
    payload["checks"] = checks
    payload["passed"] = all(checks.values())
    if device.type == "mps":
        payload["mps_validation"] = "passed" if payload["passed"] else "failed"
    elif device.type == "cpu":
        payload["mps_validation"] = "PENDING"
    else:
        payload["mps_validation"] = "not_applicable"
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    # Gradients are large. Keep the statistics and drop the tensors before JSON.
    for key in ("gradient_red_square", "gradient_red_square_repeat", "gradient_blue_square"):
        payload[key].pop("gradient", None)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/models/feasibility-smolvlm2-mps.json"),
    )
    args = parser.parse_args()
    try:
        payload = run_measurement(args.device)
    except Exception as exc:
        mps_unavailable = args.device == "mps" and "MPS was requested but" in str(exc)
        if args.device == "mps" and mps_unavailable:
            mps_validation = "PENDING"
        elif args.device == "mps":
            mps_validation = "failed"
        else:
            mps_validation = "not_applicable"
        payload = {
            "measurement_kind": "checkpoint0_gradient_feasibility",
            "device_requested": args.device,
            "passed": False,
            "mps_validation": mps_validation,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "reproduction_command": REPRODUCTION_COMMAND,
            "note": "No passing measurement was produced. Do not relabel a partial run as MPS success.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": payload["passed"],
        "mps_validation": payload["mps_validation"],
        "device": payload["model_parameter_device"],
        "output": str(args.output),
    }, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


def _timed_gradient(session: VLMSession, device: torch.device, image: torch.Tensor, prompt: str, answer: str):
    synchronize(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    started = time.perf_counter()
    gradient = session.image_loss_gradient(image, prompt, answer)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "seconds": round(elapsed, 3),
        "stats": gradient_stats(gradient),
        "gradient": gradient,
        "memory": memory_snapshot(device),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def _survey_qwen() -> dict[str, Any]:
    try:
        surveyed = survey_weights(QWEN_CANDIDATE_ID)
    except Exception as exc:
        return {
            "repo_id": QWEN_CANDIDATE_ID,
            "weights_loaded": False,
            "action": "not_loaded",
            "survey_error_type": type(exc).__name__,
            "survey_error": str(exc),
        }
    surveyed["action"] = (
        "load_aborted"
        if not surveyed["load_permitted_by_weight_guard"]
        else "not_loaded_weight_guard_passed"
    )
    surveyed["adopted"] = False
    surveyed["weights_loaded"] = False
    return surveyed


def _machine() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    if platform.system() == "Darwin":
        import subprocess

        summary["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        summary["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    return summary


def _versions() -> dict[str, str]:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


if __name__ == "__main__":
    main()
