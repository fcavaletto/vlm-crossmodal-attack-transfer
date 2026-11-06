"""Run the Checkpoint 1 development pilot on MPS.

The command scores the 24 committed scenes. It does not fit H*, and it does not
run an attack. A request for any device other than MPS is refused.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from analysis.pilot_schema import LABEL, MODEL_ID, MODEL_REVISION, PREPROCESS, PRIMARY_CONTEXTS
from analysis.schemas import CONTEXTS
from scene_gen.paths import resolve_image_path
from vlm_iface.devices import memory_snapshot, resolve_device, synchronize
from vlm_iface.images import to_image_01
from vlm_iface.parsing import task_vocabulary
from vlm_iface.pins import PINNED_REVISION
from vlm_iface.preprocess import compare_preprocess
from vlm_iface.session import VLMSession

from .constants import GATE_SCENE_ID, IMAGE_HEIGHT, IMAGE_WIDTH, MANIFEST_SHA256
from .manifest import load_committed_manifest, repo_root
from .prompt import user_prompt
from .rows import gradient_stats, make_alignment_row, make_prediction_row
from .verify import VERIFICATION_COMMAND, assemble_summary, file_sha256, verify_directory

PYTHONPATH = "src/scene_gen:src/vlm_iface:src/analysis:src/pilot"


def require_mps(device_name: str) -> torch.device:
    """Refuse every device except MPS. There is no CPU fallback."""
    if device_name != "mps":
        raise RuntimeError(
            f"Refusing device {device_name!r}. A CPU result is not an MPS result."
        )
    device = resolve_device(device_name)
    if device.type != "mps":
        raise RuntimeError(f"Requested MPS but resolved {device.type}.")
    return device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint 1 development pilot. MPS only. No attacks.")
    parser.add_argument("--device", required=True, choices=["mps"])
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root()
    output = root / "docs" / "checkpoint-1"
    output.mkdir(parents=True, exist_ok=True)
    try:
        return run_pilot(root, output, device_name=args.device, gate_only=args.gate_only)
    except Exception as exc:
        payload = {
            "label": LABEL,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "PILOT_ERROR.json", payload)
        print(f"Pilot stopped: {type(exc).__name__}: {exc}", flush=True)
        return 1


def run_pilot(root: Path, output: Path, *, device_name: str, gate_only: bool) -> int:
    device = require_mps(device_name)
    if PINNED_REVISION != MODEL_REVISION:
        raise RuntimeError("The analysis pin and the model pin disagree.")
    manifest_path, records = load_committed_manifest(root)
    gate_record = next(row for row in records if row["scene_id"] == GATE_SCENE_ID)
    print(f"Loading {MODEL_ID} on MPS.", flush=True)
    before = memory_snapshot(device)
    load_started = time.perf_counter()
    session = VLMSession(
        device="mps",
        do_image_splitting=False,
        max_new_tokens=16,
        leading_space=True,
    )
    synchronize(device)
    load_seconds = time.perf_counter() - load_started
    after_load = memory_snapshot(device)
    parameter = next(session.model.parameters())
    model_check = _model_check(session, parameter)
    if model_check["problems"]:
        gate = _failed_gate(gate_record["scene_id"], model_check["problems"], before, after_load)
        _write_json(output / "gate.json", gate)
        print("Full-resolution gate did not run because the model check failed.", flush=True)
        return 1

    print(f"Gate scene {GATE_SCENE_ID} at {IMAGE_WIDTH}x{IMAGE_HEIGHT}.", flush=True)
    gate_started = time.perf_counter()
    image = _load_image(manifest_path, gate_record)
    preprocess_gap = compare_preprocess(
        session.processor.image_processor, image, do_image_splitting=False
    )
    gate_result = process_scene(
        session, manifest_path, gate_record, image=image, repeat=True, device=device
    )
    synchronize(device)
    gate_seconds = time.perf_counter() - gate_started
    after_gate = memory_snapshot(device)
    problems = list(model_check["problems"])
    problems.extend(gate_result["problems"])
    if not preprocess_gap["shapes_match"]:
        problems.append("float32 and uint8 preprocess shapes differ")
    gate = {
        "label": LABEL,
        "measurement_kind": "checkpoint1_full_resolution_gate",
        "passed": not problems,
        "device": device.type,
        "scene_id": GATE_SCENE_ID,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "problems": problems,
        "load_seconds": round(load_seconds, 3),
        "gate_seconds": round(gate_seconds, 3),
        "memory_before_load": before,
        "memory_after_load": after_load,
        "memory_after_gate": after_gate,
        "model_check": model_check,
        "preprocess_gap": _json_ready(preprocess_gap),
        "gradient_stats": gate_result["gradients"],
        "note": (
            "One committed 512x384 development scene, neutral and helpful_nonrevealing "
            "gradients, shared float32 preprocessing. Not an attack."
        ),
    }
    _write_json(output / "gate.json", gate)
    print(json.dumps({"gate_passed": gate["passed"], "problems": problems, "gate_seconds": gate["gate_seconds"]}), flush=True)
    if not gate["passed"]:
        return 1
    if gate_only:
        return 0

    predictions = list(gate_result["predictions"])
    gradients = list(gate_result["gradients"])
    alignments = list(gate_result["alignments"])
    scene_seconds = {GATE_SCENE_ID: round(gate_seconds, 3)}
    full_started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        if record["scene_id"] == GATE_SCENE_ID:
            continue
        print(f"Scene {index}/{len(records)} {record['scene_id']}", flush=True)
        started = time.perf_counter()
        result = process_scene(
            session, manifest_path, record, image=None, repeat=False, device=device
        )
        predictions.extend(result["predictions"])
        gradients.extend(result["gradients"])
        alignments.extend(result["alignments"])
        scene_seconds[record["scene_id"]] = round(time.perf_counter() - started, 3)
        if result["problems"]:
            print(f"  flagged {result['problems']}", flush=True)
    synchronize(device)
    full_seconds = time.perf_counter() - full_started
    after_pilot = memory_snapshot(device)
    predictions.sort(key=lambda row: (row["scene_id"], CONTEXTS.index(row["context"])))
    gradients.sort(key=lambda row: (row["scene_id"], PRIMARY_CONTEXTS.index(row["context"])))
    alignments.sort(
        key=lambda row: (row["scene_id"], row["source_context"], row["target_context"])
    )
    _write_jsonl(output / "predictions.jsonl", predictions)
    _write_jsonl(output / "gradient_stats.jsonl", gradients)
    _write_jsonl(output / "alignments.jsonl", alignments)
    run_meta = {
        "label": LABEL,
        "attacks_run": False,
        "h_star_fitted": False,
        "device": device.type,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
        "do_image_splitting": False,
        "max_new_tokens": 16,
        "leading_space": True,
        "attn_implementation": model_check["attn_implementation"],
        "gate_scene_id": GATE_SCENE_ID,
        "n_scenes": len(records),
        "load_seconds": round(load_seconds, 3),
        "gate_seconds": gate["gate_seconds"],
        "remaining_scenes_seconds": round(full_seconds, 3),
        "scene_seconds": scene_seconds,
        "memory_before_load": before,
        "memory_after_load": after_load,
        "memory_after_gate": after_gate,
        "memory_after_pilot": after_pilot,
        "environment": _environment(),
        "git_head_at_run": _git_head(root),
        "manifest_sha256": MANIFEST_SHA256,
        "commands": _commands(),
        "verification_command": _commands()["verify"],
        "helpful_text_role": "task guidance, not additional factual scene evidence",
    }
    _write_json(output / "run_meta.json", run_meta)
    reloaded_predictions = _read_jsonl(output / "predictions.jsonl")
    reloaded_gradients = _read_jsonl(output / "gradient_stats.jsonl")
    reloaded_alignments = _read_jsonl(output / "alignments.jsonl")
    reloaded_meta = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    reloaded_gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    hashes = {
        "predictions.jsonl": file_sha256(output / "predictions.jsonl"),
        "gradient_stats.jsonl": file_sha256(output / "gradient_stats.jsonl"),
        "alignments.jsonl": file_sha256(output / "alignments.jsonl"),
        "run_meta.json": file_sha256(output / "run_meta.json"),
        "gate.json": file_sha256(output / "gate.json"),
    }
    summary = assemble_summary(
        reloaded_predictions,
        reloaded_gradients,
        reloaded_alignments,
        reloaded_meta,
        reloaded_gate,
        hashes,
    )
    _write_json(output / "summary.json", summary)
    checked = verify_directory(output)
    print(json.dumps({"verify_ok": checked["ok"], "n_mismatches": checked["n_mismatches"]}), flush=True)
    if not checked["ok"]:
        for line in checked["mismatches"]:
            print(line, flush=True)
        return 1
    return 0


def process_scene(
    session: VLMSession,
    manifest_path: Path,
    record: dict[str, Any],
    *,
    image,
    repeat: bool,
    device: torch.device,
) -> dict[str, Any]:
    if image is None:
        image = _load_image(manifest_path, record)
    predictions = []
    for context in CONTEXTS:
        prompt = user_prompt(record["contexts"][context]["text"], record["question"])
        generated = session.generate_detailed(image, prompt, task_family=record["task_family"])
        scores = _score_vocabulary(session, image, prompt, record["task_family"])
        predictions.append(
            make_prediction_row(
                record=record,
                context=context,
                prompt=prompt,
                raw_output=str(generated["text"]),
                parsed_answer=generated["parsed_answer"],
                parse_reasons=list(generated["reasons"]),
                invalid=bool(generated["invalid"]),
                scores=scores,
                device=device.type,
            )
        )
    stored = {}
    gradient_rows = []
    problems = []
    for context in PRIMARY_CONTEXTS:
        prompt = user_prompt(record["contexts"][context]["text"], record["question"])
        synchronize(device)
        started = time.perf_counter()
        gradient = session.image_loss_gradient(image, prompt, record["answer"])
        synchronize(device)
        elapsed = time.perf_counter() - started
        stats = gradient_stats(gradient)
        repeat_diff = None
        if repeat:
            synchronize(device)
            second = session.image_loss_gradient(image, prompt, record["answer"])
            synchronize(device)
            repeat_diff = float((gradient - second).abs().max())
            del second
        problem = _gradient_problem(stats)
        if problem is not None:
            problems.append(f"{record['scene_id']} {context}: {problem}")
        stored[context] = gradient.detach().to(dtype=torch.float32, device="cpu")
        gradient_rows.append(
            _gradient_row(record, context, stats, elapsed, repeat_diff, device.type)
        )
        del gradient
    alignments = []
    for source, target in (
        ("neutral", "helpful_nonrevealing"),
        ("helpful_nonrevealing", "neutral"),
    ):
        alignments.append(
            make_alignment_row(
                scene_id=record["scene_id"],
                counterfactual_group_id=record["counterfactual_group_id"],
                task_family=record["task_family"],
                source_context=source,
                target_context=target,
                source_gradient=stored[source],
                target_gradient=stored[target],
                source_stats=next(row for row in gradient_rows if row["context"] == source),
                target_stats=next(row for row in gradient_rows if row["context"] == target),
                device=device.type,
            )
        )
    del stored
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return {
        "predictions": predictions,
        "gradients": gradient_rows,
        "alignments": alignments,
        "problems": problems,
    }


def _score_vocabulary(session: VLMSession, image, prompt: str, task_family: str) -> dict[str, dict[str, float]]:
    scores = {}
    for word in task_vocabulary(task_family):
        details = session.score_details(image, prompt, word)
        sum_nll = -float(details["logprob"])
        token_count = int(details["n_tokens"])
        if token_count < 1 or sum_nll != sum_nll:
            raise RuntimeError(f"Non-finite score for vocabulary item {word!r}")
        scores[word] = {
            "sum_nll": sum_nll,
            "token_count": token_count,
            "mean_nll": sum_nll / token_count,
            "library_mean_abs_diff": float(details["mean_nll_abs_diff"]),
        }
    return scores


def _gradient_row(record, context: str, stats: dict[str, Any], seconds: float, repeat, device: str) -> dict[str, Any]:
    if device != "mps":
        raise RuntimeError("A CPU result is not an MPS result.")
    return {
        "label": LABEL,
        "scene_id": record["scene_id"],
        "counterfactual_group_id": record["counterfactual_group_id"],
        "task_family": record["task_family"],
        "context": context,
        "finite": stats["finite"],
        "nonzero_fraction": stats["nonzero_fraction"],
        "l1": stats["l1"],
        "l2": stats["l2"],
        "linf": stats["linf"],
        "height": stats["height"],
        "width": stats["width"],
        "seconds": round(seconds, 3),
        "repeat_max_abs_diff": None if repeat is None else float(repeat),
        "device": device,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
    }


def _gradient_problem(stats: dict[str, Any]) -> str | None:
    if not stats["finite"]:
        return "non-finite gradient"
    if stats["l1"] == 0 or stats["nonzero_fraction"] == 0:
        return "zero gradient"
    return None


def _load_image(manifest_path: Path, record: dict[str, Any]):
    path = resolve_image_path(manifest_path, record["image_path"])
    image = to_image_01(path)
    if tuple(image.shape) != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(
            f"{record['scene_id']} has shape {tuple(image.shape)}, expected "
            f"(3, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )
    return image


def _model_check(session: VLMSession, parameter: torch.nn.Parameter) -> dict[str, Any]:
    problems = []
    if parameter.device.type != "mps":
        problems.append(f"parameters are on {parameter.device.type}")
    if parameter.dtype != torch.float32:
        problems.append(f"parameters are {parameter.dtype}")
    attention = getattr(session.model.config, "_attn_implementation", None)
    if attention != "eager":
        problems.append(f"attention implementation is {attention!r}")
    if session.do_image_splitting:
        problems.append("image splitting is on")
    if session.max_new_tokens != 16 or session.leading_space is not True:
        problems.append("generation settings differ from the approved pilot")
    if session.revision != PINNED_REVISION or session.model_id != MODEL_ID:
        problems.append("model id or revision differs from the pin")
    return {
        "problems": problems,
        "parameter_device": parameter.device.type,
        "parameter_dtype": str(parameter.dtype),
        "attn_implementation": attention,
        "do_image_splitting": session.do_image_splitting,
        "max_new_tokens": session.max_new_tokens,
        "leading_space": session.leading_space,
        "model_id": session.model_id,
        "model_revision": session.revision,
    }


def _failed_gate(scene_id: str, problems: list[str], before, after_load) -> dict[str, Any]:
    return {
        "label": LABEL,
        "measurement_kind": "checkpoint1_full_resolution_gate",
        "passed": False,
        "device": "mps",
        "scene_id": scene_id,
        "problems": problems,
        "memory_before_load": before,
        "memory_after_load": after_load,
    }


def _environment() -> dict[str, Any]:
    import PIL
    import numpy
    import transformers

    summary = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    if platform.system() == "Darwin":
        summary["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        summary["memory_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    return summary


def _commands() -> dict[str, str]:
    return {
        "tests": f"PYTHONPATH={PYTHONPATH} python -m pytest tests/scene_gen tests/vlm_iface tests/analysis tests/pilot -q",
        "pilot": f"PYTHONPATH={PYTHONPATH} python -m pilot.run --device mps",
        "gate_only": f"PYTHONPATH={PYTHONPATH} python -m pilot.run --device mps --gate-only",
        "verify": f"PYTHONPATH={PYTHONPATH} {VERIFICATION_COMMAND}",
    }


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
