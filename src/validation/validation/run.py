"""Score fixtures/development_v2 on MPS. Development only.

The prompt, the parser, the gradient loss, and the scene list are the locked
protocol. This process does not run an attack, does not fit H*, and does not
load Qwen. A failed full-resolution gate stops the run before the table.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from analysis.pilot_schema import PREPROCESS
from answer_scoring.class_loss import two_token_logprob
from answer_scoring.compare import score_logits
from answer_scoring.surfaces import SurfaceSet, build_surfaces
from pilot.prompt import user_prompt
from pilot.rows import gradient_stats, make_alignment_row
from scene_v2.paths import resolve_image_path
from vlm_iface.devices import available_bytes, memory_snapshot, resolve_device, synchronize
from vlm_iface.images import to_image_01
from vlm_iface.pins import MIN_AVAILABLE_BYTES
from vlm_iface.pins import MODEL_ID as PINNED_MODEL_ID
from vlm_iface.pins import PINNED_REVISION
from vlm_iface.preprocess import compare_preprocess
from vlm_iface.session import VLMSession

from .protocol import (
    CONTEXTS,
    FORMAT_LINE,
    GATE_SCENE_ID,
    GRADIENT_LOSS,
    HELPFUL_TEXT_ROLE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL,
    MANIFEST_RELATIVE,
    MANIFEST_SHA256,
    MEASUREMENT_KIND,
    MODEL_ID,
    MODEL_REVISION,
    N_CONTEXTS,
    N_SCENES,
    OUTPUT_RELATIVE,
    PRIMARY_CONTEXTS,
    PROTOCOL_RELATIVE,
    TASK_FAMILIES,
    assert_not_checkpoint1,
    repo_root,
)
from .summarize import build_summary

ORDERED_DIRECTIONS = (
    ("neutral", "helpful_nonrevealing"),
    ("helpful_nonrevealing", "neutral"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint 1.5 v2 validation. MPS only. No attacks.")
    parser.add_argument("--device", required=True, choices=["mps"])
    args = parser.parse_args(argv)
    root = repo_root()
    output = (root / OUTPUT_RELATIVE).resolve()
    assert_not_checkpoint1(output, root)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return run(root, output, device_name=args.device)
    except Exception as exc:
        payload = {
            "label": LABEL,
            "passed": False,
            "attacks_run": False,
            "h_star_fitted": False,
            "qwen_loaded": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "run_error.json", payload)
        print(f"Validation stopped: {type(exc).__name__}: {exc}", flush=True)
        return 1


def run(root: Path, output: Path, *, device_name: str) -> int:
    if device_name != "mps":
        raise RuntimeError(f"Refusing device {device_name!r}. A CPU result is not an MPS result.")
    if PINNED_MODEL_ID != MODEL_ID or PINNED_REVISION != MODEL_REVISION:
        raise RuntimeError("The model pin does not match the locked protocol.")
    device = resolve_device(device_name)
    if device.type != "mps":
        raise RuntimeError(f"Requested MPS but resolved {device.type}.")
    manifest_path, records = _load_manifest(root)
    protocol_sha = hashlib.sha256((root / PROTOCOL_RELATIVE).read_bytes()).hexdigest()
    free = available_bytes()
    before = memory_snapshot(device)
    if free < MIN_AVAILABLE_BYTES:
        refusal = {
            "label": LABEL,
            "refused": True,
            "reason": "available memory below the Checkpoint 0 load guard",
            "available_bytes": free,
            "min_available_bytes": MIN_AVAILABLE_BYTES,
            "memory": before,
            "device": device.type,
            "attacks_run": False,
            "h_star_fitted": False,
            "qwen_loaded": False,
        }
        _write_json(output / "memory_refusal.json", refusal)
        print(json.dumps({"refused": True, "available_bytes": free}), flush=True)
        return 2

    print(f"Loading {MODEL_ID} on MPS.", flush=True)
    load_started = time.perf_counter()
    session = VLMSession(
        device="mps",
        do_image_splitting=False,
        max_new_tokens=16,
        leading_space=True,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    synchronize(device)
    load_seconds = time.perf_counter() - load_started
    after_load = memory_snapshot(device)
    parameter = next(session.model.parameters())
    model_check = _model_check(session, parameter)
    gate_record = next(row for row in records if row["scene_id"] == GATE_SCENE_ID)
    if model_check["problems"]:
        gate = _failed_gate(model_check["problems"], before, after_load)
        _write_json(output / "gate.json", gate)
        print("Full-resolution gate did not run because the model check failed.", flush=True)
        return 1

    print(f"Full-resolution gate on {GATE_SCENE_ID}.", flush=True)
    gate_started = time.perf_counter()
    gate_image = _load_image(manifest_path, gate_record)
    preprocess_gap = compare_preprocess(
        session.processor.image_processor, gate_image, do_image_splitting=False
    )
    try:
        gate_gradients, gate_alignments, gate_problems = _gradients(
            session, gate_image, gate_record, device, repeat=True
        )
    except RuntimeError as exc:
        gate_problems = [f"{GATE_SCENE_ID}: {exc}"]
        gate_gradients, gate_alignments = [], []
    synchronize(device)
    gate_seconds = time.perf_counter() - gate_started
    after_gate = memory_snapshot(device)
    problems = list(gate_problems)
    if not preprocess_gap["shapes_match"]:
        problems.append("float32 and uint8 preprocess shapes differ")
    gate = {
        "label": LABEL,
        "measurement_kind": "checkpoint1_5_full_resolution_gate",
        "passed": not problems,
        "device": device.type,
        "scene_id": GATE_SCENE_ID,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "loss": GRADIENT_LOSS,
        "problems": problems,
        "load_seconds": round(load_seconds, 3),
        "gate_seconds": round(gate_seconds, 3),
        "memory_before_load": before,
        "memory_after_load": after_load,
        "memory_after_gate": after_gate,
        "model_check": model_check,
        "preprocess_gap": _json_ready(preprocess_gap),
        "gradient_stats": gate_gradients,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
        "note": (
            "One committed 512x384 v2 scene. Canonical-word image gradients on "
            "neutral and helpful_nonrevealing, each repeated once. Not an attack."
        ),
    }
    _write_json(output / "gate.json", gate)
    print(json.dumps({"gate_passed": gate["passed"], "problems": problems}), flush=True)
    if not gate["passed"]:
        return 1

    surfaces = {
        family: build_surfaces(family, session.processor.tokenizer, leading_space=True)
        for family in TASK_FAMILIES
    }
    predictions: list[dict[str, Any]] = []
    score_started = time.perf_counter()
    try:
        for index, record in enumerate(records, start=1):
            print(f"Score {index}/{len(records)} {record['scene_id']}", flush=True)
            image = gate_image if record["scene_id"] == GATE_SCENE_ID else _load_image(manifest_path, record)
            predictions.extend(_score_scene(session, image, record, surfaces[record["task_family"]]))
            _write_jsonl(output / "predictions.jsonl", _sort_predictions(predictions))
    finally:
        score_seconds = time.perf_counter() - score_started
    after_score = memory_snapshot(device)
    _write_jsonl(output / "predictions.jsonl", _sort_predictions(predictions))
    predictions = _read_jsonl(output / "predictions.jsonl")
    scoring_stable, stable_reason = _scoring_stable(predictions, after_score)
    print(json.dumps({"scoring_stable": scoring_stable, "reason": stable_reason, "rows": len(predictions)}), flush=True)

    gradients: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    gradient_seconds = None
    after_gradients = after_score
    gradient_reason = None
    if scoring_stable:
        gradient_started = time.perf_counter()
        try:
            for index, record in enumerate(records, start=1):
                print(f"Gradient {index}/{len(records)} {record['scene_id']}", flush=True)
                image = gate_image if record["scene_id"] == GATE_SCENE_ID else _load_image(manifest_path, record)
                scene_gradients, scene_alignments, scene_problems = _gradients(
                    session, image, record, device, repeat=False
                )
                gradients.extend(scene_gradients)
                alignments.extend(scene_alignments)
                _write_jsonl(output / "gradient_stats.jsonl", gradients)
                _write_jsonl(output / "alignments.jsonl", alignments)
                if scene_problems:
                    print(f"  flagged {scene_problems}", flush=True)
        except RuntimeError as exc:
            if not _memory_error(exc):
                raise
            gradient_reason = f"full gradient pass stopped: {exc}"
        synchronize(device)
        gradient_seconds = round(time.perf_counter() - gradient_started, 3)
        after_gradients = memory_snapshot(device)
    else:
        gradients = list(gate_gradients)
        alignments = list(gate_alignments)
        gradient_reason = stable_reason
    del session
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    covered = {row["scene_id"] for row in gradients}
    full_pass = scoring_stable and gradient_reason is None and covered == {row["scene_id"] for row in records}
    if scoring_stable and gradient_reason is None and not full_pass:
        gradient_reason = "gradient rows do not cover every v2 scene"
    run_meta = {
        "label": LABEL,
        "measurement_kind": MEASUREMENT_KIND,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
        "format_line_used": False,
        "helpful_text_role": HELPFUL_TEXT_ROLE,
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
        "do_image_splitting": False,
        "max_new_tokens": 16,
        "leading_space": True,
        "attn_implementation": model_check["attn_implementation"],
        "gradient_loss": GRADIENT_LOSS,
        "gate_scene_id": GATE_SCENE_ID,
        "n_scenes": len(records),
        "load_seconds": round(load_seconds, 3),
        "gate_seconds": gate["gate_seconds"],
        "score_seconds": round(score_seconds, 3),
        "gradient_seconds": gradient_seconds,
        "scoring_stable": scoring_stable,
        "full_gradient_pass": full_pass,
        "full_gradient_pass_reason": None if full_pass else gradient_reason,
        "memory_before_load": before,
        "memory_after_load": after_load,
        "memory_after_gate": after_gate,
        "memory_after_scoring": after_score,
        "memory_after_gradients": after_gradients,
        "environment": _environment(),
        "git_head_at_run": _git_head(root),
        "protocol_sha256": protocol_sha,
        "manifest_sha256": MANIFEST_SHA256,
        "verification_command": "PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2 python -m validation.verify",
    }
    _write_json(output / "run_meta.json", run_meta)
    _write_jsonl(output / "gradient_stats.jsonl", gradients)
    _write_jsonl(output / "alignments.jsonl", alignments)
    gradients = _read_jsonl(output / "gradient_stats.jsonl")
    alignments = _read_jsonl(output / "alignments.jsonl")
    run_meta = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    summary = build_summary(
        predictions,
        gradients,
        alignments,
        run_meta=run_meta,
        gate=gate,
        expected_scene_ids=[row["scene_id"] for row in records],
        require_full_sample=True,
    )
    _write_json(output / "summary.json", summary)
    from .verify import verify_directory

    checked = verify_directory(output, root)
    print(json.dumps({"verify_ok": checked["ok"], "n_groups_eligible": checked.get("n_groups_eligible")}), flush=True)
    if not checked["ok"]:
        for line in checked["mismatches"]:
            print(line, flush=True)
        return 1
    return 0


def _score_scene(session: VLMSession, image, record: Mapping[str, Any], surfaces: SurfaceSet) -> list[dict[str, Any]]:
    rows = []
    for context in CONTEXTS:
        context_text = record["contexts"][context]["text"]
        prompt = user_prompt(context_text, record["question"])
        if FORMAT_LINE in prompt:
            raise RuntimeError("The locked prompt contains the format line.")
        generated = session.generate_detailed(image, prompt, task_family=record["task_family"])
        logits = session.prompt_last_logits(image, prompt)
        extra = _extra_logprobs(session, image, prompt, logits, surfaces)
        scored = score_logits(
            logits,
            surfaces,
            raw_output=str(generated["text"]),
            truncated=bool(generated["truncated"]),
            answer=record["answer"],
            aliases=list(record["answer_aliases"]),
            extra_logprobs=extra,
        )
        for value in list(scored["class_nll"].values()) + list(scored["word_only_nll"].values()):
            if value != value or abs(value) == float("inf"):
                raise RuntimeError(f"Non-finite loss for {record['scene_id']} {context}")
        rows.append(
            {
                "label": LABEL,
                "measurement_kind": MEASUREMENT_KIND,
                "scene_id": record["scene_id"],
                "counterfactual_group_id": record["counterfactual_group_id"],
                "task_family": record["task_family"],
                "context": context,
                "answer": record["answer"],
                "answer_aliases": list(record["answer_aliases"]),
                "context_text": context_text,
                "question": record["question"],
                "prompt": prompt,
                "format_line_appended": False,
                "reveals_answer": bool(record["contexts"][context]["reveals_answer"]),
                "n_objects": len(record["objects"]),
                "ambiguity_flags": list(record["ambiguity_flags"]),
                "raw_output": str(generated["text"]),
                "parsed_answer": scored["strict_parsed_answer"],
                "parse_reasons": list(scored["strict_reasons"]),
                "correct": scored["strict_correct"],
                "valid_wrong": scored["strict_valid_wrong"],
                "invalid": scored["strict_invalid"],
                "truncated": bool(generated["truncated"]),
                "class_nll": scored["class_nll"],
                "class_argmin": scored["class_argmin"],
                "class_disagreement": scored["class_disagreement"],
                "word_only_nll": scored["word_only_nll"],
                "word_only_argmin": scored["word_only_argmin"],
                "word_only_disagreement": scored["word_only_disagreement"],
                "device": "mps",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "preprocess": PREPROCESS,
            }
        )
    if len(rows) != N_CONTEXTS:
        raise RuntimeError(f"{record['scene_id']} did not produce four contexts")
    return rows


def _gradients(session, image, record, device, *, repeat: bool):
    stored = {}
    rows = []
    problems = []
    for context in PRIMARY_CONTEXTS:
        prompt = user_prompt(record["contexts"][context]["text"], record["question"])
        synchronize(device)
        started = time.perf_counter()
        try:
            gradient = session.image_loss_gradient(image, prompt, record["answer"])
        except RuntimeError as exc:
            if _memory_error(exc):
                raise
            problems.append(f"{record['scene_id']} {context}: {exc}")
            rows.append(_failed_gradient_row(record, context, str(exc)))
            continue
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
        if not stats["finite"] or stats["l1"] == 0 or stats["nonzero_fraction"] == 0:
            problems.append(f"{record['scene_id']} {context}: gradient is non-finite or zero")
        stored[context] = gradient.detach().to(dtype=torch.float32, device="cpu")
        rows.append(_gradient_row(record, context, stats, elapsed, repeat_diff))
        del gradient
    alignments = []
    if set(stored) == set(PRIMARY_CONTEXTS):
        for source, target in ORDERED_DIRECTIONS:
            alignments.append(
                make_alignment_row(
                    scene_id=record["scene_id"],
                    counterfactual_group_id=record["counterfactual_group_id"],
                    task_family=record["task_family"],
                    source_context=source,
                    target_context=target,
                    source_gradient=stored[source],
                    target_gradient=stored[target],
                    source_stats=next(row for row in rows if row["context"] == source),
                    target_stats=next(row for row in rows if row["context"] == target),
                    device="mps",
                )
            )
    del stored
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return rows, alignments, problems


def _gradient_row(record, context, stats, seconds, repeat_diff) -> dict[str, Any]:
    return {
        "label": LABEL,
        "measurement_kind": MEASUREMENT_KIND,
        "loss": GRADIENT_LOSS,
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
        "repeat_max_abs_diff": None if repeat_diff is None else float(repeat_diff),
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
        "error": None,
    }


def _failed_gradient_row(record, context: str, error: str) -> dict[str, Any]:
    return {
        "label": LABEL,
        "measurement_kind": MEASUREMENT_KIND,
        "loss": GRADIENT_LOSS,
        "scene_id": record["scene_id"],
        "counterfactual_group_id": record["counterfactual_group_id"],
        "task_family": record["task_family"],
        "context": context,
        "finite": False,
        "nonzero_fraction": None,
        "l1": None,
        "l2": None,
        "linf": None,
        "height": None,
        "width": None,
        "seconds": None,
        "repeat_max_abs_diff": None,
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
        "error": error,
    }


def _scoring_stable(predictions: list[dict[str, Any]], after_score: Mapping[str, Any]) -> tuple[bool, str]:
    if len(predictions) != N_SCENES * N_CONTEXTS:
        return False, f"scoring produced {len(predictions)} rows"
    scenes = {row["scene_id"] for row in predictions}
    if len(scenes) != N_SCENES:
        return False, "scoring did not cover every scene"
    available = available_bytes()
    if available < MIN_AVAILABLE_BYTES:
        return False, (
            "available memory after scoring was below the 6 GiB guard, so the "
            "full gradient pass was not started"
        )
    if after_score.get("device") != "mps":
        return False, "scoring memory snapshot is not MPS"
    return True, "scoring completed with finite losses and memory still above the guard"


def _memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "not enough memory" in text


def _extra_logprobs(session, image, prompt: str, logits, surfaces: SurfaceSet) -> dict[tuple[int, ...], float]:
    sequences = [
        surface.token_ids
        for group in surfaces.by_class.values()
        for surface in group
        if len(surface.token_ids) > 1
    ]
    if not sequences:
        return {}
    if any(len(sequence) != 2 for sequence in sequences):
        raise RuntimeError("Alias sequences longer than two tokens are not in this loss.")
    found: dict[tuple[int, ...], float] = {}
    for prefix in {sequence[:1] for sequence in sequences}:
        continuation = session.prompt_last_logits(image, prompt, prefix_token_ids=list(prefix))
        for sequence in sequences:
            if sequence[:1] != prefix:
                continue
            found[sequence] = float(
                two_token_logprob(logits, continuation, sequence[0], sequence[1]).detach().cpu()
            )
    return found


def _load_manifest(root: Path):
    path = root / MANIFEST_RELATIVE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        raise RuntimeError("The v2 manifest does not match the sha256 locked in the protocol.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != N_SCENES:
        raise RuntimeError(f"Expected {N_SCENES} v2 scenes, found {len(records)}.")
    if records[0]["scene_id"] != GATE_SCENE_ID:
        raise RuntimeError("The gate scene is not the first committed v2 scene.")
    for record in records:
        if record["split"] != "development":
            raise RuntimeError(f"{record['scene_id']} is not a development scene.")
        if set(record["contexts"]) != set(CONTEXTS):
            raise RuntimeError(f"{record['scene_id']} does not have the four contexts.")
    return path, records


def _load_image(manifest_path: Path, record: Mapping[str, Any]):
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
        problems.append("generation settings differ from the locked protocol")
    if session.revision != MODEL_REVISION or session.model_id != MODEL_ID:
        problems.append("model id or revision differs from the pin")
    if "qwen" in session.model_id.lower():
        problems.append("Qwen is loaded")
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


def _failed_gate(problems: list[str], before, after_load) -> dict[str, Any]:
    return {
        "label": LABEL,
        "measurement_kind": "checkpoint1_5_full_resolution_gate",
        "passed": False,
        "device": "mps",
        "scene_id": GATE_SCENE_ID,
        "problems": problems,
        "memory_before_load": before,
        "memory_after_load": after_load,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
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
        summary["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    return summary


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _sort_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["scene_id"], CONTEXTS.index(row["context"])))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_not_checkpoint1(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    assert_not_checkpoint1(path)
    lines = [json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
