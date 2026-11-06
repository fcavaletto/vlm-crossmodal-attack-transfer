"""Guarded Qwen load and labeled-color RGB gradients for development_v4.

One process calls ``from_pretrained`` at most once, and only after the
Checkpoint 1.5 budget permits it. The script does not lower that budget, does
not take a signed step, and does not fit H*.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .gate import gate_scene_ids, gate_scene_table, judge_gate
from .measure import MemoryStop, appended_forward_gap, differentiate_scene, finite_difference_check, prepare_encoded
from .offline import load_predictions, verify_locked_inputs
from .protocol import (
    LABEL,
    MODEL_ID,
    MODEL_KEY,
    OUTPUT_RELATIVE,
    PINNED_REVISION,
    PREDICTION_RELATIVE,
    PRIMARY_CONTEXTS,
    PROTOCOL_RELATIVE,
    REPRODUCTION_COMMAND,
    STARTED_FROM_COMMIT,
    output_dir,
)
from .records import clean_margin, directed_alignments, stability_report
from .summarize import build_summary, canonical_json, sha256_text

LEGACY_RESIDENT = Path("docs/checkpoint-1.7/development_v4/resident.json")


class StopRun(RuntimeError):
    """A recorded stop. The model load is not repeated."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def main() -> None:
    root = _repo_root(Path(__file__).resolve())
    started = time.perf_counter()
    payload = execute(root)
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    _write_outputs(root, payload)
    print(json.dumps(_public_status(payload), sort_keys=True), flush=True)


def execute(root: Path) -> dict[str, Any]:
    """Run or resume. A saved budget refusal is not opened again."""
    progress_path = output_dir(root) / "progress.json"
    progress = _load_progress(progress_path)
    terminal = {
        "completed",
        "stopped_budget",
        "stopped_oom",
        "stopped_gate",
        "stopped_loss_identity",
        "stopped_revision",
        "stopped_prompt",
        "stopped_tokenization",
        "stopped_unexpected_header",
        "stopped_missing_weights",
        "stopped_unexpected_load",
        "stopped_protocol",
        "stopped_resident",
        "error",
    }
    if progress.get("status") in terminal:
        if progress.get("status") != "completed":
            progress["resume_refused"] = "the saved run already stopped; the model is not loaded again"
        return progress
    try:
        _execute(root, progress, progress_path)
    except StopRun as exc:
        progress["status"] = exc.status
        progress["stopped_reason"] = str(exc)
        _release_resident(root)
    except Exception as exc:
        progress["status"] = "error"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        _release_resident(root)
        _finalize(root, progress)
        _save(progress_path, progress)
        raise
    _finalize(root, progress)
    _save(progress_path, progress)
    return progress


def _execute(root: Path, progress: dict[str, Any], progress_path: Path) -> None:
    import torch
    from dev_pilot.fixtures import load_manifest
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.budget import dtype_decisions, load_budget_module, select_dtype
    from qwen_rgb.compare import compare_image
    from qwen_rgb.measure import (
        bf16_probe,
        load_model_after_budget,
        load_processor,
        prepare_environment,
        read_local_header,
        weight_file,
    )
    from qwen_rgb.rgb_path import assert_pinned_processor
    from vlm_iface.devices import available_bytes, memory_snapshot
    from vlm_iface.images import to_image_01

    if progress.get("protocol_sha256") and progress["protocol_sha256"] != _protocol_sha256(root):
        raise StopRun("stopped_protocol", "The measurement protocol file changed. Refusing to resume under a new rule.")
    progress.setdefault("status", "running")
    progress.setdefault("scenes", {})
    progress.setdefault("processor_gaps", {})
    progress["label"] = LABEL
    progress["started_from_commit"] = STARTED_FROM_COMMIT
    progress["reproduction_command"] = REPRODUCTION_COMMAND
    progress["protocol_sha256"] = _protocol_sha256(root)
    progress["git_head_at_start"] = _git_head(root)
    progress["model_from_pretrained_called"] = bool(progress.get("model_from_pretrained_called"))
    progress["attacks_run"] = False
    progress["h_star_fitted"] = False
    progress["environment"] = prepare_environment()
    locked = verify_locked_inputs(root)
    progress["offline_verification"] = locked
    records = load_manifest(root)
    progress["gate_scenes"] = gate_scene_table(records)
    gate_ids = gate_scene_ids(records)
    _acquire_resident(root, progress)
    _save(progress_path, progress)

    print("Loading the processor. Model weights stay on disk.", flush=True)
    processor, image_processor = load_processor()
    assert_pinned_processor(image_processor)
    identity = assert_locked_tokenization(processor, MODEL_KEY)
    progress["tokenization"] = identity
    if identity["answer_prefix"] != "<|im_end|>\n<|im_start|>assistant\n":
        raise StopRun("stopped_tokenization", "The answer prefix is not the Checkpoint 1.7 Qwen cue.")
    token_ids = {color: tuple(int(token) for token in ids) for color, ids in identity["token_ids"].items()}
    progress["token_ids"] = {color: list(ids) for color, ids in token_ids.items()}
    _save(progress_path, progress)

    budget = load_budget_module(root)
    probe = bf16_probe(torch)
    progress["bf16_probe"] = probe
    weights = weight_file()
    if not weights.is_file():
        raise StopRun("stopped_missing_weights", "The local safetensors file is absent. This script does not download it.")
    header = read_local_header(weights)
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS:
        raise StopRun("stopped_unexpected_header", "The header parameter count is not the Checkpoint 1.5 Qwen count.")
    if header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
        raise StopRun("stopped_unexpected_header", "The header float32 expansion is not the Checkpoint 1.5 Qwen size.")
    if header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        raise StopRun("stopped_unexpected_header", "The header is not exclusively BF16.")
    progress["header"] = {
        "parameter_count": header["parameter_count"],
        "float32_weight_bytes": header["float32_weight_bytes"],
        "file_bytes": header["file_bytes"],
        "dtype_counts": header["dtype_counts"],
    }
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    available = available_bytes()
    decisions = dtype_decisions(
        budget,
        parameter_count=header["parameter_count"],
        float32_weight_bytes=header["float32_weight_bytes"],
        available_bytes=available,
        mps_available=bool(torch.backends.mps.is_available()),
    )
    selection = select_dtype(decisions, bf16_probe_ok=bool(probe.get("matmul_ok")))
    progress["budget_recheck_immediately_before_load"] = decisions
    progress["dtype_selection_immediately_before_load"] = selection
    progress["available_bytes_entering_decision"] = available
    progress["available_bytes_at_decision"] = available_bytes()
    if not selection["permitted"] or selection["dtype"] is None:
        progress["model_from_pretrained_called"] = False
        _collect_gaps(progress, records, image_processor, root, to_image_01, compare_image)
        _save(progress_path, progress)
        required = decisions["bfloat16"]["required_available_bytes"]
        raise StopRun(
            "stopped_budget",
            selection["reason"]
            + f" Available bytes entering the decision: {available}. "
            + f"Fresh available bytes: {progress['available_bytes_at_decision']}. "
            + f"BF16 required bytes: {required}.",
        )

    dtype_name = str(selection["dtype"])
    device = torch.device("mps")
    progress["model_load_started"] = True
    progress["model_from_pretrained_called"] = True
    _save(progress_path, progress)
    print("Loading Qwen once.", flush=True)
    started = time.perf_counter()
    model = load_model_after_budget(decisions[dtype_name], dtype_name)
    try:
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from vlm_iface.devices import synchronize

        synchronize(device)
        progress["load_seconds"] = round(time.perf_counter() - started, 3)
        parameter = next(model.parameters())
        commit = getattr(model.config, "_commit_hash", None)
        progress["loaded_model"] = {
            "device": parameter.device.type,
            "dtype": str(parameter.dtype),
            "commit_hash": commit,
            "revision_argument": PINNED_REVISION,
            "model_id": MODEL_ID,
        }
        _track_memory(progress, memory_snapshot(device), available_bytes())
        if parameter.device.type != "mps" or str(parameter.dtype) != f"torch.{dtype_name}":
            raise StopRun("stopped_unexpected_load", "The loaded parameters are not the permitted dtype on MPS.")
        if commit not in (None, PINNED_REVISION):
            raise StopRun("stopped_revision", f"Loaded commit {commit} is not {PINNED_REVISION}.")
        _collect_gaps(progress, records, image_processor, root, to_image_01, compare_image)
        _save(progress_path, progress)
        rows = load_predictions(root)
        stored = {(row["scene_id"], row["context"]): row for row in rows}
        by_id = {record["scene_id"]: record for record in records}
        _measure_all(
            progress,
            progress_path,
            root=root,
            records=records,
            by_id=by_id,
            stored=stored,
            gate_ids=gate_ids,
            model=model,
            processor=processor,
            image_processor=image_processor,
            token_ids=token_ids,
            device=device,
            dtype_name=dtype_name,
        )
        gate = judge_gate(progress["scenes"], gate_ids)
        progress["gate"] = gate
        if not gate["passed"]:
            raise StopRun("stopped_gate", "The four-scene gate did not pass. The remaining scenes were not differentiated.")
        progress["status"] = "completed"
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _release_resident(root)


def _measure_all(progress, progress_path, *, root, records, by_id, stored, gate_ids, model, processor, image_processor, token_ids, device, dtype_name) -> None:
    from choice_eval.prompt import build_prompt
    from dev_pilot.rows import user_text_sha256
    from vlm_iface.devices import available_bytes, memory_snapshot

    order = [record["scene_id"] for record in records]
    gate_set = set(gate_ids)
    if progress.get("gate", {}).get("passed") is True:
        pending = [scene_id for scene_id in order if not _scene_done(progress["scenes"].get(scene_id))]
    else:
        pending = [scene_id for scene_id in gate_ids if not _scene_done(progress["scenes"].get(scene_id))]
        if not pending and all(_scene_done(progress["scenes"].get(scene_id)) for scene_id in gate_ids):
            gate = judge_gate(progress["scenes"], gate_ids)
            progress["gate"] = gate
            if not gate["passed"]:
                raise StopRun("stopped_gate", "The four-scene gate did not pass. The remaining scenes were not differentiated.")
            pending = [scene_id for scene_id in order if not _scene_done(progress["scenes"].get(scene_id))]
    for index, scene_id in enumerate(pending, start=1):
        record = by_id[scene_id]
        print(f"Scene {scene_id} ({index}/{len(pending)})", flush=True)
        block = progress["scenes"].setdefault(
            scene_id,
            {
                "scene_id": scene_id,
                "counterfactual_group_id": record["counterfactual_group_id"],
                "answer": record["answer"],
                "role": "gate" if scene_id in gate_set else "continuation",
            },
        )
        arrays: dict[str, np.ndarray | None] = {}
        for context in PRIMARY_CONTEXTS:
            if context in block and isinstance(block[context].get("gradient"), dict):
                arrays[context] = _load_scratch(root, scene_id, context)
                continue
            user_text = build_prompt(record["contexts"][context]["text"], record["question"]).user_text()
            text_hash = user_text_sha256(user_text)
            stored_row = stored[(scene_id, context)]
            if stored_row.get("user_text_sha256") != text_hash:
                raise StopRun("stopped_prompt", f"{scene_id} {context} prompt hash does not match the stored row.")
            image_path = root / "fixtures" / "development_v4" / record["image_path"]
            try:
                measured, array = differentiate_scene(
                    model=model,
                    processor=processor,
                    image_processor=image_processor,
                    image_path=image_path,
                    user_text=user_text,
                    token_ids=token_ids,
                    label=str(record["answer"]),
                    device=device,
                )
            except MemoryStop as exc:
                block[context] = _attach_common(
                    _oom_record(str(exc)),
                    stored_row,
                    record,
                    text_hash,
                )
                _save(progress_path, progress)
                raise StopRun("stopped_oom", "The runtime reported out of memory. The budget was not lowered and the load was not retried.") from exc
            measured = _attach_common(measured, stored_row, record, text_hash)
            if measured.get("loss_identity_agreed") is False:
                block[context] = measured
                _save(progress_path, progress)
                raise StopRun("stopped_loss_identity", "The labeled-color NLL formulas disagreed, so the gradient was not used.")
            if scene_id in gate_set and array is not None and measured["gradient"]["finite"] is True:
                try:
                    _gate_checks(
                        measured,
                        array,
                        model=model,
                        processor=processor,
                        image_processor=image_processor,
                        image_path=image_path,
                        user_text=user_text,
                        token_ids=token_ids,
                        label=str(record["answer"]),
                        device=device,
                        dtype_name=dtype_name,
                        token_id=int(token_ids[str(record["answer"])][0]),
                    )
                except MemoryStop as exc:
                    block[context] = measured
                    _save_scratch(root, scene_id, context, array)
                    _save(progress_path, progress)
                    raise StopRun(
                        "stopped_oom",
                        "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
                    ) from exc
            else:
                measured["finite_difference"] = {
                    "status": "not_run",
                    "numerical_check": "not_run",
                    "reason": "finite-difference checks run on the four gate scenes when the gradient is finite",
                }
                measured["stability"] = {
                    "ran": False,
                    "numerical_check": "not_run",
                    "reason": "stability repeats run on the four gate scenes when the gradient is finite",
                }
                measured["appended_forward"] = {"status": "not_run", "reason": "the appended-forward check runs on the four gate scenes"}
            block[context] = measured
            arrays[context] = array
            if array is not None and measured["gradient"]["finite"] is True:
                _save_scratch(root, scene_id, context, array)
            _track_memory(progress, memory_snapshot(device), available_bytes())
            _save(progress_path, progress)
        alignment = directed_alignments(arrays.get("neutral"), arrays.get("helpful_nonrevealing"))
        block.update(alignment)
        for context in PRIMARY_CONTEXTS:
            _delete_scratch(root, scene_id, context)
        del arrays
        _save(progress_path, progress)
        if scene_id in gate_set and scene_id == gate_ids[-1] and progress.get("gate", {}).get("passed") is not True:
            gate = judge_gate(progress["scenes"], gate_ids)
            progress["gate"] = gate
            _save(progress_path, progress)
            if not gate["passed"]:
                raise StopRun("stopped_gate", "The four-scene gate did not pass. The remaining scenes were not differentiated.")
            # List iteration visits these appended continuation scenes.
            seen = set(pending)
            for extra in order:
                if extra in seen or _scene_done(progress["scenes"].get(extra)):
                    continue
                pending.append(extra)
                seen.add(extra)


def _gate_checks(measured, array, *, model, processor, image_processor, image_path, user_text, token_ids, label, device, dtype_name, token_id) -> None:
    prepared = prepare_encoded(
        processor=processor,
        image_processor=image_processor,
        image_path=image_path,
        user_text=user_text,
    )
    if prepared is None:
        measured["finite_difference"] = {"status": "not_run", "reason": "the finite-difference prompt grid did not match"}
        measured["appended_forward"] = {"status": "not_run", "reason": "the prompt grid did not match"}
    else:
        try:
            measured["appended_forward"] = appended_forward_gap(
                model=model,
                image_processor=image_processor,
                encoded_cpu=prepared["encoded"],
                image=prepared["image"],
                token_id=token_id,
                prompt_loss=float(measured["loss"]),
                device=device,
            )
            measured["finite_difference"] = finite_difference_check(
                model=model,
                image_processor=image_processor,
                encoded_cpu=prepared["encoded"],
                image=prepared["image"],
                gradient=array,
                token_id=token_id,
                device=device,
                dtype_name=dtype_name,
            )
        except MemoryStop as exc:
            measured["finite_difference"] = {"status": "not_run", "numerical_check": "not_run", "reason": f"out of memory: {exc}"}
            _save_parent = measured
            del _save_parent
            raise
    gradient_invalid = measured["gradient"]["finite"] is not True or measured["gradient"]["nonzero"] is not True
    if isinstance(measured.get("finite_difference"), dict):
        measured["finite_difference"]["gradient_invalid"] = gradient_invalid
    try:
        _repeated, repeated = differentiate_scene(
            model=model,
            processor=processor,
            image_processor=image_processor,
            image_path=image_path,
            user_text=user_text,
            token_ids=token_ids,
            label=label,
            device=device,
        )
    except MemoryStop as exc:
        measured["stability"] = {"ran": False, "numerical_check": "not_run", "gradient_invalid": True, "reason": f"out of memory: {exc}"}
        raise
    measured["stability"] = stability_report(array, repeated)
    measured["stability"]["gradient_invalid"] = bool(
        gradient_invalid or measured["stability"].get("gradient_invalid")
    )
    del repeated


def _collect_gaps(progress, records, image_processor, root, to_image_01, compare_image) -> None:
    if len(progress.get("processor_gaps", {})) == len(records):
        return
    print("Comparing the differentiable path with the official processor.", flush=True)
    for index, record in enumerate(records, start=1):
        scene_id = record["scene_id"]
        if scene_id in progress["processor_gaps"]:
            continue
        path = root / "fixtures" / "development_v4" / record["image_path"]
        try:
            image = to_image_01(path)
            progress["processor_gaps"][scene_id] = _json_ready(compare_image(image, image_processor))
            del image
        except Exception as exc:
            progress["processor_gaps"][scene_id] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        if index % 16 == 0:
            print(f"Processor comparison {index}/{len(records)}", flush=True)
            gc.collect()
    gc.collect()


def _attach_common(measured: dict[str, Any], stored_row: dict[str, Any], record: dict[str, Any], text_hash: str) -> dict[str, Any]:
    measured["user_text_sha256"] = text_hash
    measured["user_text_sha256_matches_stored"] = True
    measured["stored_chosen"] = stored_row.get("chosen")
    measured["stored_revision"] = stored_row.get("revision")
    summed = measured.get("summed_nll")
    if isinstance(summed, dict):
        measured["decision_matches_stored"] = measured.get("chosen") == stored_row.get("chosen")
        measured["max_abs_nll_gap_vs_stored"] = max(
            abs(float(summed[color]) - float(stored_row["summed_nll"][color])) for color in summed
        )
        measured["stored_margin"] = clean_margin(stored_row["summed_nll"], str(record["answer"]))
    else:
        measured["decision_matches_stored"] = None
        measured["max_abs_nll_gap_vs_stored"] = None
        measured["stored_margin"] = clean_margin(stored_row["summed_nll"], str(record["answer"]))
    measured["attacks_run"] = False
    measured["h_star_fitted"] = False
    return measured


def _oom_record(message: str) -> dict[str, Any]:
    from .records import gradient_record

    return {
        "loss": None,
        "loss_finite": False,
        "loss_identity_agreed": None,
        "summed_nll": None,
        "margin": None,
        "chosen": None,
        "gradient": gradient_record(None, missing_reason=f"out of memory: {message}"),
        "stopped_reason": "out of memory",
    }


def _finalize(root: Path, progress: dict[str, Any]) -> None:
    from dev_pilot.fixtures import load_manifest

    records = load_manifest(root)
    scene_ids = [record["scene_id"] for record in records]
    gate_ids = progress.get("gate_scenes")
    if progress.get("gate"):
        gate = progress["gate"]
    elif progress.get("status") == "stopped_budget":
        gate = {"passed": False, "not_run": True, "reason": progress.get("stopped_reason"), "failures": []}
    else:
        ids = [item["scene_id"] for item in gate_ids] if isinstance(gate_ids, list) else gate_scene_ids(records)
        if all(_scene_done(progress.get("scenes", {}).get(scene_id)) for scene_id in ids):
            gate = judge_gate(progress.get("scenes", {}), ids)
        else:
            gate = {"passed": False, "not_run": True, "reason": progress.get("stopped_reason"), "failures": []}
    progress["gate"] = gate
    rows = load_predictions(root)
    summary = build_summary(
        scenes=progress.get("scenes", {}),
        records=records,
        rows=rows,
        scene_ids=scene_ids,
        gate=gate,
    )
    if progress.get("status") == "stopped_budget":
        reason = progress.get("stopped_reason") or "the pre-load budget refused the load"
        for item in summary["missing_gradients"]:
            item["reason"] = reason
        summary["any_gradient_missing"] = True
        summary["n_groups_eligible"] = None
        summary["eligible"]["n_groups_eligible"] = None
        summary["eligible"]["eligible_group_ids"] = None
        summary["eligible"]["reason"] = "the gradient pass did not run"
    elif progress.get("status") in {"stopped_gate", "stopped_oom", "stopped_loss_identity"}:
        fallback = {
            "stopped_gate": "not attempted because the four-scene gate did not pass",
            "stopped_oom": "not attempted after an out-of-memory stop; the load was not retried",
            "stopped_loss_identity": "not attempted after the labeled-color loss formulas disagreed",
        }[progress["status"]]
        for item in summary["missing_gradients"]:
            if item["reason"] == "not recorded":
                item["reason"] = fallback
        summary["n_groups_eligible"] = None
        summary["eligible"]["n_groups_eligible"] = None
        summary["eligible"]["eligible_group_ids"] = None
        summary["eligible"]["reason"] = "the gradient pass is incomplete"
    summary["n_missing_gradients"] = len(summary["missing_gradients"])
    summary["any_gradient_missing"] = summary["n_missing_gradients"] > 0
    summary["status"] = progress.get("status")
    summary["model_from_pretrained_called"] = progress.get("model_from_pretrained_called")
    summary["peak_memory"] = progress.get("peak_memory")
    summary["load_seconds"] = progress.get("load_seconds")
    summary["processor_gap_summary"] = _gap_summary(progress.get("processor_gaps", {}))
    summary["verification"] = _verification_summary(progress)
    progress["summary"] = summary


def _verification_summary(progress: dict[str, Any]) -> dict[str, Any]:
    scenes = progress.get("scenes", {})
    gaps = []
    mismatches = 0
    compared = 0
    identity_failures = 0
    for block in scenes.values():
        for context in PRIMARY_CONTEXTS:
            entry = block.get(context)
            if not isinstance(entry, dict):
                continue
            if entry.get("loss_identity_agreed") is False:
                identity_failures += 1
            value = entry.get("max_abs_nll_gap_vs_stored")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                gaps.append(float(value))
                compared += 1
            if entry.get("decision_matches_stored") is False:
                mismatches += 1
    return {
        "offline": progress.get("offline_verification"),
        "tokenization": progress.get("tokenization"),
        "loaded_commit_hash": (progress.get("loaded_model") or {}).get("commit_hash"),
        "pinned_revision": PINNED_REVISION,
        "loss_identity_failures": identity_failures,
        "stored_decision_mismatches": mismatches,
        "stored_nll_rows_compared": compared,
        "max_abs_nll_gap_vs_stored": max(gaps) if gaps else None,
        "predictions_path": str(PREDICTION_RELATIVE),
        "predictions_rewritten": False,
    }


def _gap_summary(gaps: dict[str, Any]) -> dict[str, Any]:
    values = []
    outside = []
    for scene_id, gap in gaps.items():
        versus = gap.get("versus_pil") if isinstance(gap, dict) else None
        max_abs = versus.get("max_abs") if isinstance(versus, dict) else None
        if isinstance(max_abs, (int, float)) and not isinstance(max_abs, bool):
            values.append(float(max_abs))
        if isinstance(gap, dict) and gap.get("within_official_match") is not True:
            outside.append(scene_id)
    return {
        "n_scenes": len(gaps),
        "n_outside_official_match": len(outside),
        "outside_official_match": outside,
        "max_abs_vs_pil": max(values) if values else None,
        "mean_of_scene_max_abs_vs_pil": float(sum(values) / len(values)) if values else None,
    }


def _scene_done(block: object) -> bool:
    if not isinstance(block, dict):
        return False
    for context in PRIMARY_CONTEXTS:
        entry = block.get(context)
        if not isinstance(entry, dict) or "gradient" not in entry:
            return False
    return "eligibility_alignment" in block or "undefined_reason" in block


def _track_memory(progress: dict[str, Any], snapshot: dict[str, Any], available: int) -> None:
    peak = progress.setdefault("peak_memory", {})
    peak["peak_rss_mb"] = _max_number(peak.get("peak_rss_mb"), snapshot.get("peak_rss_mb"))
    peak["peak_mps_current_allocated_mb"] = _max_number(
        peak.get("peak_mps_current_allocated_mb"), snapshot.get("mps_current_allocated_mb")
    )
    peak["peak_mps_driver_allocated_mb"] = _max_number(
        peak.get("peak_mps_driver_allocated_mb"), snapshot.get("mps_driver_allocated_mb")
    )
    previous = peak.get("min_os_available_bytes")
    peak["min_os_available_bytes"] = available if previous is None else min(int(previous), int(available))
    peak["last_snapshot"] = snapshot


def _max_number(previous: object, new: object) -> float | None:
    if not isinstance(new, (int, float)) or isinstance(new, bool):
        return previous if isinstance(previous, (int, float)) and not isinstance(previous, bool) else None
    if not isinstance(previous, (int, float)) or isinstance(previous, bool):
        return float(new)
    return float(max(previous, new))


def _write_outputs(root: Path, progress: dict[str, Any]) -> None:
    directory = output_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    summary = progress.get("summary", {})
    text = canonical_json(summary)
    (directory / "summary.json").write_text(text, encoding="utf-8")
    progress["summary_sha256"] = sha256_text(text)
    _save(directory / "progress.json", progress)
    gaps = progress.get("processor_gaps", {})
    (directory / "processor_gaps.json").write_text(canonical_json({"scenes": gaps}), encoding="utf-8")


def _public_status(progress: dict[str, Any]) -> dict[str, Any]:
    summary = progress.get("summary", {})
    gate = summary.get("gate", progress.get("gate", {}))
    return {
        "status": progress.get("status"),
        "gate_passed": gate.get("passed"),
        "gate_not_run": gate.get("not_run"),
        "n_groups_eligible": summary.get("n_groups_eligible"),
        "any_gradient_missing": summary.get("any_gradient_missing"),
        "n_missing_gradients": summary.get("n_missing_gradients"),
        "model_from_pretrained_called": progress.get("model_from_pretrained_called"),
        "stopped_reason": progress.get("stopped_reason"),
    }


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        raise ValueError("refusing to write a raw gradient array")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ValueError("refusing to write a non-finite number")
    return value


def _scratch_path(root: Path, scene_id: str, context: str) -> Path:
    return output_dir(root) / "work" / f"{scene_id}__{context}.npy"


def _save_scratch(root: Path, scene_id: str, context: str, array: np.ndarray) -> None:
    path = _scratch_path(root, scene_id, context)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array, dtype=np.float32))


def _load_scratch(root: Path, scene_id: str, context: str) -> np.ndarray | None:
    path = _scratch_path(root, scene_id, context)
    if not path.is_file():
        return None
    return np.load(path)


def _delete_scratch(root: Path, scene_id: str, context: str) -> None:
    path = _scratch_path(root, scene_id, context)
    if path.is_file():
        path.unlink()


def _protocol_sha256(root: Path) -> str:
    return sha256_text((root / PROTOCOL_RELATIVE).read_text(encoding="utf-8"))


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "checkpoint-1.5" / "run_model_feasibility.py").is_file():
            return candidate
    raise RuntimeError(f"repository root is not above {start}")


def _acquire_resident(root: Path, progress: dict[str, Any]) -> None:
    others = _other_model_processes()
    if others:
        raise StopRun("stopped_resident", "another scoring process is already running: " + others[0])
    legacy = root / LEGACY_RESIDENT
    if legacy.is_file() and _resident_alive(legacy):
        raise StopRun("stopped_resident", f"a model is already resident according to {legacy}")
    path = root / OUTPUT_RELATIVE / "resident.json"
    if path.is_file():
        if _resident_alive(path):
            raise StopRun("stopped_resident", f"a model is already resident according to {path}")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "model_key": MODEL_KEY}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    progress["resident_pid"] = os.getpid()


def _release_resident(root: Path) -> None:
    path = root / OUTPUT_RELATIVE / "resident.json"
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if payload.get("pid") == os.getpid():
        path.unlink()


def _resident_alive(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _other_model_processes() -> list[str]:
    me = os.getpid()
    completed = subprocess.run(["ps", "-ax", "-o", "pid=,command="], check=True, capture_output=True, text=True)
    hits = []
    markers = (" -m dev_pilot", "qwen_rgb.measure", " -m qwen_rgb_loss", "qwen_rgb_loss.run")
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        pid_text, _, command = text.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == me:
            continue
        if any(marker in command for marker in markers):
            hits.append(text)
    return hits
