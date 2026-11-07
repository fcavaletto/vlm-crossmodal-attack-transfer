"""Guarded Qwen load and the Checkpoint 2.0 calibration grid.

One process calls ``from_pretrained`` at most once, and only after the
Checkpoint 1.5 bfloat16 budget permits it. The script does not lower that
budget, does not add an epsilon, and does not fit H*.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from qwen_rgb.budget import OS_FLOOR_BYTES

from .counts import configuration_counts, flatten_scenes
from .protocol import (
    BLOCKING_STATUSES,
    BUDGETS,
    LABEL,
    OUTPUT_RELATIVE,
    PGD_STEPS,
    PROTOCOL_RELATIVE,
    RECORDED_STEPS,
    RULE_ID,
    SAMPLE_LABEL,
    SEED_PREFIX,
    SOURCE_CONTEXT,
    STARTED_FROM_COMMIT,
    CalibrationStop,
    assert_protocol_committed,
    output_dir,
)
from .report import checkpoint_1_8_description, file_sha256, write_report
from .select import assert_frozen_selection, select_calibration

LEGACY_RESIDENTS = (
    Path("docs/checkpoint-1.7/development_v4/resident.json"),
    Path("docs/checkpoint-1.8/qwen_rgb_loss/resident.json"),
    Path("docs/checkpoint-1.9/attack_feasibility/resident.json"),
)


def main() -> None:
    root = _repo_root(Path(__file__).resolve())
    started = time.perf_counter()
    payload = execute(root)
    if not payload.get("resume_refused"):
        payload["wall_seconds"] = round(time.perf_counter() - started, 3)
        _persist(root, payload)
    print(json.dumps(_public_status(payload), indent=2, sort_keys=True), flush=True)


def execute(root: Path) -> dict[str, Any]:
    path = output_dir(root) / "progress.json"
    progress = _load(path)
    if progress.get("status") in BLOCKING_STATUSES:
        progress["resume_refused"] = "the saved run already stopped; the model is not loaded again"
        return progress
    try:
        _execute(root, progress)
    except CalibrationStop as exc:
        progress["status"] = exc.status
        progress["stopped_reason"] = str(exc)
    except Exception as exc:
        progress["status"] = "error"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        _persist(root, progress)
        _release_resident(root)
        raise
    _persist(root, progress)
    _release_resident(root)
    return progress


def _execute(root: Path, progress: dict[str, Any]) -> None:
    import torch
    from dev_pilot.fixtures import load_manifest
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.budget import dtype_decisions, load_budget_module
    from qwen_rgb.measure import (
        bf16_probe,
        load_model_after_budget,
        load_processor,
        prepare_environment,
        read_local_header,
        weight_file,
    )
    from qwen_rgb.pins import MODEL_ID, PINNED_REVISION
    from qwen_rgb.rgb_path import assert_pinned_processor
    from qwen_rgb_loss.offline import load_predictions
    from qwen_rgb_loss.protocol import MODEL_KEY, assert_unchanged_lines
    from vlm_iface.devices import available_bytes, memory_snapshot, synchronize

    from .scene import run_scene

    protocol_commit = assert_protocol_committed(root)
    protocol_hash = hashlib.sha256((root / PROTOCOL_RELATIVE).read_bytes()).hexdigest()
    if progress.get("protocol_sha256") and progress["protocol_sha256"] != protocol_hash:
        raise CalibrationStop("stopped_protocol", "The calibration protocol file changed. Refusing to resume under a new rule.")
    assert_unchanged_lines()
    records = {row["scene_id"]: row for row in load_manifest(root)}
    selected = select_calibration(list(records.values()))
    progress["label"] = LABEL
    progress["checkpoint"] = "checkpoint-2.0"
    progress["started_from_commit"] = STARTED_FROM_COMMIT
    progress["protocol_commit"] = protocol_commit
    progress["protocol_sha256"] = protocol_hash
    progress["git_head_at_start"] = progress.get("git_head_at_start") or _git_head(root)
    progress["selection_rule"] = RULE_ID
    progress["sample_label"] = SAMPLE_LABEL
    progress["unbiased_population_attack_success_rate"] = False
    progress["n_groups"] = selected["n_groups"]
    progress["group_ids_found"] = selected["group_ids"]
    progress["h_star_fitted"] = False
    progress["held_out_scenes_opened"] = False
    progress["alignment_used_to_choose_configuration"] = False
    progress["epsilon_raised"] = False
    progress["steps_added"] = False
    progress["objective_changed"] = False
    progress["source_context"] = SOURCE_CONTEXT
    progress["budgets"] = [key for key, _numerator in BUDGETS]
    progress["pgd_steps"] = PGD_STEPS
    progress["recorded_steps"] = list(RECORDED_STEPS)
    progress["seed_prefix"] = SEED_PREFIX
    progress["float32_loaded"] = False
    progress["os_floor_bytes"] = OS_FLOOR_BYTES
    if selected.get("ok") is not True:
        progress["attacks_started"] = False
        progress["model_from_pretrained_called"] = False
        status = "stopped_group_count" if selected.get("n_groups") != 48 else "stopped_scene_rule"
        raise CalibrationStop(status, str(selected.get("reason")))
    assert_frozen_selection(selected)
    scene_ids = [row["scene_id"] for row in selected["scenes"]]
    progress["selected_group_ids"] = list(selected["selected_group_ids"])
    progress["scene_ids"] = scene_ids
    progress["n_scenes"] = len(scene_ids)
    progress.setdefault("scenes", {})
    progress.setdefault("budget_decision_history", [])
    progress["environment"] = prepare_environment()
    _acquire_resident(root, progress)
    progress["status"] = "running"
    _persist(root, progress)

    if any(entry.get("pid") == os.getpid() and entry.get("permitted") is False for entry in progress["budget_decision_history"]):
        raise CalibrationStop(
            "stopped_budget",
            "This process already recorded a budget refusal. It was not reread.",
        )

    print("Loading the processor. Model weights stay on disk.", flush=True)
    processor, image_processor = load_processor()
    assert_pinned_processor(image_processor)
    identity = assert_locked_tokenization(processor, MODEL_KEY)
    progress["tokenization"] = {
        "answer_prefix": identity["answer_prefix"],
        "token_ids": identity["token_ids"],
    }
    token_ids = {color: tuple(int(token) for token in ids) for color, ids in identity["token_ids"].items()}
    if any(len(ids) != 1 for ids in token_ids.values()):
        raise CalibrationStop("stopped_tokenization", "A locked color continuation is not one token.")
    expected = {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)}
    if token_ids != expected:
        raise CalibrationStop("stopped_tokenization", "The processor color ids are not the locked bare-color ids.")
    stored = {(row["scene_id"], row["context"]): row for row in load_predictions(root)}
    _preflight_prompts(records, stored, scene_ids)

    budget = load_budget_module(root)
    probe = bf16_probe(torch)
    progress["bf16_probe"] = probe
    weights = weight_file()
    if not weights.is_file():
        raise CalibrationStop("stopped_missing_weights", "The local safetensors file is absent. This script does not download it.")
    header = read_local_header(weights)
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS:
        raise CalibrationStop("stopped_unexpected_header", "The header parameter count is not the Checkpoint 1.5 Qwen count.")
    if header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
        raise CalibrationStop("stopped_unexpected_header", "The header float32 expansion is not the Checkpoint 1.5 Qwen size.")
    if header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        raise CalibrationStop("stopped_unexpected_header", "The header is not exclusively BF16.")
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
    permitted = bool(decisions["bfloat16"].get("permitted")) and bool(probe.get("matmul_ok"))
    reason = _dtype_reason(decisions, probe_ok=bool(probe.get("matmul_ok")))
    fresh = available_bytes()
    progress.setdefault("budget_decision_history", []).append(
        {
            "pid": os.getpid(),
            "available_bytes_entering_decision": available,
            "available_bytes_at_decision": fresh,
            "permitted": permitted,
            "reason": reason,
            "float32_loaded": False,
            "guard_lowered": False,
            "model_from_pretrained_called": False,
            "budget_recheck_immediately_before_load": decisions,
        }
    )
    progress["budget_recheck_immediately_before_load"] = decisions
    progress["available_bytes_entering_decision"] = available
    progress["available_bytes_at_decision"] = fresh
    progress["float32_loaded"] = False
    progress["explicit_recheck"] = (
        "A later process may read the same Checkpoint 1.5 lines again. "
        "A refusal stays in budget_decision_history. The process that recorded it does not reread it."
    )
    if not permitted:
        progress["model_from_pretrained_called"] = False
        progress["attacks_started"] = False
        _persist(root, progress)
        required = decisions["bfloat16"]["required_available_bytes"]
        raise CalibrationStop(
            "stopped_budget",
            reason
            + f" Available bytes entering the decision: {available}. "
            + f"Fresh available bytes: {fresh}. "
            + f"BF16 required bytes: {required}. The guard was not lowered.",
        )

    loads = progress.setdefault("load_pids", [])
    if os.getpid() in loads:
        raise CalibrationStop("stopped_unexpected_load", "This process already called from_pretrained.")
    loads.append(os.getpid())
    progress["model_from_pretrained_called"] = True
    _persist(root, progress)
    print("Loading Qwen once, bfloat16, eager attention.", flush=True)
    started = time.perf_counter()
    model = load_model_after_budget(decisions["bfloat16"], "bfloat16")
    try:
        device = torch.device("mps")
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
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
            "attn_implementation": "eager",
            "parameters_require_grad": any(item.requires_grad for item in model.parameters()),
            "float32_loaded": False,
        }
        _track(progress, memory_snapshot(device), available_bytes())
        if parameter.device.type != "mps" or str(parameter.dtype) != "torch.bfloat16":
            raise CalibrationStop(
                "stopped_unexpected_load",
                "The loaded parameters are not bfloat16 on MPS. A CPU load was not kept.",
            )
        if commit not in (None, PINNED_REVISION):
            raise CalibrationStop("stopped_revision", f"Loaded commit {commit} is not {PINNED_REVISION}.")
        progress["attacks_started"] = True
        _persist(root, progress)
        _run_scenes(
            progress,
            root,
            records,
            stored,
            scene_ids,
            model,
            processor,
            image_processor,
            token_ids,
            device,
            run_scene,
        )
        progress["status"] = "completed"
        progress["stopped_reason"] = (
            "The listed budgets and recorded steps finished. Epsilon was not raised and no step was added."
        )
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _dtype_reason(decisions: dict[str, Any], *, probe_ok: bool) -> str:
    bf16 = decisions["bfloat16"]
    if not bf16.get("permitted"):
        return str(bf16.get("reason"))
    if not probe_ok:
        return "The BF16 budget passed, but the MPS BF16 probe did not. from_pretrained was not called."
    return "Workstream A loads bfloat16. The BF16 budget passed and the MPS probe ran. Float32 was not loaded."


def _preflight_prompts(records: dict[str, dict[str, Any]], stored: dict[tuple[str, str], dict[str, Any]], scene_ids: list[str]) -> None:
    from choice_eval.prompt import build_prompt
    from dev_pilot.rows import user_text_sha256

    from .protocol import EVAL_CONTEXTS

    for scene_id in scene_ids:
        record = records[scene_id]
        if record.get("split") != "development":
            raise CalibrationStop("stopped_scene_rule", f"{scene_id} is not a development scene.")
        for context in EVAL_CONTEXTS:
            user_text = build_prompt(record["contexts"][context]["text"], record["question"]).user_text()
            row = stored.get((scene_id, context))
            if row is None or row.get("user_text_sha256") != user_text_sha256(user_text):
                raise CalibrationStop(
                    "stopped_prompt",
                    f"{scene_id} {context} prompt hash does not match the stored row.",
                )


def _run_scenes(progress, root, records, stored, scene_ids, model, processor, image_processor, token_ids, device, run_scene) -> None:
    from vlm_iface.devices import available_bytes, memory_snapshot

    scenes = progress["scenes"]
    for index, scene_id in enumerate(scene_ids, start=1):
        if scenes.get(scene_id, {}).get("finished") is True:
            print(f"skip {scene_id} ({index}/{len(scene_ids)})", flush=True)
            continue
        print(f"calibrate {scene_id} ({index}/{len(scene_ids)})", flush=True)
        measured = run_scene(
            root=root,
            model=model,
            processor=processor,
            image_processor=image_processor,
            record=records[scene_id],
            stored_rows=stored,
            token_ids=token_ids,
            device=device,
        )
        scenes[scene_id] = measured
        _track(progress, memory_snapshot(device), available_bytes())
        _persist(root, progress)
        if measured.get("fatal_status"):
            raise CalibrationStop(str(measured["fatal_status"]), str(measured.get("fatal_reason")))


def _persist(root: Path, progress: dict[str, Any]) -> None:
    scenes = progress.get("scenes") or {}
    scene_ids = list(progress.get("scene_ids") or [])
    finished = [scene_id for scene_id in scene_ids if scenes.get(scene_id, {}).get("finished") is True]
    progress["n_finished"] = len(finished)
    progress["n_scenes"] = progress.get("n_scenes", len(scene_ids))
    progress["scene_seconds_sum"] = round(
        sum(float(scenes[scene_id].get("seconds") or 0.0) for scene_id in finished),
        3,
    )
    rows = flatten_scenes(scenes, scene_ids)
    progress["configurations"] = configuration_counts(rows)
    progress["h_star_fitted"] = False
    progress["alignment_used_to_choose_configuration"] = False
    directory = output_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "progress.json", progress)
    _write_jsonl(directory / "records.jsonl", rows)
    summary = _summary(progress)
    _write_json(directory / "summary.json", summary)
    description: list[dict[str, Any]] = []
    if scene_ids:
        try:
            description = checkpoint_1_8_description(root, scene_ids)
        except Exception as exc:
            description = []
            progress["checkpoint_1_8_description_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(directory / "progress.json", progress)
    _write_json(directory / "checkpoint_1_8_description.json", description)
    rendered = dict(progress)
    rendered["file_sha256"] = {
        "docs/checkpoint-2.0/calibration/progress.json": file_sha256(directory / "progress.json"),
        "docs/checkpoint-2.0/calibration/summary.json": file_sha256(directory / "summary.json"),
        "docs/checkpoint-2.0/calibration/records.jsonl": file_sha256(directory / "records.jsonl"),
        "docs/checkpoint-2.0/calibration/checkpoint_1_8_description.json": file_sha256(
            directory / "checkpoint_1_8_description.json"
        ),
    }
    write_report(root, rendered, description)


def _summary(progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": progress.get("status"),
        "stopped_reason": progress.get("stopped_reason"),
        "protocol_commit": progress.get("protocol_commit"),
        "protocol_sha256": progress.get("protocol_sha256"),
        "n_groups": progress.get("n_groups"),
        "n_scenes": progress.get("n_scenes"),
        "n_finished": progress.get("n_finished"),
        "configurations": progress.get("configurations"),
        "loaded_model": progress.get("loaded_model"),
        "available_bytes_entering_decision": progress.get("available_bytes_entering_decision"),
        "available_bytes_at_decision": progress.get("available_bytes_at_decision"),
        "budget_recheck_immediately_before_load": progress.get("budget_recheck_immediately_before_load"),
        "peak_memory": progress.get("peak_memory"),
        "load_seconds": progress.get("load_seconds"),
        "scene_seconds_sum": progress.get("scene_seconds_sum"),
        "wall_seconds": progress.get("wall_seconds"),
        "float32_loaded": False,
        "epsilon_raised": False,
        "steps_added": False,
        "objective_changed": False,
        "h_star_fitted": False,
        "held_out_scenes_opened": False,
        "alignment_used_to_choose_configuration": False,
        "model_from_pretrained_called": progress.get("model_from_pretrained_called"),
        "attacks_started": progress.get("attacks_started"),
    }


def _public_status(progress: dict[str, Any]) -> dict[str, Any]:
    counts = []
    for row in progress.get("configurations") or []:
        counts.append(
            {
                "configuration": row.get("configuration"),
                "source_success": [row.get("n_source_success"), row.get("n_source_success_denominator")],
                "conditional_transfer": [
                    row.get("n_conditional_transfer"),
                    row.get("n_conditional_transfer_denominator"),
                ],
                "adequate_source_and_transfer_variation": row.get("adequate_source_and_transfer_variation"),
            }
        )
    return {
        "status": progress.get("status"),
        "stopped_reason": progress.get("stopped_reason"),
        "protocol_commit": progress.get("protocol_commit"),
        "n_finished": progress.get("n_finished"),
        "n_scenes": progress.get("n_scenes"),
        "counts": counts,
        "h_star_fitted": False,
        "available_bytes_entering_decision": progress.get("available_bytes_entering_decision"),
        "model_from_pretrained_called": progress.get("model_from_pretrained_called"),
    }


def _track(progress: dict[str, Any], snapshot: dict[str, Any], available: int) -> None:
    if snapshot.get("device") != "mps":
        raise CalibrationStop("stopped_unexpected_load", "A memory snapshot was not from MPS and was not relabeled.")
    peak = progress.setdefault("peak_memory", {})
    for source, dest in (
        ("peak_rss_mb", "peak_rss_mb"),
        ("mps_current_allocated_mb", "peak_mps_current_allocated_mb"),
        ("mps_driver_allocated_mb", "peak_mps_driver_allocated_mb"),
    ):
        peak[dest] = _max_number(peak.get(dest), snapshot.get(source))
    previous = peak.get("min_os_available_bytes")
    peak["min_os_available_bytes"] = available if previous is None else min(int(previous), int(available))
    peak["last_snapshot"] = snapshot


def _max_number(previous: object, new: object) -> float | None:
    if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(float(new)):
        return float(previous) if isinstance(previous, (int, float)) and not isinstance(previous, bool) else None
    if isinstance(previous, bool) or not isinstance(previous, (int, float)):
        return float(new)
    return float(max(previous, new))


def _acquire_resident(root: Path, progress: dict[str, Any]) -> None:
    from qwen_rgb_loss.protocol import MODEL_KEY

    others = _other_model_processes()
    if others:
        raise CalibrationStop("stopped_resident", "another scoring process is already running: " + others[0])
    for relative in LEGACY_RESIDENTS:
        path = root / relative
        if path.is_file() and _resident_alive(path):
            raise CalibrationStop("stopped_resident", f"a model is already resident according to {path}")
    path = root / OUTPUT_RELATIVE / "resident.json"
    if path.is_file():
        if _resident_alive(path):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if int(payload.get("pid", -1)) != os.getpid():
                raise CalibrationStop("stopped_resident", f"a model is already resident according to {path}")
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "model_key": MODEL_KEY}) + "\n", encoding="utf-8")
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
    markers = (
        " -m dev_pilot",
        "qwen_rgb.measure",
        " -m qwen_rgb_loss",
        "qwen_rgb_loss.run",
        " -m attack_feasibility",
        "attack_feasibility.run",
        " -m attack_calibration",
        "attack_calibration.run",
    )
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


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(_json_ready(row), sort_keys=True, allow_nan=False) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return value


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / PROTOCOL_RELATIVE).is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 2.0 protocol is not above {start}")
