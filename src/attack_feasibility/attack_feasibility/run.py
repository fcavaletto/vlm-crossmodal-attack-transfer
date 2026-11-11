"""Guarded Qwen load and the Checkpoint 1.9 feasibility steps.

One process calls ``from_pretrained`` at most once, and only after the
Checkpoint 1.5 budget permits it. The script does not lower that budget and
does not fit H*.
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
from qwen_rgb.pins import MODEL_ID, PINNED_REVISION
from qwen_rgb_loss.protocol import MODEL_KEY, assert_unchanged_lines

from .judge import judge_stage_a, judge_stage_b
from .protocol import (
    LABEL,
    OUTPUT_RELATIVE,
    PROTOCOL_RELATIVE,
    STAGE_A_BUDGETS,
    STAGE_A_SAMPLE_LABEL,
    STAGE_A_SCENES,
    STARTED_FROM_COMMIT,
    TERMINAL_STATUSES,
    output_dir,
)
from .select import assert_frozen_selection, load_prior_progress
from .stage_a import StageStop, run_scene as run_stage_a_scene
from .stage_b import run_scene as run_stage_b_scene

LEGACY_RESIDENTS = (
    Path("docs/checkpoint-1.7/development_v4/resident.json"),
    Path("docs/checkpoint-1.8/qwen_rgb_loss/resident.json"),
)


def main() -> None:
    root = _repo_root(Path(__file__).resolve())
    started = time.perf_counter()
    payload = execute(root)
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    _save(output_dir(root) / "progress.json", payload)
    print(json.dumps(_public_status(payload), indent=2, sort_keys=True), flush=True)


def execute(root: Path) -> dict[str, Any]:
    path = output_dir(root) / "progress.json"
    progress = _load(path)
    if progress.get("status") in TERMINAL_STATUSES:
        if progress.get("status") != "completed":
            progress["resume_refused"] = "the saved run already stopped; the model is not loaded again"
        return progress
    try:
        _execute(root, progress, path)
    except StageStop as exc:
        progress["status"] = exc.status
        progress["stopped_reason"] = str(exc)
        progress["stage_b_ran"] = False
    except Exception as exc:
        progress["status"] = "error"
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        progress["stage_b_ran"] = False
        _finalize(progress)
        _save(path, progress)
        _release_resident(root)
        raise
    _finalize(progress)
    _save(path, progress)
    _release_resident(root)
    return progress


def _execute(root: Path, progress: dict[str, Any], path: Path) -> None:
    import torch
    from dev_pilot.fixtures import load_manifest
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.budget import dtype_decisions, load_budget_module, select_dtype
    from qwen_rgb.measure import (
        bf16_probe,
        load_model_after_budget,
        load_processor,
        prepare_environment,
        read_local_header,
        weight_file,
    )
    from qwen_rgb.rgb_path import assert_pinned_processor
    from qwen_rgb_loss.offline import load_predictions
    from vlm_iface.devices import available_bytes, memory_snapshot

    protocol_hash = _sha256(root / PROTOCOL_RELATIVE)
    if progress.get("protocol_sha256") and progress["protocol_sha256"] != protocol_hash:
        raise StageStop("stopped_protocol", "The attack protocol file changed. Refusing to resume under a new rule.")
    assert_unchanged_lines()
    selection = assert_frozen_selection(root)
    progress.setdefault("status", "running")
    progress["label"] = LABEL
    progress["started_from_commit"] = STARTED_FROM_COMMIT
    progress["protocol_commit"] = _protocol_commit(root)
    progress["protocol_sha256"] = protocol_hash
    progress["git_head_at_start"] = progress.get("git_head_at_start") or _git_head(root)
    progress["stage_a_rule"] = selection["stage_a_rule"]
    progress["stage_b_rule"] = selection["stage_b_rule"]
    progress["stage_a_sample"] = STAGE_A_SAMPLE_LABEL
    progress["stage_a_unbiased_population_attack_success_rate"] = False
    progress["h_star_fitted"] = False
    progress["held_out_scenes_opened"] = False
    progress["model_from_pretrained_called"] = bool(progress.get("model_from_pretrained_called"))
    progress["environment"] = prepare_environment()
    progress.setdefault("stage_a_scenes", {})
    progress.setdefault("stage_b_ran", False)
    _acquire_resident(root, progress)
    _save(path, progress)

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
        raise StageStop("stopped_tokenization", "A locked color continuation is not one token.")
    budget = load_budget_module(root)
    probe = bf16_probe(torch)
    progress["bf16_probe"] = probe
    weights = weight_file()
    if not weights.is_file():
        raise StageStop("stopped_missing_weights", "The local safetensors file is absent. This script does not download it.")
    header = read_local_header(weights)
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS:
        raise StageStop("stopped_unexpected_header", "The header parameter count is not the Checkpoint 1.5 Qwen count.")
    if header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
        raise StageStop("stopped_unexpected_header", "The header float32 expansion is not the Checkpoint 1.5 Qwen size.")
    if header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        raise StageStop("stopped_unexpected_header", "The header is not exclusively BF16.")
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
    chosen = select_dtype(decisions, bf16_probe_ok=bool(probe.get("matmul_ok")))
    progress["budget_recheck_immediately_before_load"] = decisions
    progress["dtype_selection_immediately_before_load"] = chosen
    progress["available_bytes_entering_decision"] = available
    progress["available_bytes_at_decision"] = available_bytes()
    progress["os_floor_bytes"] = OS_FLOOR_BYTES
    if not chosen["permitted"] or chosen["dtype"] is None:
        progress["model_from_pretrained_called"] = False
        _save(path, progress)
        required = decisions["bfloat16"]["required_available_bytes"]
        raise StageStop(
            "stopped_budget",
            chosen["reason"]
            + f" Available bytes entering the decision: {available}. "
            + f"Fresh available bytes: {progress['available_bytes_at_decision']}. "
            + f"BF16 required bytes: {required}. The guard was not lowered.",
        )

    dtype_name = str(chosen["dtype"])
    device = torch.device("mps")
    if progress.get("model_from_pretrained_called") and progress.get("status") == "running" and progress.get("load_seconds"):
        # A previous process already loaded and then died. This process may load once.
        pass
    progress["model_from_pretrained_called"] = True
    _save(path, progress)
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
            "parameters_require_grad": any(item.requires_grad for item in model.parameters()),
        }
        _track(progress, memory_snapshot(device), available_bytes())
        if parameter.device.type != "mps" or str(parameter.dtype) != f"torch.{dtype_name}":
            raise StageStop("stopped_unexpected_load", "The loaded parameters are not the permitted dtype on MPS.")
        if commit not in (None, PINNED_REVISION):
            raise StageStop("stopped_revision", f"Loaded commit {commit} is not {PINNED_REVISION}.")
        _save(path, progress)
        records = {row["scene_id"]: row for row in load_manifest(root)}
        stored = {(row["scene_id"], row["context"]): row for row in load_predictions(root)}
        prior = load_prior_progress(root)["scenes"]
        _run_stage_a(progress, path, root, records, stored, prior, selection, model, processor, image_processor, token_ids, device)
        judgment = judge_stage_a(progress["stage_a_scenes"])
        progress["stage_a_judgment"] = judgment
        _save(path, progress)
        if not judgment["passed"]:
            progress["stage_b_ran"] = False
            progress["status"] = "stopped_stage_a"
            progress["stopped_reason"] = (
                "Positive-sign steps did not show a meaningful advantage over both controls at every budget. "
                "Stage B was not started."
            )
            return
        _run_stage_b(progress, path, root, records, stored, prior, selection, model, processor, image_processor, token_ids, device)
        progress["stage_b_judgment"] = judge_stage_b(list(progress["stage_b_scenes"].values()))
        progress["stage_b_ran"] = True
        progress["status"] = "completed"
        if progress["stage_b_judgment"]["ineffective_at_every_listed_budget"]:
            progress["stopped_reason"] = (
                "PGD was ineffective at every listed budget. Epsilon was not raised and the objective was not changed."
            )
        else:
            progress["stopped_reason"] = "The listed Stage B budgets finished. No further epsilon was added."
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _run_stage_a(progress, path, root, records, stored, prior, selection, model, processor, image_processor, token_ids, device) -> None:
    from vlm_iface.devices import available_bytes, memory_snapshot

    scenes = progress["stage_a_scenes"]
    for index, row in enumerate(selection["stage_a"], start=1):
        scene_id = row["scene_id"]
        if scenes.get(scene_id, {}).get("finished") is True:
            print(f"Stage A {scene_id} already finished ({index}/8)", flush=True)
            continue
        print(f"Stage A {scene_id} ({index}/8)", flush=True)
        measured = run_stage_a_scene(
            root=root,
            model=model,
            processor=processor,
            image_processor=image_processor,
            record=records[scene_id],
            selection=row,
            stored_row=stored[(scene_id, "neutral")],
            prior_gradient_sha256=prior[scene_id]["neutral"]["gradient"].get("sha256"),
            token_ids=token_ids,
            device=device,
        )
        scenes[scene_id] = measured
        _track(progress, memory_snapshot(device), available_bytes())
        _save(path, progress)


def _run_stage_b(progress, path, root, records, stored, prior, selection, model, processor, image_processor, token_ids, device) -> None:
    from vlm_iface.devices import available_bytes, memory_snapshot

    scenes = progress.setdefault("stage_b_scenes", {})
    ids = selection["stage_b"]
    for index, scene_id in enumerate(ids, start=1):
        if scenes.get(scene_id, {}).get("finished") is True:
            print(f"Stage B {scene_id} already finished ({index}/{len(ids)})", flush=True)
            continue
        print(f"Stage B {scene_id} ({index}/{len(ids)})", flush=True)
        measured = run_stage_b_scene(
            root=root,
            model=model,
            processor=processor,
            image_processor=image_processor,
            record=records[scene_id],
            stored_rows=stored,
            prior=prior[scene_id],
            token_ids=token_ids,
            device=device,
        )
        scenes[scene_id] = measured
        _track(progress, memory_snapshot(device), available_bytes())
        _save(path, progress)


def _finalize(progress: dict[str, Any]) -> None:
    scenes = progress.get("stage_a_scenes") or {}
    if scenes and "stage_a_judgment" not in progress:
        progress["stage_a_judgment"] = judge_stage_a(scenes)
    progress["stage_a_description"] = _describe_stage_a(scenes)
    progress["checkpoint_1_8_finite_difference"] = {
        "kept": True,
        "replaced_with_a_passing_result": False,
        "note": (
            "Checkpoint 1.8 confirmed one of eight gate pairs, and only at the smaller step. "
            "This checkpoint does not rerun that check to overturn it."
        ),
    }
    if progress.get("stage_b_scenes") and "stage_b_judgment" not in progress and progress.get("stage_b_ran") is True:
        progress["stage_b_judgment"] = judge_stage_b(list(progress["stage_b_scenes"].values()))
    progress["summary_counts"] = {
        "stage_a_scenes_finished": sum(1 for row in scenes.values() if row.get("finished") is True),
        "stage_a_scene_ids": [scene_id for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES],
        "stage_b_ran": bool(progress.get("stage_b_ran")),
        "h_star_fitted": False,
    }


def _describe_stage_a(scenes: dict[str, Any]) -> dict[str, Any]:
    """Counts for the review. These counts do not change the stop rule."""
    described: dict[str, Any] = {}
    for key, _numerator in STAGE_A_BUDGETS:
        positive_up = 0
        agreement = 0
        uint8_up = 0
        answer_changes = 0
        uint8_answer_changes = 0
        n = 0
        for scene in scenes.values():
            block = scene.get("budgets", {}).get(key, {})
            step = block.get("positive")
            if not isinstance(step, dict):
                continue
            n += 1
            observed = step.get("observed_change")
            if isinstance(observed, (int, float)) and not isinstance(observed, bool) and observed > 0:
                positive_up += 1
            if step.get("directional_agreement") is True:
                agreement += 1
            uint8_observed = step.get("uint8", {}).get("observed_change")
            if isinstance(uint8_observed, (int, float)) and not isinstance(uint8_observed, bool) and uint8_observed > 0:
                uint8_up += 1
            if step.get("answer_changed") is True:
                answer_changes += 1
            if step.get("uint8", {}).get("answer_changed") is True:
                uint8_answer_changes += 1
        described[key] = {
            "n_positive_records": n,
            "n_float_loss_increased": positive_up,
            "n_directional_agreement": agreement,
            "n_uint8_loss_increased": uint8_up,
            "n_float_answer_changed": answer_changes,
            "n_uint8_answer_changed": uint8_answer_changes,
        }
    hashes = [
        scene.get("gradient", {}).get("sha256_matches_checkpoint_1_8")
        for scene in scenes.values()
        if isinstance(scene.get("gradient"), dict)
    ]
    described["gradient_sha256_matches_checkpoint_1_8"] = {
        "n_recorded": len(hashes),
        "n_match": sum(1 for item in hashes if item is True),
        "n_mismatch": sum(1 for item in hashes if item is False),
    }
    return described


def _public_status(progress: dict[str, Any]) -> dict[str, Any]:
    judgment = progress.get("stage_a_judgment") or {}
    return {
        "status": progress.get("status"),
        "stopped_reason": progress.get("stopped_reason"),
        "stage_b_ran": progress.get("stage_b_ran"),
        "stage_a_passed": judgment.get("passed"),
        "protocol_commit": progress.get("protocol_commit"),
        "h_star_fitted": False,
        "available_bytes_entering_decision": progress.get("available_bytes_entering_decision"),
        "model_from_pretrained_called": progress.get("model_from_pretrained_called"),
    }


def _track(progress: dict[str, Any], snapshot: dict[str, Any], available: int) -> None:
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
    others = _other_model_processes()
    if others:
        raise StageStop("stopped_resident", "another scoring process is already running: " + others[0])
    for relative in LEGACY_RESIDENTS:
        path = root / relative
        if path.is_file() and _resident_alive(path):
            raise StageStop("stopped_resident", f"a model is already resident according to {path}")
    path = root / OUTPUT_RELATIVE / "resident.json"
    if path.is_file():
        if _resident_alive(path):
            raise StageStop("stopped_resident", f"a model is already resident according to {path}")
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


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "status": payload.get("status"),
        "stopped_reason": payload.get("stopped_reason"),
        "stage_b_ran": payload.get("stage_b_ran"),
        "stage_a_judgment": payload.get("stage_a_judgment"),
        "stage_a_description": payload.get("stage_a_description"),
        "stage_b_judgment": payload.get("stage_b_judgment"),
        "summary_counts": payload.get("summary_counts"),
        "protocol_commit": payload.get("protocol_commit"),
        "protocol_sha256": payload.get("protocol_sha256"),
        "loaded_model": payload.get("loaded_model"),
        "budget_recheck_immediately_before_load": payload.get("budget_recheck_immediately_before_load"),
        "dtype_selection_immediately_before_load": payload.get("dtype_selection_immediately_before_load"),
        "available_bytes_entering_decision": payload.get("available_bytes_entering_decision"),
        "available_bytes_at_decision": payload.get("available_bytes_at_decision"),
        "peak_memory": payload.get("peak_memory"),
        "load_seconds": payload.get("load_seconds"),
        "wall_seconds": payload.get("wall_seconds"),
        "checkpoint_1_8_finite_difference": payload.get("checkpoint_1_8_finite_difference"),
        "h_star_fitted": False,
        "model_from_pretrained_called": payload.get("model_from_pretrained_called"),
        "stage_a_sample": payload.get("stage_a_sample"),
    }
    (path.parent / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = float(value) if isinstance(value, float) else value
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return value
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _protocol_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(PROTOCOL_RELATIVE)],
        cwd=root,
        text=True,
    ).strip()


def _repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "checkpoint-1.9" / "ATTACK_PROTOCOL.md").is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 1.9 protocol is not above {start}")
