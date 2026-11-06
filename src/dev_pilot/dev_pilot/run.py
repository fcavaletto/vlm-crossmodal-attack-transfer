"""Score development_v4 with one model. No attacks and no second model.

The protocol file must already be an ancestor of HEAD. This process loads
either SmolVLM2 or Qwen3-VL-2B, never both. A Qwen budget refusal is recorded
and is not retried.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

from choice_eval.models import require_model
from choice_eval.prompt import build_prompt
from scene_v4.paths import resolve_image_path

from .fixtures import load_manifest, verify_checksums
from .lock import (
    DO_IMAGE_SPLITTING,
    EXTRA_CONTEXTS,
    LABEL,
    MANIFEST_RELATIVE,
    MAX_NEW_TOKENS,
    OUTPUT_RELATIVE,
    PRIMARY_CONTEXTS,
    REQUIRED_HF_HOME,
)
from .provenance import protocol_provenance
from .resident import acquire, release
from .rows import assemble_row, protocol_errors
from .scoring import assert_locked_tokenization, nlls_from_first_logit
from .summarize import summarize_model

Forward = Callable[[Path, str], tuple[dict[str, float], dict[str, float], str, bool]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean development_v4 pilot for one model.")
    parser.add_argument("--model", required=True, choices=("smolvlm2", "qwen3-vl-2b"))
    args = parser.parse_args(argv)
    _prepare_cache()
    root = repo_root()
    try:
        return execute(root, args.model)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", flush=True)
        return 1


def execute(root: Path, model_key: str) -> int:
    provenance = protocol_provenance(root)
    print(
        "protocol_commit={protocol_commit} ancestor={protocol_commit_is_ancestor} "
        "predictions_at_protocol_commit={predictions_present_at_protocol_commit}".format(**provenance),
        flush=True,
    )
    if provenance["checkpoint_1_6_diff_empty"] is not True:
        raise RuntimeError("docs/checkpoint-1.6 differs from the Checkpoint 1.6 review commit")
    verify_checksums(root)
    records = load_manifest(root)
    output = root / OUTPUT_RELATIVE / model_key
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "run_meta.json"
    existing_meta = _read_json(meta_path)
    if existing_meta and existing_meta.get("status") in {"stopped_budget", "stopped_memory", "out_of_memory"}:
        print(
            f"{model_key} already stopped with {existing_meta.get('status')}. Not retrying.",
            flush=True,
        )
        return 2
    if existing_meta and existing_meta.get("status") == "completed":
        print(f"{model_key} already has a completed run. Not loading the model again.", flush=True)
        return 0
    rows = _validated_rows(output / "predictions.jsonl", records, model_key)
    by_id = {record["scene_id"]: record for record in records}
    if model_key == "smolvlm2":
        return _run_smol(root, output, records, by_id, rows, provenance)
    if model_key == "qwen3-vl-2b":
        return _run_qwen(root, output, records, by_id, rows, provenance)
    raise RuntimeError(f"unknown model {model_key}")


def _run_smol(root, output, records, by_id, rows, provenance) -> int:
    from vlm_iface.devices import available_bytes, memory_snapshot
    from vlm_iface.pins import MIN_AVAILABLE_BYTES

    if MIN_AVAILABLE_BYTES != 6 * 1024**3:
        raise RuntimeError("The 6 GiB OS floor changed. Refusing to adopt a lower floor.")
    lock = acquire(root, "smolvlm2")
    session = None
    called = False
    try:
        import torch

        if not torch.backends.mps.is_available():
            _write_stop(output, provenance, model_key="smolvlm2", status="stopped_memory", called=False, reason="MPS is not available.")
            return 2
        free = available_bytes()
        if free < MIN_AVAILABLE_BYTES:
            _write_stop(
                output,
                provenance,
                model_key="smolvlm2",
                status="stopped_memory",
                called=False,
                reason="OS-available memory is below the 6 GiB guard.",
                available_bytes=free,
                memory=memory_snapshot(torch.device("mps")),
            )
            return 2
        before = memory_snapshot(torch.device("mps"))
        print("Loading SmolVLM2 on MPS.", flush=True)
        from vlm_iface.session import VLMSession

        pin = require_model("smolvlm2")
        started = time.perf_counter()
        called = True
        session = VLMSession(
            device="mps",
            do_image_splitting=DO_IMAGE_SPLITTING,
            max_new_tokens=MAX_NEW_TOKENS,
            leading_space=True,
            model_id=pin.model_id,
            revision=pin.revision,
        )
        load_seconds = time.perf_counter() - started
        parameter = next(session.model.parameters())
        if parameter.device.type != "mps" or parameter.dtype != torch.float32:
            raise RuntimeError("SmolVLM2 is not float32 on MPS.")
        tokenization = assert_locked_tokenization(session.processor, "smolvlm2")
        if tokenization["any_selected_multi_token"]:
            raise RuntimeError("a selected color is multi-token; the one-row path would be the wrong sum")
        identity = _identity("smolvlm2", tokenization, dtype="float32")

        def forward(image_path: Path, user_text: str):
            logits = session.prompt_last_logits(image_path, user_text)
            summed, mean = nlls_from_first_logit(logits, identity["token_ids_tuples"])
            detailed = session.generate_detailed(image_path, user_text, task_family="attribute")
            return summed, mean, str(detailed["text"]), bool(detailed["truncated"])

        return _score(
            output,
            records,
            by_id,
            rows,
            provenance,
            identity=identity,
            forward=forward,
            load_seconds=load_seconds,
            memory_before=before,
            memory_after_load=memory_snapshot(torch.device("mps")),
            called=True,
        )
    except Exception as exc:
        _write_exception(output, provenance, "smolvlm2", exc, called=called)
        if _terminal_failure(exc):
            return 2
        raise
    finally:
        if session is not None:
            del session
            gc.collect()
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        release(lock)


def _run_qwen(root, output, records, by_id, rows, provenance) -> int:
    lock = acquire(root, "qwen3-vl-2b")
    model = None
    called = False
    try:
        import torch
        from qwen_rgb.budget import dtype_decisions, load_budget_module, select_dtype
        from qwen_rgb.measure import bf16_probe, load_model_after_budget, read_local_header, weight_file
        from vlm_iface.devices import available_bytes, memory_snapshot

        budget = load_budget_module(root)
        if not torch.backends.mps.is_available():
            _write_stop(output, provenance, model_key="qwen3-vl-2b", status="stopped_memory", called=False, reason="MPS is not available.")
            return 2
        weights = weight_file()
        if not weights.is_file():
            _write_stop(
                output,
                provenance,
                model_key="qwen3-vl-2b",
                status="stopped_missing_weights",
                called=False,
                reason="The local Qwen safetensors file is absent. This run does not download it.",
            )
            return 2
        header = read_local_header(weights)
        if header["parameter_count"] != budget.QWEN_C0_PARAMETERS or header["float32_weight_bytes"] != budget.QWEN_C0_FLOAT32_BYTES:
            raise RuntimeError("The Qwen header does not match the Checkpoint 1.5 size.")
        free = available_bytes()
        probe = bf16_probe(torch)
        decisions = dtype_decisions(
            budget,
            parameter_count=header["parameter_count"],
            float32_weight_bytes=header["float32_weight_bytes"],
            available_bytes=free,
            mps_available=bool(probe["mps_available"]),
        )
        selection = select_dtype(decisions, bf16_probe_ok=bool(probe.get("matmul_ok")))
        if not selection["permitted"] or selection["dtype"] is None:
            _write_stop(
                output,
                provenance,
                model_key="qwen3-vl-2b",
                status="stopped_budget",
                called=False,
                reason=str(selection["reason"]),
                available_bytes=free,
                available_bytes_at_stop=available_bytes(),
                memory=memory_snapshot(torch.device("mps")),
                dtype_decisions=decisions,
                dtype_selection=selection,
                bf16_probe=probe,
            )
            return 2
        from transformers import AutoProcessor

        pin = require_model("qwen3-vl-2b")
        processor = AutoProcessor.from_pretrained(pin.model_id, revision=pin.revision, local_files_only=True)
        tokenization = assert_locked_tokenization(processor, "qwen3-vl-2b")
        if tokenization["any_selected_multi_token"]:
            raise RuntimeError("a selected Qwen color is multi-token; the one-row path would be the wrong sum")
        gc.collect()
        torch.mps.empty_cache()
        recheck_free = available_bytes()
        recheck = dtype_decisions(
            budget,
            parameter_count=header["parameter_count"],
            float32_weight_bytes=header["float32_weight_bytes"],
            available_bytes=recheck_free,
            mps_available=bool(torch.backends.mps.is_available()),
        )
        reselection = select_dtype(recheck, bf16_probe_ok=bool(probe.get("matmul_ok")))
        if not reselection["permitted"] or reselection["dtype"] is None:
            _write_stop(
                output,
                provenance,
                model_key="qwen3-vl-2b",
                status="stopped_budget",
                called=False,
                reason=str(reselection["reason"]),
                available_bytes=free,
                available_bytes_at_stop=recheck_free,
                memory=memory_snapshot(torch.device("mps")),
                dtype_decisions=recheck,
                dtype_selection=reselection,
                bf16_probe=probe,
            )
            return 2
        dtype_name = str(reselection["dtype"])
        before = memory_snapshot(torch.device("mps"))
        print(f"Loading Qwen3-VL-2B on MPS as {dtype_name}.", flush=True)
        started = time.perf_counter()
        called = True
        model = load_model_after_budget(recheck[dtype_name], dtype_name)
        device = torch.device("mps")
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        load_seconds = time.perf_counter() - started
        parameter = next(model.parameters())
        if parameter.device.type != "mps" or str(parameter.dtype) != f"torch.{dtype_name}":
            raise RuntimeError("Qwen did not load as the permitted dtype on MPS.")
        identity = _identity("qwen3-vl-2b", tokenization, dtype=dtype_name)

        def forward(image_path: Path, user_text: str):
            return _qwen_forward(model, processor, image_path, user_text, device, identity["token_ids_tuples"])

        return _score(
            output,
            records,
            by_id,
            rows,
            provenance,
            identity=identity,
            forward=forward,
            load_seconds=load_seconds,
            memory_before=before,
            memory_after_load=memory_snapshot(device),
            called=True,
        )
    except Exception as exc:
        _write_exception(output, provenance, "qwen3-vl-2b", exc, called=called)
        if _terminal_failure(exc):
            return 2
        raise
    finally:
        if model is not None:
            del model
            gc.collect()
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        release(lock)


def _score(output, records, by_id, rows, provenance, *, identity, forward: Forward, load_seconds, memory_before, memory_after_load, called: bool) -> int:
    import torch
    from vlm_iface.devices import available_bytes, memory_snapshot
    from vlm_iface.pins import MIN_AVAILABLE_BYTES

    if MIN_AVAILABLE_BYTES != 6 * 1024**3:
        raise RuntimeError("The 6 GiB OS floor changed. Refusing to adopt a lower floor.")
    manifest = repo_root() / MANIFEST_RELATIVE
    started_meta = _base_meta(provenance, identity, called=called, status="scoring")
    started_meta["inference_started_unix"] = time.time()
    started_meta["available_bytes_before_first_scene"] = available_bytes()
    _write(output / "run_meta.json", started_meta)
    rows = _score_contexts(
        output / "predictions.jsonl",
        manifest,
        records,
        by_id,
        rows,
        PRIMARY_CONTEXTS,
        forward,
        identity,
    )
    extra_reason = None
    extra = False
    if _covered(rows, records, PRIMARY_CONTEXTS) and available_bytes() >= MIN_AVAILABLE_BYTES:
        extra = True
        rows = _score_contexts(
            output / "predictions.jsonl",
            manifest,
            records,
            by_id,
            rows,
            EXTRA_CONTEXTS,
            forward,
            identity,
        )
    elif _covered(rows, records, PRIMARY_CONTEXTS):
        extra_reason = "OS-available memory after the primary pass was below the 6 GiB floor."
    summary = summarize_model(rows, records, model_key=identity["model_key"], gradients=None)
    device = torch.device("mps")
    meta = _base_meta(provenance, identity, called=called, status="completed")
    meta.update(
        {
            "inference_started_unix": started_meta["inference_started_unix"],
            "available_bytes_before_first_scene": started_meta["available_bytes_before_first_scene"],
            "load_seconds": round(load_seconds, 3),
            "score_seconds": round(sum(float(row["seconds"]) for row in rows), 3),
            "memory_before_load": memory_before,
            "memory_after_load": memory_after_load,
            "memory_after_scoring": memory_snapshot(device),
            "extra_contexts_scored": extra and _covered(rows, records, EXTRA_CONTEXTS),
            "extra_contexts_skipped_reason": extra_reason,
            "n_rows": len(rows),
            "competence_gate_passed": summary["competence_gate"]["passed"],
            "contexts_above_80": summary["competence_gate"]["contexts_above_80"],
            "n_groups_eligible": summary["eligible"]["n_groups_eligible"],
        }
    )
    _write(output / "summary.json", summary)
    _write(output / "run_meta.json", meta)
    print(
        json.dumps(
            {
                "model": identity["model_key"],
                "status": "completed",
                "competence_gate_passed": summary["competence_gate"]["passed"],
                "neutral_closed": summary["contexts"]["neutral"]["closed_vocabulary_correct"],
                "helpful_closed": summary["contexts"]["helpful_nonrevealing"]["closed_vocabulary_correct"],
                "n_groups_eligible": None,
            }
        ),
        flush=True,
    )
    return 0


def _score_contexts(path, manifest, records, by_id, rows, contexts, forward: Forward, identity) -> list[dict[str, Any]]:
    have = {(row["scene_id"], row["context"]) for row in rows}
    pending = [
        (record, context)
        for record in records
        for context in contexts
        if (record["scene_id"], context) not in have
    ]
    total = len(records) * len(contexts)
    for index, (record, context) in enumerate(pending, start=1):
        user_text = build_prompt(record["contexts"][context]["text"], record["question"]).user_text()
        image_path = resolve_image_path(manifest, record["image_path"])
        print(f"{identity['model_key']} {context} {index}/{len(pending)} {record['scene_id']}", flush=True)
        started = time.perf_counter()
        summed, mean, text, truncated = forward(image_path, user_text)
        row = assemble_row(
            model_key=identity["model_key"],
            model_id=identity["model_id"],
            revision=identity["revision"],
            processor_class=identity["processor_class"],
            image_processor_class=identity["image_processor_class"],
            dtype=identity["dtype"],
            device="mps",
            scene_id=record["scene_id"],
            counterfactual_group_id=record["counterfactual_group_id"],
            context=context,
            answer=record["answer"],
            summed_nll=summed,
            mean_nll=mean,
            token_ids=identity["token_ids_tuples"],
            continuations=identity["continuations"],
            answer_prefix=identity["answer_prefix"],
            generation_text=text,
            generation_truncated=truncated,
            seconds=time.perf_counter() - started,
            user_text=user_text,
        )
        errors = protocol_errors(row, record)
        if errors:
            raise RuntimeError(f"{record['scene_id']} {context} failed protocol checks: {errors}")
        _append(path, row)
        rows.append(row)
        if index % 8 == 0:
            gc.collect()
    if len(pending) != total and pending:
        pass
    del by_id
    return rows


def _qwen_forward(model, processor, image_path: Path, user_text: str, device, token_ids) -> tuple[dict[str, float], dict[str, float], str, bool]:
    import torch
    from qwen_rgb.measure import encode_chat, move_to_device
    from vlm_iface.images import image_01_to_pil, to_image_01
    from vlm_iface.session import _continuation_ids, _end_token_ids

    image = to_image_01(image_path)
    pil = image_01_to_pil(image)
    encoded = encode_chat(processor, pil, user_text)
    moved = move_to_device(dict(encoded), device)
    with torch.inference_mode():
        outputs = model(**moved, use_cache=False)
    logits = outputs.logits[0, -1].detach().to(dtype=torch.float32, device="cpu")
    del outputs
    summed, mean = nlls_from_first_logit(logits, token_ids)
    prompt_ids = moved["input_ids"]
    with torch.inference_mode():
        generated = model.generate(**moved, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    new_ids = _continuation_ids(generated, prompt_ids)
    token_list = [int(token) for token in new_ids.tolist()]
    text = processor.tokenizer.decode(token_list, skip_special_tokens=True).strip()
    truncated = not any(token in _end_token_ids(model, processor.tokenizer) for token in token_list)
    del generated, moved, logits
    return summed, mean, text, truncated


def _identity(model_key: str, tokenization: Mapping[str, Any], *, dtype: str) -> dict[str, Any]:
    pin = require_model(model_key)
    token_ids = tokenization["token_ids"]
    return {
        "model_key": model_key,
        "model_id": pin.model_id,
        "revision": pin.revision,
        "processor_class": tokenization["processor_class"],
        "image_processor_class": tokenization["image_processor_class"] or "",
        "dtype": dtype,
        "answer_prefix": tokenization["answer_prefix"],
        "continuations": tokenization["continuations"],
        "token_ids_tuples": {color: tuple(ids) for color, ids in token_ids.items()},
    }


def _covered(rows, records, contexts) -> bool:
    have = {(row["scene_id"], row["context"]) for row in rows}
    return all((record["scene_id"], context) in have for record in records for context in contexts)


def _validated_rows(path: Path, records, model_key: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    by_id = {record["scene_id"]: record for record in records}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("model_key") != model_key:
            raise RuntimeError("an existing prediction belongs to another model")
        key = (row["scene_id"], row["context"])
        if key in seen:
            raise RuntimeError(f"duplicate prediction {key}")
        seen.add(key)
        errors = protocol_errors(row, by_id[row["scene_id"]])
        if errors:
            raise RuntimeError(f"existing prediction {key} does not match the protocol: {errors}")
    return rows


def _base_meta(provenance, identity, *, called: bool, status: str) -> dict[str, Any]:
    return {
        "label": LABEL,
        "status": status,
        "model_key": identity["model_key"] if identity else None,
        "model_id": None if identity is None else identity["model_id"],
        "revision": None if identity is None else identity["revision"],
        "processor_class": None if identity is None else identity["processor_class"],
        "image_processor_class": None if identity is None else identity["image_processor_class"],
        "dtype": None if identity is None else identity["dtype"],
        "device": "mps",
        "model_from_pretrained_called": called,
        "attacks_run": False,
        "h_star_fitted": False,
        "training_run": False,
        "cloud_gpu": False,
        "held_out_opened": False,
        "perturbations_run": False,
        "gradients_measured_on_development_v4": False,
        "do_image_splitting": DO_IMAGE_SPLITTING,
        "max_new_tokens": MAX_NEW_TOKENS,
        "protocol_commit": provenance["protocol_commit"],
        "protocol_commit_is_ancestor": provenance["protocol_commit_is_ancestor"],
        "protocol_sha256": provenance["protocol_sha256"],
        "predictions_present_at_protocol_commit": provenance["predictions_present_at_protocol_commit"],
        "git_head_at_run": provenance["head"],
        "manifest_sha256": provenance["manifest_sha256"],
        "checkpoint_1_6_diff_empty": provenance["checkpoint_1_6_diff_empty"],
    }


def _write_stop(output: Path, provenance, *, model_key: str, status: str, called: bool, reason: str, **extra: Any) -> None:
    pin = require_model(model_key)
    identity = {
        "model_key": model_key,
        "model_id": pin.model_id,
        "revision": pin.revision,
        "processor_class": None,
        "image_processor_class": None,
        "dtype": None,
    }
    meta = _base_meta(provenance, identity, called=called, status=status)
    meta["stopped_reason"] = reason
    meta["n_groups_eligible"] = None
    meta.update(extra)
    _write(output / "run_meta.json", meta)
    if status in {"stopped_budget", "stopped_memory"}:
        _write(output / "memory_refusal.json", meta)
    print(json.dumps({"model": model_key, "status": status, "available_bytes": extra.get("available_bytes"), "reason": reason}), flush=True)


def _terminal_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "refusing to load" in text or "from_pretrained refused" in text


def _write_exception(output: Path, provenance, model_key: str, exc: Exception, *, called: bool) -> None:
    text = str(exc).lower()
    status = "out_of_memory" if "out of memory" in text else "error"
    pin = require_model(model_key)
    identity = {
        "model_key": model_key,
        "model_id": pin.model_id,
        "revision": pin.revision,
        "processor_class": None,
        "image_processor_class": None,
        "dtype": None,
    }
    meta = _base_meta(provenance, identity, called=called and status != "stopped_budget", status=status)
    if "refusing to load" in text or "from_pretrained refused" in text:
        meta["model_from_pretrained_called"] = False
        meta["status"] = "stopped_memory"
    meta["error_type"] = type(exc).__name__
    meta["error"] = str(exc)
    meta["traceback"] = traceback.format_exc()
    try:
        from vlm_iface.devices import available_bytes

        meta["available_bytes"] = available_bytes()
    except Exception:
        meta["available_bytes"] = None
    meta["n_groups_eligible"] = None
    _write(output / "run_meta.json", meta)


def _prepare_cache() -> None:
    home = os.environ.get("HF_HOME")
    if home is None:
        os.environ["HF_HOME"] = REQUIRED_HF_HOME
        return
    if Path(home).resolve() != Path(REQUIRED_HF_HOME).resolve():
        raise SystemExit(f"HF_HOME is {home}, expected {REQUIRED_HF_HOME}")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md").is_file():
            return candidate
    raise RuntimeError("development protocol was not found above the runner")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
