"""Score the 24 row-color scenes on MPS. Development only. No attacks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from answer_scoring.compare import score_logits
from answer_scoring.surfaces import build_surfaces
from pilot.prompt import user_prompt
from pilot.rows import gradient_stats
from scene_simple.paths import resolve_image_path
from vlm_iface.devices import available_bytes, memory_snapshot, resolve_device, synchronize
from vlm_iface.images import to_image_01
from vlm_iface.pins import MIN_AVAILABLE_BYTES, MODEL_ID, PINNED_REVISION
from vlm_iface.session import VLMSession

from .summarize import CONTEXTS, LABEL, PRIMARY, PARSE_FAMILY, build_summary

MANIFEST_SHA = "bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367"
GATE_SCENE = "dev3-row-00-a"
ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Row-color clean evaluation. MPS only.")
    parser.add_argument("--device", required=True, choices=["mps"])
    args = parser.parse_args(argv)
    output = ROOT / "docs" / "checkpoint-1.6" / "development_simple"
    output.mkdir(parents=True, exist_ok=True)
    try:
        return run(output, device_name=args.device)
    except Exception as exc:
        _write(output / "run_error.json", {"label": LABEL, "error": str(exc), "traceback": traceback.format_exc(), "attacks_run": False, "h_star_fitted": False, "qwen_loaded": False})
        print(f"Evaluation stopped: {type(exc).__name__}: {exc}", flush=True)
        return 1


def run(output: Path, *, device_name: str) -> int:
    if device_name != "mps":
        raise RuntimeError("A CPU result is not an MPS result.")
    device = resolve_device("mps")
    manifest_path = ROOT / "fixtures" / "development_simple" / "manifest.jsonl"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise RuntimeError("The row-color manifest does not match the protocol.")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 24 or records[0]["scene_id"] != GATE_SCENE:
        raise RuntimeError("The row-color manifest is not the locked 24-scene list.")
    protocol_sha = hashlib.sha256((ROOT / "docs" / "checkpoint-1.6" / "EVAL_PROTOCOL.md").read_bytes()).hexdigest()
    before = memory_snapshot(device)
    if available_bytes() < MIN_AVAILABLE_BYTES:
        _write(output / "memory_refusal.json", {"label": LABEL, "refused": True, "available_bytes": available_bytes(), "memory": before, "qwen_loaded": False})
        return 2
    print(f"Loading {MODEL_ID} on MPS.", flush=True)
    started = time.perf_counter()
    session = VLMSession(device="mps", do_image_splitting=False, max_new_tokens=16, leading_space=True, model_id=MODEL_ID, revision=PINNED_REVISION)
    synchronize(device)
    load_seconds = time.perf_counter() - started
    after_load = memory_snapshot(device)
    parameter = next(session.model.parameters())
    if parameter.device.type != "mps" or parameter.dtype != torch.float32:
        raise RuntimeError("The loaded checkpoint is not float32 on MPS.")
    if getattr(session.model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("Attention is not eager.")
    surfaces = build_surfaces(PARSE_FAMILY, session.processor.tokenizer, leading_space=True)
    predictions: list[dict[str, Any]] = []
    score_started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        print(f"Score {index}/24 {record['scene_id']}", flush=True)
        image = _image(manifest_path, record)
        predictions.extend(_score(session, image, record, surfaces))
        _write_jsonl(output / "predictions.jsonl", _sort(predictions))
    score_seconds = time.perf_counter() - score_started
    after_score = memory_snapshot(device)
    predictions = _read_jsonl(output / "predictions.jsonl")
    gate = {"label": LABEL, "attempted": False, "passed": None, "scene_id": GATE_SCENE, "reason": None, "gradients": []}
    gradient_seconds = None
    if available_bytes() < MIN_AVAILABLE_BYTES:
        gate["reason"] = "available memory after scoring was below the 6 GiB guard"
    else:
        gate["attempted"] = True
        print(f"Gradient feasibility on {GATE_SCENE}.", flush=True)
        gate_started = time.perf_counter()
        record = next(row for row in records if row["scene_id"] == GATE_SCENE)
        image = _image(manifest_path, record)
        problems = []
        gradients = []
        for context in PRIMARY:
            prompt = user_prompt(record["contexts"][context]["text"], record["question"])
            try:
                synchronize(device)
                t0 = time.perf_counter()
                first = session.image_loss_gradient(image, prompt, record["answer"])
                synchronize(device)
                elapsed = time.perf_counter() - t0
                second = session.image_loss_gradient(image, prompt, record["answer"])
                synchronize(device)
                stats = gradient_stats(first)
                repeat = float((first - second).abs().max())
                del second
                if not stats["finite"] or stats["l1"] == 0:
                    problems.append(f"{context}: non-finite or zero")
                gradients.append({**stats, "context": context, "seconds": round(elapsed, 3), "repeat_max_abs_diff": repeat, "loss": "canonical_word", "device": "mps"})
                del first
            except RuntimeError as exc:
                problems.append(f"{context}: {exc}")
        gradient_seconds = round(time.perf_counter() - gate_started, 3)
        gate["passed"] = not problems
        gate["problems"] = problems
        gate["gradients"] = gradients
    del session
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    run_meta = {
        "label": LABEL,
        "attacks_run": False,
        "h_star_fitted": False,
        "qwen_loaded": False,
        "model_id": MODEL_ID,
        "model_revision": PINNED_REVISION,
        "device": "mps",
        "load_seconds": round(load_seconds, 3),
        "score_seconds": round(score_seconds, 3),
        "gradient_seconds": gradient_seconds,
        "memory_before_load": before,
        "memory_after_load": after_load,
        "memory_after_scoring": after_score,
        "memory_after_gradients": memory_snapshot(device),
        "protocol_sha256": protocol_sha,
        "manifest_sha256": MANIFEST_SHA,
        "git_head_at_run": _git_head(),
    }
    _write(output / "run_meta.json", run_meta)
    _write(output / "gate.json", gate)
    summary = build_summary(predictions, run_meta=json.loads((output / "run_meta.json").read_text()), gate=json.loads((output / "gate.json").read_text()))
    _write(output / "summary.json", summary)
    print(json.dumps({"scenes_clean_both": summary["n_scenes_clean_both_primary"], "groups_clean_both": summary["n_groups_clean_both_primary"], "gate_passed": summary["gate_passed"], "unsuitable": summary["unsuitable"]}), flush=True)
    return 0


def _score(session, image, record, surfaces) -> list[dict[str, Any]]:
    rows = []
    for context in CONTEXTS:
        text = record["contexts"][context]["text"]
        prompt = user_prompt(text, record["question"])
        generated = session.generate_detailed(image, prompt, task_family=PARSE_FAMILY)
        logits = session.prompt_last_logits(image, prompt)
        extra = _extra(session, image, prompt, logits, surfaces)
        scored = score_logits(logits, surfaces, raw_output=str(generated["text"]), truncated=bool(generated["truncated"]), answer=record["answer"], aliases=list(record["answer_aliases"]), extra_logprobs=extra)
        rows.append({
            "label": LABEL,
            "scene_id": record["scene_id"],
            "counterfactual_group_id": record["counterfactual_group_id"],
            "task_family": "row_color",
            "context": context,
            "answer": record["answer"],
            "answer_aliases": list(record["answer_aliases"]),
            "context_text": text,
            "question": record["question"],
            "prompt": prompt,
            "reveals_answer": bool(record["contexts"][context]["reveals_answer"]),
            "raw_output": str(generated["text"]),
            "parsed_answer": scored["strict_parsed_answer"],
            "correct": scored["strict_correct"],
            "valid_wrong": scored["strict_valid_wrong"],
            "invalid": scored["strict_invalid"],
            "truncated": bool(generated["truncated"]),
            "class_nll": scored["class_nll"],
            "class_argmin": scored["class_argmin"],
            "class_disagreement": scored["class_disagreement"],
            "device": "mps",
            "model_id": MODEL_ID,
            "model_revision": PINNED_REVISION,
        })
    return rows


def _extra(session, image, prompt, logits, surfaces):
    from answer_scoring.class_loss import two_token_logprob
    sequences = [surface.token_ids for group in surfaces.by_class.values() for surface in group if len(surface.token_ids) > 1]
    found = {}
    for prefix in {sequence[:1] for sequence in sequences}:
        continuation = session.prompt_last_logits(image, prompt, prefix_token_ids=list(prefix))
        for sequence in sequences:
            if sequence[:1] == prefix:
                found[sequence] = float(two_token_logprob(logits, continuation, sequence[0], sequence[1]).detach().cpu())
    return found


def _image(manifest_path: Path, record: Mapping[str, Any]):
    image = to_image_01(resolve_image_path(manifest_path, record["image_path"]))
    if tuple(image.shape) != (3, 384, 512):
        raise RuntimeError(f"{record['scene_id']} has shape {tuple(image.shape)}")
    return image


def _sort(rows):
    return sorted(rows, key=lambda row: (row["scene_id"], CONTEXTS.index(row["context"])))


def _git_head() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write(path: Path, payload: dict) -> None:
    if "docs/checkpoint-1/" in path.as_posix():
        raise RuntimeError(f"Refusing to write {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if "docs/checkpoint-1/" in path.as_posix():
        raise RuntimeError(f"Refusing to write {path}")
    lines = [json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
