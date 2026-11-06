"""Score the 24 committed development scenes on MPS.

The run loads one pinned SmolVLM2 checkpoint. It records first-token class
losses and greedy generations. It does not run PGD, FGSM, noise, or any other
attack. It does not fit H*. It does not write ``docs/checkpoint-1``.

Every number it writes is DEVELOPMENT ONLY. A CPU result is refused.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from analysis.pilot_schema import LABEL, MODEL_ID, MODEL_REVISION
from analysis.schemas import CONTEXTS
from pilot.constants import EXPECTED_SCENE_IDS, GATE_SCENE_ID, IMAGE_HEIGHT, IMAGE_WIDTH, N_SCENES
from pilot.manifest import load_committed_manifest
from pilot.prompt import user_prompt
from scene_gen.paths import resolve_image_path
from vlm_iface.devices import available_bytes, memory_snapshot, resolve_device, synchronize
from vlm_iface.images import to_image_01
from vlm_iface.pins import MIN_AVAILABLE_BYTES, PINNED_REVISION
from vlm_iface.session import VLMSession

from .class_loss import two_token_logprob
from .compare import attach_first_token, score_logits
from .instruction import FORMAT_INSTRUCTION, format_user_prompt
from .surfaces import SurfaceSet, build_surfaces

CHECKPOINT1 = "checkpoint1_prompt"
FORMAT = "format_instruction"
COUNTING_SPOT_SCENE = "dev-counting-00-a"


def repo_root() -> Path:
    """Repository root. This file lives at ``src/scoring/answer_scoring``."""
    return Path(__file__).resolve().parents[3]


def assert_writable_scoring_path(path: Path, root: Path) -> None:
    """Refuse any output path that would replace Checkpoint 1 files."""
    resolved = path.resolve()
    forbidden = (root / "docs" / "checkpoint-1").resolve()
    if resolved == forbidden or forbidden in resolved.parents or resolved.is_relative_to(forbidden):
        raise RuntimeError(f"Refusing to write under {forbidden}.")
    if "docs/checkpoint-1" in resolved.as_posix():
        raise RuntimeError(f"Refusing output path {resolved}.")


def scoring_records_dir(root: Path) -> Path:
    """Directory this workstream may write. Checkpoint 1 files are refused."""
    path = (root / "src" / "scoring" / "records").resolve()
    assert_writable_scoring_path(path, root)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint 1.5 answer scoring. MPS only. No attacks.")
    parser.add_argument("--device", required=True, choices=["mps"])
    args = parser.parse_args(argv)
    root = repo_root()
    output = scoring_records_dir(root)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return run(root, output, device_name=args.device)
    except Exception as exc:
        payload = {
            "label": LABEL,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "run_error.json", payload)
        print(f"Answer scoring stopped: {type(exc).__name__}: {exc}", flush=True)
        return 1


def run(root: Path, output: Path, *, device_name: str) -> int:
    if device_name != "mps":
        raise RuntimeError(f"Refusing device {device_name!r}. A CPU result is not an MPS result.")
    device = resolve_device(device_name)
    if device.type != "mps":
        raise RuntimeError(f"Requested MPS but resolved {device.type}.")
    if PINNED_REVISION != MODEL_REVISION:
        raise RuntimeError("The analysis pin and the model pin disagree.")
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
        }
        _write_json(output / "memory_refusal.json", refusal)
        print(json.dumps(refusal), flush=True)
        return 2

    manifest_path, records = load_committed_manifest(root)
    if [row["scene_id"] for row in records] != sorted(EXPECTED_SCENE_IDS) or len(records) != N_SCENES:
        raise RuntimeError("The development manifest is not the committed 24-scene list.")
    stored = _load_stored_predictions(root)
    print(f"Loading {MODEL_ID} on MPS.", flush=True)
    started = time.perf_counter()
    session = VLMSession(device="mps", do_image_splitting=False, max_new_tokens=16, leading_space=True)
    synchronize(device)
    load_seconds = time.perf_counter() - started
    after_load = memory_snapshot(device)
    parameter = next(session.model.parameters())
    if parameter.device.type != "mps" or parameter.dtype != torch.float32:
        raise RuntimeError("The loaded checkpoint is not float32 on MPS.")
    if session.model_id != MODEL_ID or session.revision != PINNED_REVISION:
        raise RuntimeError("The loaded checkpoint does not match the pin.")
    try:
        surfaces = {
            family: build_surfaces(family, session.processor.tokenizer, leading_space=True)
            for family in ("attribute", "counting", "spatial")
        }
        rows: list[dict[str, Any]] = []
        spot: list[dict[str, Any]] = []
        digit_checks: list[dict[str, Any]] = []
        score_started = time.perf_counter()
        for record in records:
            image = _load_image(manifest_path, record)
            family_surfaces = surfaces[record["task_family"]]
            for context in CONTEXTS:
                base = user_prompt(record["contexts"][context]["text"], record["question"])
                logits = session.prompt_last_logits(image, base)
                extra = _extra_logprobs(session, image, base, logits, family_surfaces)
                stored_row = stored[(record["scene_id"], context)]
                scored = score_logits(
                    logits,
                    family_surfaces,
                    raw_output=stored_row["raw_output"],
                    truncated=False,
                    answer=record["answer"],
                    aliases=list(record["answer_aliases"]),
                    extra_logprobs=extra,
                )
                rows.append(
                    _envelope(
                        record,
                        context=context,
                        prompt=base,
                        condition=CHECKPOINT1,
                        raw_output_source="checkpoint1_predictions",
                        scored=scored,
                        stored_row=stored_row,
                    )
                )
                if record["scene_id"] in {GATE_SCENE_ID, COUNTING_SPOT_SCENE}:
                    generated = session.generate_detailed(image, base, task_family=record["task_family"])
                    spot.append(
                        {
                            "scene_id": record["scene_id"],
                            "context": context,
                            "stored_raw_output": stored_row["raw_output"],
                            "fresh_raw_output": generated["text"],
                            "match": generated["text"] == stored_row["raw_output"],
                            "fresh_token_ids": generated["token_ids"],
                        }
                    )
            if record["scene_id"] == records[0]["scene_id"]:
                worst = max(
                    float(row["word_nll_abs_diff_vs_stored"])
                    for row in rows
                    if row["prompt_condition"] == CHECKPOINT1
                )
                print(f"first scene word-nll max abs diff {worst}", flush=True)
                if worst > 1e-2:
                    raise RuntimeError(
                        "Canonical-word NLL does not reproduce the stored Checkpoint 1 loss "
                        f"(max abs diff {worst})."
                    )
            if record["scene_id"] == COUNTING_SPOT_SCENE:
                digit_checks.append(_check_digit_sequence(session, image, record, family_surfaces))
            for context in CONTEXTS:
                prompted = format_user_prompt(record["contexts"][context]["text"], record["question"])
                generated = session.generate_detailed(image, prompted, task_family=record["task_family"])
                logits = session.prompt_last_logits(image, prompted)
                extra = _extra_logprobs(session, image, prompted, logits, family_surfaces)
                scored = score_logits(
                    logits,
                    family_surfaces,
                    raw_output=str(generated["text"]),
                    truncated=bool(generated["truncated"]),
                    answer=record["answer"],
                    aliases=list(record["answer_aliases"]),
                    extra_logprobs=extra,
                )
                scored = attach_first_token(scored, list(generated["token_ids"]))
                rows.append(
                    _envelope(
                        record,
                        context=context,
                        prompt=prompted,
                        condition=FORMAT,
                        raw_output_source="mps_generation",
                        scored=scored,
                        stored_row=None,
                        generated_token_ids=list(generated["token_ids"]),
                        truncated=bool(generated["truncated"]),
                    )
                )
            print(f"scored {record['scene_id']}", flush=True)
        synchronize(device)
        score_seconds = time.perf_counter() - score_started
        after = memory_snapshot(device)
    finally:
        del session
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    summary = _summary(
        rows,
        surfaces=surfaces,
        spot=spot,
        digit_checks=digit_checks,
        load_seconds=load_seconds,
        score_seconds=score_seconds,
        memory_before_load=before,
        memory_after_load=after_load,
        memory_after_run=after,
    )
    _write_jsonl(output / "development_mps.jsonl", rows)
    _write_json(output / "development_mps.json", summary)
    print(
        json.dumps(
            {
                "label": LABEL,
                "rows": len(rows),
                "word_nll_max_abs_diff": summary["checkpoint1_word_nll_max_abs_diff"],
                "spot_matches": summary["generation_spotcheck_matches"],
            }
        ),
        flush=True,
    )
    return 0


def _envelope(
    record: Mapping[str, Any],
    *,
    context: str,
    prompt: str,
    condition: str,
    raw_output_source: str,
    scored: Mapping[str, Any],
    stored_row: Mapping[str, Any] | None,
    generated_token_ids: list[int] | None = None,
    truncated: bool | None = None,
) -> dict[str, Any]:
    row = {
        "label": LABEL,
        "measurement_kind": "checkpoint1_5_answer_scoring",
        "prompt_condition": condition,
        "format_instruction": FORMAT_INSTRUCTION if condition == FORMAT else None,
        "scene_id": record["scene_id"],
        "counterfactual_group_id": record["counterfactual_group_id"],
        "task_family": record["task_family"],
        "context": context,
        "answer": record["answer"],
        "answer_aliases": list(record["answer_aliases"]),
        "prompt": prompt,
        "raw_output_source": raw_output_source,
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "generated_token_ids": generated_token_ids,
        "truncated": truncated,
        **scored,
    }
    if stored_row is not None:
        same_parse = (
            scored["strict_parsed_answer"] == stored_row["parsed_answer"]
            and scored["strict_invalid"] == stored_row["invalid"]
            and scored["strict_reasons"] == list(stored_row["parse_reasons"])
        )
        row["stored_parsed_answer"] = stored_row["parsed_answer"]
        row["stored_invalid"] = stored_row["invalid"]
        row["stored_correct"] = stored_row["correct"]
        row["stored_valid_wrong"] = stored_row["valid_wrong"]
        row["stored_parse_reasons"] = list(stored_row["parse_reasons"])
        row["stored_sum_nll"] = stored_row["sum_nll"]
        row["stored_scored_argmin"] = stored_row["scored_argmin"]
        row["stored_disagreement"] = stored_row["scoring_generation_disagreement"]
        row["stored_reparse_match"] = same_parse
        row["word_nll_abs_diff_vs_stored"] = abs(float(scored["true_word_nll"]) - float(stored_row["sum_nll"]))
        row["word_argmin_matches_stored"] = scored["word_only_argmin"] == stored_row["scored_argmin"]
    return row


def _summary(
    rows: list[dict[str, Any]],
    *,
    surfaces,
    spot: list[dict[str, Any]],
    digit_checks: list[dict[str, Any]],
    load_seconds: float,
    score_seconds: float,
    memory_before_load: Mapping[str, Any],
    memory_after_load: Mapping[str, Any],
    memory_after_run: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_rows = [row for row in rows if row["prompt_condition"] == CHECKPOINT1]
    diffs = [float(row["word_nll_abs_diff_vs_stored"]) for row in checkpoint_rows]
    max_diff = max(diffs) if diffs else None
    return {
        "label": LABEL,
        "measurement_kind": "checkpoint1_5_answer_scoring",
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_scenes": N_SCENES,
        "n_rows": len(rows),
        "format_instruction": FORMAT_INSTRUCTION,
        "format_instruction_selected_by_accuracy": False,
        "class_loss_selected_by_accuracy": False,
        "strict_parser_replaced": False,
        "alternative_selected": False,
        "attack_run": False,
        "h_star_fitted": False,
        "load_seconds": round(load_seconds, 3),
        "score_seconds": round(score_seconds, 3),
        "memory_before_load": dict(memory_before_load),
        "memory_after_load": dict(memory_after_load),
        "memory_after_run": dict(memory_after_run),
        "checkpoint1_word_nll_max_abs_diff": max_diff,
        "checkpoint1_word_nll_reproduced": bool(max_diff is not None and max_diff <= 1e-3),
        "checkpoint1_reparse_mismatches": sum(not row["stored_reparse_match"] for row in checkpoint_rows),
        "checkpoint1_word_argmin_mismatches": sum(not row["word_argmin_matches_stored"] for row in checkpoint_rows),
        "generation_spotcheck": spot,
        "generation_spotcheck_matches": sum(bool(item["match"]) for item in spot),
        "digit_sequence_checks": digit_checks,
        "surfaces": {
            family: [surface.as_dict() for group in surface_set.by_class.values() for surface in group]
            for family, surface_set in surfaces.items()
        },
        "skipped_surfaces": {
            family: list(surface_set.skipped) for family, surface_set in surfaces.items()
        },
        "by_condition": {condition: _condition_summary(rows, condition) for condition in (CHECKPOINT1, FORMAT)},
        "note": (
            "DEVELOPMENT ONLY. Class NLL on the 24 committed scenes. "
            "Single-token surfaces use the first answer position. "
            "The leading-space digit uses its two-token joint log-probability. "
            "The strict parser is unchanged. No attack and no H* fit."
        ),
    }


def _condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["prompt_condition"] == condition]
    return {
        "n": len(chosen),
        "strict_correct": sum(bool(row["strict_correct"]) for row in chosen),
        "strict_valid_wrong": sum(bool(row["strict_valid_wrong"]) for row in chosen),
        "strict_invalid": sum(bool(row["strict_invalid"]) for row in chosen),
        "word_only_disagreement": sum(bool(row["word_only_disagreement"]) for row in chosen),
        "class_disagreement": sum(bool(row["class_disagreement"]) for row in chosen),
        "class_differs_from_word": sum(bool(row["class_differs_from_word"]) for row in chosen),
        "alternative_applicable": sum(bool(row["alternative_applicable"]) for row in chosen),
        "alternative_matches_answer": sum(row["alternative_matches_answer"] is True for row in chosen),
        "first_token_outside_surface_set": sum(row.get("first_token_in_surface_set") is False for row in chosen),
        "by_context": {
            context: _slice_counts(chosen, context=context) for context in CONTEXTS
        },
        "by_family_context": {
            family: {context: _slice_counts(chosen, context=context, family=family) for context in CONTEXTS}
            for family in ("attribute", "counting", "spatial")
        },
    }


def _slice_counts(
    rows: list[dict[str, Any]], *, context: str, family: str | None = None
) -> dict[str, int]:
    chosen = [row for row in rows if row["context"] == context and (family is None or row["task_family"] == family)]
    return {
        "n": len(chosen),
        "strict_correct": sum(bool(row["strict_correct"]) for row in chosen),
        "strict_valid_wrong": sum(bool(row["strict_valid_wrong"]) for row in chosen),
        "strict_invalid": sum(bool(row["strict_invalid"]) for row in chosen),
        "word_only_disagreement": sum(bool(row["word_only_disagreement"]) for row in chosen),
        "class_disagreement": sum(bool(row["class_disagreement"]) for row in chosen),
        "class_differs_from_word": sum(bool(row["class_differs_from_word"]) for row in chosen),
    }


def _load_stored_predictions(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "docs" / "checkpoint-1" / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {(row["scene_id"], row["context"]): row for row in rows}
    if len(index) != 96:
        raise RuntimeError(f"Expected 96 stored predictions, found {len(index)}.")
    return index


def _extra_logprobs(session, image, prompt: str, logits: torch.Tensor, surfaces: SurfaceSet) -> dict[tuple[int, ...], float]:
    """Joint log-probabilities of two-token alias sequences."""
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


def _check_digit_sequence(session, image, record: Mapping[str, Any], surfaces: SurfaceSet) -> dict[str, Any]:
    """One counting scene: the two-token digit log-probability must match ``score_details``."""
    context = "neutral"
    prompt = user_prompt(str(record["contexts"][context]["text"]), str(record["question"]))
    digit = next(alias for alias in record["answer_aliases"] if alias != record["answer"])
    sequence = next(
        surface.token_ids
        for surface in surfaces.by_class[str(record["answer"])]
        if surface.kind == "digit_alias" and surface.leading_space
    )
    logits = session.prompt_last_logits(image, prompt)
    extra = _extra_logprobs(session, image, prompt, logits, surfaces)
    details = session.score_details(image, prompt, digit)
    diff = abs(float(extra[sequence]) - float(details["logprob"]))
    print(f"digit sequence abs diff {diff}", flush=True)
    if diff > 1e-3:
        raise RuntimeError(
            "Digit sequence log-probability does not match score_details "
            f"(abs diff {diff})."
        )
    return {
        "scene_id": record["scene_id"],
        "context": context,
        "digit": digit,
        "token_ids": list(sequence),
        "abs_diff": diff,
        "sequence_logprob": extra[sequence],
        "score_details_logprob": details["logprob"],
    }


def _load_image(manifest_path: Path, record: Mapping[str, Any]) -> torch.Tensor:
    path = resolve_image_path(manifest_path, record["image_path"])
    image = to_image_01(path)
    if tuple(image.shape) != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise RuntimeError(
            f"{record['scene_id']} has shape {tuple(image.shape)}, expected "
            f"(3, {IMAGE_HEIGHT}, {IMAGE_WIDTH})."
        )
    return image


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
