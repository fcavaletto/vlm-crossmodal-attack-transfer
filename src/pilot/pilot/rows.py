"""Pure row builders for the development pilot."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from analysis.metrics import sign_alignment
from analysis.pilot_schema import LABEL, MODEL_ID, MODEL_REVISION, PREPROCESS

DISAGREEMENT_TOLERANCE = 1e-6


def judge_parse(parsed_answer: str | None, invalid: bool, answer: str, aliases: list[str]) -> dict[str, bool]:
    """Split one generation into correct, valid-wrong, or invalid."""
    if invalid:
        if parsed_answer is not None:
            raise ValueError("an invalid parse has no parsed answer")
        return {"correct": False, "valid_wrong": False, "invalid": True}
    if not isinstance(parsed_answer, str) or parsed_answer == "":
        raise ValueError("a valid parse needs a parsed answer")
    if answer not in aliases:
        raise ValueError("the canonical answer is missing from its aliases")
    matched = parsed_answer == answer or parsed_answer in aliases
    if matched:
        return {"correct": True, "valid_wrong": False, "invalid": False}
    return {"correct": False, "valid_wrong": True, "invalid": False}


def scoring_disagreement(parsed_answer: str | None, invalid: bool, scores: Mapping[str, Mapping[str, float]]) -> tuple[str, float, bool]:
    """True when a valid parse is not the lowest-sum-NLL vocabulary item."""
    if not scores:
        raise ValueError("vocabulary scores are empty")
    argmin = min(scores, key=lambda word: (scores[word]["sum_nll"], word))
    best = float(scores[argmin]["sum_nll"])
    if invalid or parsed_answer is None:
        return argmin, best, False
    if parsed_answer not in scores:
        raise ValueError(f"parsed answer {parsed_answer!r} was not scored")
    worse = float(scores[parsed_answer]["sum_nll"]) > best + DISAGREEMENT_TOLERANCE
    return argmin, best, worse


def gradient_stats(gradient) -> dict[str, Any]:
    """Compact statistics for one ``(3, 384, 512)`` float gradient. The tensor is not kept."""
    array = np.asarray(gradient, dtype=np.float64)
    if array.shape != (3, 384, 512):
        raise ValueError(f"expected gradient shape (3, 384, 512), got {array.shape}")
    finite = bool(np.isfinite(array).all())
    nonzero = float(np.mean(array != 0))
    if not finite:
        return {
            "finite": False,
            "nonzero_fraction": nonzero,
            "l1": None,
            "l2": None,
            "linf": None,
            "height": 384,
            "width": 512,
        }
    return {
        "finite": True,
        "nonzero_fraction": nonzero,
        "l1": float(np.abs(array).sum()),
        "l2": float(np.linalg.norm(array.reshape(-1), ord=2)),
        "linf": float(np.abs(array).max()),
        "height": 384,
        "width": 512,
    }


def make_alignment_row(
    *,
    scene_id: str,
    counterfactual_group_id: str,
    task_family: str,
    source_context: str,
    target_context: str,
    source_gradient,
    target_gradient,
    source_stats: Mapping[str, Any],
    target_stats: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Store ``A(source, target)`` without keeping the tensors."""
    if device != "mps":
        raise RuntimeError("A CPU result is not an MPS result.")
    source_finite = bool(source_stats["finite"])
    target_finite = bool(target_stats["finite"])
    source_l1 = source_stats["l1"]
    target_l1 = target_stats["l1"]
    source_l1_zero = bool(source_finite and source_l1 == 0)
    if not source_finite or not target_finite:
        alignment = None
        reason = "non-finite gradient"
    elif target_l1 == 0:
        alignment = None
        reason = "target gradient L1 norm is zero"
    else:
        result = sign_alignment(np.asarray(source_gradient), np.asarray(target_gradient))
        alignment = result["alignment_a"]
        reason = result["undefined_reason"]
    return {
        "label": LABEL,
        "scene_id": scene_id,
        "counterfactual_group_id": counterfactual_group_id,
        "task_family": task_family,
        "source_context": source_context,
        "target_context": target_context,
        "alignment_a": alignment,
        "undefined_reason": reason,
        "source_l1": source_l1,
        "target_l1": target_l1,
        "source_finite": source_finite,
        "target_finite": target_finite,
        "source_l1_zero": source_l1_zero,
        "source_gradient_excludes_h_star": source_l1_zero or (not source_finite),
        "device": device,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def make_prediction_row(
    *,
    record: Mapping[str, Any],
    context: str,
    prompt: str,
    raw_output: str,
    parsed_answer: str | None,
    parse_reasons: list[str],
    invalid: bool,
    scores: Mapping[str, Mapping[str, float]],
    device: str,
) -> dict[str, Any]:
    """One scene-context row. ``scores`` is keyed by the canonical vocabulary."""
    if device != "mps":
        raise RuntimeError("A CPU result is not an MPS result.")
    answer = str(record["answer"])
    aliases = list(record["answer_aliases"])
    judgment = judge_parse(parsed_answer, invalid, answer, aliases)
    argmin, argmin_nll, disagreement = scoring_disagreement(parsed_answer, invalid, scores)
    if answer not in scores:
        raise ValueError(f"{record['scene_id']}: the canonical answer was not scored")
    truth = scores[answer]
    return {
        "label": LABEL,
        "scene_id": record["scene_id"],
        "counterfactual_group_id": record["counterfactual_group_id"],
        "task_family": record["task_family"],
        "context": context,
        "answer": answer,
        "answer_aliases": aliases,
        "context_text": record["contexts"][context]["text"],
        "question": record["question"],
        "prompt": prompt,
        "n_objects": len(record["objects"]),
        "ambiguity_flags": list(record["ambiguity_flags"]),
        "raw_output": raw_output,
        "parsed_answer": parsed_answer,
        "parse_reasons": list(parse_reasons),
        "correct": judgment["correct"],
        "valid_wrong": judgment["valid_wrong"],
        "invalid": judgment["invalid"],
        "sum_nll": float(truth["sum_nll"]),
        "token_count": int(truth["token_count"]),
        "mean_nll": float(truth["mean_nll"]),
        "library_mean_abs_diff": float(truth["library_mean_abs_diff"]),
        "scored_argmin": argmin,
        "scored_argmin_sum_nll": float(argmin_nll),
        "scoring_generation_disagreement": disagreement,
        "device": device,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
    }
