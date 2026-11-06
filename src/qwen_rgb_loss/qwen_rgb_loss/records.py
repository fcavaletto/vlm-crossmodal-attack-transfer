"""Compact gradient records. Arrays are hashed here and then discarded by the caller."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from analysis.metrics import sign_alignment

from .protocol import (
    ALIGNMENT_SOURCE_FOR_ELIGIBILITY,
    ALIGNMENT_TARGET_FOR_ELIGIBILITY,
    HISTOGRAM_EDGES,
    PRIMARY_CONTEXTS,
    STABILITY_RELATIVE_L2_CEILING,
)
from qwen_rgb.finite_difference import BF16_ULP_AT_ONE


def gradient_sha256(values: np.ndarray) -> str:
    """SHA-256 of contiguous little-endian float32 bytes. This is not the array."""
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def gradient_record(values: np.ndarray | None, *, missing_reason: str | None = None) -> dict[str, Any]:
    """Finite, nonzero, L1, and L2. A missing gradient has null norms, not zeros."""
    if values is None:
        if not missing_reason:
            raise ValueError("a missing gradient needs a reason")
        return {
            "present": False,
            "finite": None,
            "nonzero": None,
            "l1": None,
            "l2": None,
            "sha256": None,
            "shape": None,
            "fraction_nonzero": None,
            "reason": missing_reason,
        }
    array = np.asarray(values, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    if not finite:
        return {
            "present": True,
            "finite": False,
            "nonzero": None,
            "l1": None,
            "l2": None,
            "sha256": None,
            "shape": list(array.shape),
            "fraction_nonzero": None,
            "reason": "gradient is not finite",
        }
    flat = array.reshape(-1)
    nonzero = bool(np.any(flat != 0.0))
    return {
        "present": True,
        "finite": True,
        "nonzero": nonzero,
        "l1": float(np.abs(flat).sum()),
        "l2": float(np.linalg.norm(flat, ord=2)),
        "sha256": gradient_sha256(array),
        "shape": list(array.shape),
        "fraction_nonzero": float(np.mean(flat != 0.0)),
        "reason": None if nonzero else "gradient is zero",
    }


def clean_margin(summed_nll: dict[str, float], label: str) -> float:
    """Best other summed NLL minus the labeled summed NLL. Positive favors the label."""
    others = [float(summed_nll[color]) for color in summed_nll if color != label]
    if not others or label not in summed_nll:
        raise ValueError("margin needs the labeled color and at least one alternative")
    return float(min(others) - float(summed_nll[label]))


def directed_alignments(
    neutral: np.ndarray | None,
    helpful: np.ndarray | None,
) -> dict[str, Any]:
    """``A(neutral, helpful)`` and ``A(helpful, neutral)``.

    The eligibility scalar is ``A(neutral, helpful)`` when both directions are
    defined and inside [-1, 1]. An undefined alignment is null, not zero.
    """
    result: dict[str, Any] = {
        "alignment_neutral_to_helpful": None,
        "alignment_helpful_to_neutral": None,
        "eligibility_alignment": None,
        "eligibility_alignment_definition": (
            f"A({ALIGNMENT_SOURCE_FOR_ELIGIBILITY}, {ALIGNMENT_TARGET_FOR_ELIGIBILITY}) "
            "when both directed alignments are finite and inside [-1, 1]"
        ),
        "undefined_reason": None,
    }
    if neutral is None or helpful is None:
        result["undefined_reason"] = "a gradient is missing"
        return result
    left = np.asarray(neutral, dtype=np.float64)
    right = np.asarray(helpful, dtype=np.float64)
    if left.shape != right.shape:
        result["undefined_reason"] = "gradient shapes differ"
        return result
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        result["undefined_reason"] = "a gradient is not finite"
        return result
    try:
        neutral_to_helpful = sign_alignment(left, right)
        helpful_to_neutral = sign_alignment(right, left)
    except ValueError as exc:
        result["undefined_reason"] = str(exc)
        return result
    result["alignment_neutral_to_helpful"] = neutral_to_helpful["alignment_a"]
    result["alignment_helpful_to_neutral"] = helpful_to_neutral["alignment_a"]
    reasons = [
        text
        for text in (neutral_to_helpful["undefined_reason"], helpful_to_neutral["undefined_reason"])
        if text
    ]
    if reasons:
        result["undefined_reason"] = "; ".join(reasons)
        return result
    pair = (
        neutral_to_helpful["alignment_a"],
        helpful_to_neutral["alignment_a"],
    )
    if any(not isinstance(value, float) or not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in pair):
        result["undefined_reason"] = "an alignment is outside [-1, 1]"
        return result
    result["eligibility_alignment"] = float(neutral_to_helpful["alignment_a"])
    return result


def stability_report(first: np.ndarray | None, second: np.ndarray | None) -> dict[str, Any]:
    """Repeat check. A numerical miss is not the same event as an invalid gradient."""
    first_record = gradient_record(first, missing_reason="the first gradient is missing")
    second_record = gradient_record(second, missing_reason="the repeated gradient is missing")
    report: dict[str, Any] = {
        "ran": first is not None or second is not None,
        "ceiling": STABILITY_RELATIVE_L2_CEILING,
        "relative_l2": None,
        "cosine": None,
        "max_abs": None,
        "sha256_matches": None,
        "numerical_check": "not_run",
        "gradient_invalid": True,
        "reason": None,
        "repeat_l1": second_record["l1"],
        "repeat_l2": second_record["l2"],
        "repeat_finite": second_record["finite"],
        "repeat_nonzero": second_record["nonzero"],
        "repeat_sha256": second_record["sha256"],
    }
    if first is None or second is None or first_record["finite"] is not True or second_record["finite"] is not True:
        report["reason"] = first_record["reason"] if first is None or first_record["finite"] is not True else second_record["reason"]
        report["gradient_invalid"] = True
        return report
    report["gradient_invalid"] = not (first_record["nonzero"] and second_record["nonzero"])
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        report["numerical_check"] = "failed"
        report["reason"] = "the repeat shape differs"
        report["gradient_invalid"] = True
        return report
    gap = left - right
    denominator = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), BF16_ULP_AT_ONE)
    relative = float(np.linalg.norm(gap) / denominator)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = None
    if left_norm > 0.0 and right_norm > 0.0:
        cosine = float(np.dot(left, right) / (left_norm * right_norm))
    report["relative_l2"] = relative
    report["cosine"] = cosine
    report["max_abs"] = float(np.max(np.abs(gap)))
    report["sha256_matches"] = first_record["sha256"] == second_record["sha256"]
    if report["gradient_invalid"]:
        report["numerical_check"] = "not_applied"
        report["reason"] = "a repeated gradient is zero or missing, so the gradient is invalid"
        return report
    if relative <= STABILITY_RELATIVE_L2_CEILING:
        report["numerical_check"] = "confirmed"
        report["reason"] = None
        return report
    report["numerical_check"] = "failed"
    report["reason"] = "relative L2 of the repeat exceeds the documented BF16 ceiling"
    return report


def distribution(values: Sequence[float | None], *, undefined: int, missing: int) -> dict[str, Any]:
    """Distribution of defined scalars. Nulls stay out of the mean."""
    defined = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    edges = np.asarray(HISTOGRAM_EDGES, dtype=np.float64)
    if defined:
        counts, _ = np.histogram(np.asarray(defined, dtype=np.float64), bins=edges)
        array = np.asarray(defined, dtype=np.float64)
        summary: dict[str, Any] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "max": float(array.max()),
            "p10": _quantile(array, 0.10),
            "p25": _quantile(array, 0.25),
            "p50": _quantile(array, 0.50),
            "p75": _quantile(array, 0.75),
            "p90": _quantile(array, 0.90),
            "n_negative": int(np.sum(array < 0.0)),
            "n_zero": int(np.sum(array == 0.0)),
            "n_positive": int(np.sum(array > 0.0)),
        }
        histogram = [int(count) for count in counts.tolist()]
    else:
        summary = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "n_negative": 0,
            "n_zero": 0,
            "n_positive": 0,
        }
        histogram = [0] * (len(HISTOGRAM_EDGES) - 1)
    summary.update(
        {
            "n_defined": len(defined),
            "n_undefined": int(undefined),
            "n_missing": int(missing),
            "nulls_excluded_from_mean": True,
            "nulls_imputed_as_zero": False,
            "histogram_edges": list(HISTOGRAM_EDGES),
            "histogram_counts": histogram,
            "histogram_rule": "NumPy histogram. The rightmost edge is included. Other edges are left-closed.",
        }
    )
    return summary


def norms_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """L1 and L2 over gradients that are present and finite. Missing norms are excluded."""
    return {
        context: {
            "l1": distribution(
                [_norm(record, context, "l1") for record in records],
                undefined=0,
                missing=sum(_norm(record, context, "l1") is None for record in records),
            ),
            "l2": distribution(
                [_norm(record, context, "l2") for record in records],
                undefined=0,
                missing=sum(_norm(record, context, "l2") is None for record in records),
            ),
        }
        for context in PRIMARY_CONTEXTS
    }


def _norm(record: dict[str, Any], context: str, key: str) -> float | None:
    entry = record.get(context)
    if not isinstance(entry, dict):
        return None
    gradient = entry.get("gradient")
    if not isinstance(gradient, dict) or gradient.get("finite") is not True:
        return None
    value = gradient.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _quantile(array: np.ndarray, quantile: float) -> float:
    return float(np.quantile(array, quantile, method="linear"))
