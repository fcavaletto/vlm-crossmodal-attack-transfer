"""Pre-specified Checkpoint 2.1 analysis.

The effective-sample check and the insufficient-variation checks run before any
fit of T. A failed check stores no coefficient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from analysis.bootstrap import scene_bootstrap_rate
from analysis.hstar import (
    DESIGN_CONDITION_LIMIT,
    FLAT_PREDICTOR_RSS_RATIO,
    NEAR_FLAT_HIGH,
    NEAR_FLAT_LOW,
    fit_linear_design,
    interval_entirely_above_zero,
)
from analysis.hstar import _fit_logistic, _pairs_cluster_logistic, _percentile
from attack_calibration.counts import scene_flags
from attack_calibration.prior import describe_scenes, load_prior
from dev_pilot.fixtures import verify_checksums

from .counts import configuration_counts, flatten_scenes
from .protocol import (
    BOOTSTRAP_SEED,
    CALIBRATION_RECORDS_RELATIVE,
    COLORS,
    EPSILON_KEY,
    EVAL_CONTEXTS,
    HSTAR_UNESTIMATED,
    MIN_EVENTS,
    MIN_GROUPS,
    N_GROUPS,
    N_PARAMETERS,
    N_SCENES,
    OUTPUT_RELATIVE,
    PRIMARY_STEPS,
    RATE_ALPHA,
    RATE_DRAWS,
    REPORT_RELATIVE,
    SOURCE_CONTEXT,
    TARGET_CONTEXT,
    WILD_DRAWS,
    epsilon,
)
from .report import write_report
from .verify import (
    attack_constant_errors,
    memory_line_errors,
    preserved_changes,
    record_bound_errors,
    record_decision_errors,
)

COLUMN_NAMES = (
    "intercept",
    "alignment",
    "neutral_margin",
    "helpful_margin",
    "neutral_gradient_l1",
)
RATE_KINDS = ("source_success", "conditional_transfer", "target_attack_success")
CHECK_ORDER = (
    "effective_sample",
    "too_few_groups",
    "flat_outcome",
    "insufficient_variation",
    "near_flat_outcome",
    "flat_predictor",
    "singular_design",
    "verification_failed",
)


def apply_analysis(root: Path) -> dict[str, Any]:
    """Read the finished attack and write the development report."""
    directory = root / OUTPUT_RELATIVE
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    scene_ids = list(progress.get("scene_ids") or [])
    scalars = describe_scenes(load_prior(root), scene_ids) if scene_ids else []
    verification = build_verification(root, progress)
    analysis = analyze_run(progress, scalars, verification)
    _write_json(directory / "analysis.json", analysis)
    _write_json(directory / "verification.json", verification)
    rendered = dict(analysis)
    rendered["file_sha256"] = {
        "docs/checkpoint-2.1/development/progress.json": _sha256(directory / "progress.json"),
        "docs/checkpoint-2.1/development/records.jsonl": _sha256(directory / "records.jsonl"),
        "docs/checkpoint-2.1/development/summary.json": _sha256(directory / "summary.json"),
        "docs/checkpoint-2.1/development/checkpoint_1_8_scalars.json": _sha256(
            directory / "checkpoint_1_8_scalars.json"
        ),
        "docs/checkpoint-2.1/development/analysis.json": _sha256(directory / "analysis.json"),
        "docs/checkpoint-2.1/development/verification.json": _sha256(directory / "verification.json"),
    }
    write_report(root / REPORT_RELATIVE, rendered, progress, scalars)
    return rendered


def analyze_run(
    progress: Mapping[str, Any],
    scalars: list[dict[str, Any]],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the protocol to one saved run. This does not load a model."""
    scene_ids = list(progress.get("scene_ids") or [])
    scenes = progress.get("scenes") or {}
    rows = flatten_scenes(scenes, scene_ids)
    counts = _public_counts(configuration_counts(rows))
    base: dict[str, Any] = {
        "label": "DEVELOPMENT ONLY",
        "h_star_estimated": False,
        "h_star": None,
        "confirmatory_supported": False,
        "h_star_statement": HSTAR_UNESTIMATED,
        "stop_reason": None,
        "failed_checks": [],
        "effective_sample_applied": False,
        "counts": counts,
        "rate_intervals": {},
        "eligible": None,
        "predictor_exclusions": {},
        "gradient_l1_comparison": _gradient_comparison(scenes, scalars),
        "margin_comparison": _margin_comparison(scenes, scalars),
        "verification_passed": bool(verification.get("passed")),
        "n_scenes": progress.get("n_scenes"),
        "n_finished": progress.get("n_finished"),
        "status": progress.get("status"),
        "attack_stop_reason": progress.get("stopped_reason"),
        "protocol_commit": progress.get("protocol_commit"),
        "attack_code_commit": progress.get("git_head_at_start"),
    }
    complete = (
        progress.get("status") == "completed"
        and progress.get("n_finished") == N_SCENES
        and progress.get("n_scenes") == N_SCENES
        and len(scene_ids) == N_SCENES
    )
    if not complete:
        base["stop_reason"] = str(progress.get("status") or "incomplete")
        base["h_star_statement"] = "H* was not estimated. The development run did not finish."
        return base

    base["rate_intervals"] = _rate_intervals(rows)
    eligible, exclusions, group_mismatches = _eligible(rows, scalars)
    base["predictor_exclusions"] = exclusions
    base["eligible"] = _eligible_summary(eligible)
    verification_passed = bool(verification.get("passed"))
    if group_mismatches:
        verification_passed = False
        base["verification_passed"] = False
        base["group_id_mismatches"] = group_mismatches
        if isinstance(verification, dict):
            verification["passed"] = False
            verification["group_id_mismatches"] = group_mismatches
    checks = _prefit_checks(eligible)
    base["effective_sample_applied"] = True
    base["prefit"] = checks["details"]
    failed = [name for name in CHECK_ORDER if name in checks["failed"]]
    if not verification_passed:
        failed.append("verification_failed")
    base["failed_checks"] = failed
    if failed:
        base["stop_reason"] = failed[0]
        base["h_star_statement"] = HSTAR_UNESTIMATED
        base["h_star_estimated"] = False
        base["h_star"] = None
        return base

    fitted, logistic = _estimate(eligible)
    if fitted is None:
        base["stop_reason"] = logistic or "not_estimated"
        base["failed_checks"] = [base["stop_reason"]]
        base["h_star_statement"] = HSTAR_UNESTIMATED
        base["h_star"] = None
        base["h_star_estimated"] = False
        return base
    base["h_star_estimated"] = True
    base["h_star"] = fitted
    base["h_star_statement"] = (
        "H* is the coefficient on alignment in the pre-specified linear probability model."
    )
    base["stop_reason"] = "estimated"
    base["confirmatory_supported"] = False
    return base


def build_verification(root: Path, progress: Mapping[str, Any]) -> dict[str, Any]:
    """Re-check the saved run. A failure keeps H* unestimated."""
    checksum_error = None
    checksum_files = None
    try:
        checksum_files = int(verify_checksums(root)["files"])
    except RuntimeError as exc:
        checksum_error = str(exc)
    scenes = progress.get("scenes") or {}
    bound_violations = record_bound_errors(scenes, epsilon())
    illegal_decisions = record_decision_errors(scenes)
    preserved = preserved_changes(root)
    constants = attack_constant_errors(dict(progress))
    memory = memory_line_errors(dict(progress))
    overlap = calibration_overlap(root, flatten_scenes(scenes, list(progress.get("scene_ids") or [])))
    passed = not any(
        (
            checksum_error,
            bound_violations,
            illegal_decisions,
            preserved,
            constants,
            memory,
            progress.get("held_out_scenes_opened") is True,
            progress.get("float32_loaded") is True,
        )
    )
    return {
        "passed": passed,
        "checksums_ok": checksum_error is None,
        "checksum_files": checksum_files,
        "checksum_error": checksum_error,
        "bounds_ok": not bound_violations,
        "bound_violations": bound_violations[:24],
        "n_bound_violations": len(bound_violations),
        "decisions_ok": not illegal_decisions,
        "illegal_decisions": illegal_decisions[:24],
        "n_illegal_decisions": len(illegal_decisions),
        "preserved_ok": not preserved,
        "preserved_changes": preserved,
        "attack_constants_ok": not constants,
        "attack_constant_errors": constants,
        "memory_lines_ok": not memory,
        "memory_line_errors": memory,
        "held_out_scenes_opened": progress.get("held_out_scenes_opened") is True,
        "float32_loaded": progress.get("float32_loaded") is True,
        "calibration_overlap": overlap,
    }


def calibration_overlap(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare decisions on the Checkpoint 2.0 scenes. This does not change the attack."""
    path = root / CALIBRATION_RECORDS_RELATIVE
    if not path.is_file():
        return {"present": False, "compared": 0, "n_decision_mismatches": None}
    stored: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("epsilon") != EPSILON_KEY:
            continue
        method = str(row.get("method"))
        steps = int(row.get("steps"))
        if (method, steps) not in {("pgd", 3), ("pgd", 7), ("fgsm", 1), ("random", 1)}:
            continue
        stored[(str(row["scene_id"]), method, steps)] = row
    mismatches = []
    compared = 0
    max_linf = 0.0
    for row in rows:
        key = (str(row["scene_id"]), str(row["method"]), int(row["steps"]))
        previous = stored.get(key)
        if previous is None:
            continue
        compared += 1
        for context in EVAL_CONTEXTS:
            current = _decision((row.get("perturbed") or {}).get(context))
            earlier = _decision((previous.get("perturbed") or {}).get(context))
            if current != earlier:
                mismatches.append(
                    {
                        "scene_id": key[0],
                        "method": key[1],
                        "steps": key[2],
                        "context": context,
                        "this_run": current,
                        "calibration": earlier,
                    }
                )
        current_linf = row.get("linf_after_round_trip")
        earlier_linf = previous.get("linf_after_round_trip")
        if isinstance(current_linf, (int, float)) and isinstance(earlier_linf, (int, float)):
            max_linf = max(max_linf, abs(float(current_linf) - float(earlier_linf)))
    return {
        "present": True,
        "compared": compared,
        "n_decision_mismatches": len(mismatches),
        "decision_mismatches": mismatches[:24],
        "max_abs_linf_after_round_trip": max_linf if compared else None,
    }


def _eligible(
    rows: list[dict[str, Any]],
    scalars: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    by_id = {str(row["scene_id"]): row for row in scalars}
    primary = [
        row
        for row in rows
        if row.get("method") == "pgd" and row.get("epsilon") == EPSILON_KEY and row.get("steps") == PRIMARY_STEPS
    ]
    eligible: list[dict[str, Any]] = []
    reasons: dict[str, int] = {
        "alignment_undefined": 0,
        "margin_not_finite": 0,
        "gradient_l1_not_positive": 0,
    }
    excluded_scenes: list[str] = []
    mismatches: list[str] = []
    for row in primary:
        flags = scene_flags(str(row["answer"]), row.get("clean") or {}, row.get("perturbed") or {})
        if flags["transfer"] is None:
            continue
        scene_id = str(row["scene_id"])
        scalar = by_id.get(scene_id) or {}
        scalar_group = scalar.get("counterfactual_group_id")
        if scalar_group is not None and scalar_group != row.get("counterfactual_group_id"):
            mismatches.append(scene_id)
        failed = _predictor_reasons(scalar)
        if failed:
            for reason in failed:
                reasons[reason] = reasons.get(reason, 0) + 1
            excluded_scenes.append(scene_id)
            continue
        eligible.append(
            {
                "scene_id": scene_id,
                "counterfactual_group_id": str(row["counterfactual_group_id"]),
                "T": 1 if flags["transfer"] is True else 0,
                "alignment": float(scalar["alignment_neutral_to_helpful"]),
                "neutral_margin": float(scalar["neutral_margin"]),
                "helpful_margin": float(scalar["helpful_margin"]),
                "neutral_gradient_l1": float(scalar["neutral_gradient_l1"]),
            }
        )
    reasons["n_scenes"] = len(excluded_scenes)
    reasons["scene_ids"] = excluded_scenes
    return eligible, reasons, mismatches


def _predictor_reasons(scalar: Mapping[str, Any]) -> list[str]:
    reasons = []
    alignment = _finite(scalar.get("alignment_neutral_to_helpful"))
    if alignment is None or alignment < -1.0 or alignment > 1.0:
        reasons.append("alignment_undefined")
    if _finite(scalar.get("neutral_margin")) is None or _finite(scalar.get("helpful_margin")) is None:
        reasons.append("margin_not_finite")
    gradient_l1 = _finite(scalar.get("neutral_gradient_l1"))
    if gradient_l1 is None or gradient_l1 <= 0.0:
        reasons.append("gradient_l1_not_positive")
    return reasons


def _eligible_summary(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    n_transfer = sum(int(row["T"]) for row in eligible)
    n_fail = len(eligible) - n_transfer
    groups = sorted({row["counterfactual_group_id"] for row in eligible})
    mean = None if not eligible else n_transfer / len(eligible)
    return {
        "n_eligible": len(eligible),
        "n_transfer": n_transfer,
        "n_fail": n_fail,
        "n_groups": len(groups),
        "group_ids": groups,
        "outcome_mean": mean,
        "min_transfer_failures": MIN_EVENTS,
        "min_transfers": MIN_EVENTS,
        "min_groups": MIN_GROUPS,
    }


def _prefit_checks(eligible: list[dict[str, Any]]) -> dict[str, Any]:
    """Checks that do not fit T. Design checks inspect the predictors only."""
    summary = _eligible_summary(eligible)
    failed: list[str] = []
    if summary["n_fail"] < MIN_EVENTS:
        failed.append("effective_sample")
    if summary["n_groups"] < MIN_GROUPS:
        failed.append("too_few_groups")
    if summary["n_eligible"] == 0 or summary["n_fail"] == 0 or summary["n_transfer"] == 0:
        failed.append("flat_outcome")
    if summary["n_transfer"] < MIN_EVENTS:
        failed.append("insufficient_variation")
    mean = summary["outcome_mean"]
    if mean is not None and (mean < NEAR_FLAT_LOW or mean > NEAR_FLAT_HIGH):
        failed.append("near_flat_outcome")
    details: dict[str, Any] = {"summary": summary, "design_checked": False}
    if eligible:
        design_failed, design_details = _design_checks(eligible)
        failed.extend(design_failed)
        details["design_checked"] = True
        details["design"] = design_details
    ordered = [name for name in CHECK_ORDER if name in failed]
    return {"failed": ordered, "passed": not ordered, "details": details}


def _design_checks(eligible: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    alignment = np.asarray([row["alignment"] for row in eligible], dtype=np.float64)
    neutral = np.asarray([row["neutral_margin"] for row in eligible], dtype=np.float64)
    helpful = np.asarray([row["helpful_margin"] for row in eligible], dtype=np.float64)
    gradient_l1 = np.asarray([row["neutral_gradient_l1"] for row in eligible], dtype=np.float64)
    columns = {
        "alignment": alignment,
        "neutral_margin": neutral,
        "helpful_margin": helpful,
        "neutral_gradient_l1": gradient_l1,
    }
    failed: list[str] = []
    centered = {name: float(np.dot(values - np.mean(values), values - np.mean(values))) for name, values in columns.items()}
    constant = [name for name, total in centered.items() if total <= 1e-18]
    others = np.column_stack([np.ones(len(eligible)), neutral, helpful, gradient_l1])
    residual_ratio = None
    if not constant and float(centered["alignment"]) > 1e-18:
        beta, _, _, _ = np.linalg.lstsq(others, alignment, rcond=None)
        residual = alignment - others @ beta
        rss = float(np.dot(residual, residual))
        residual_ratio = rss / float(centered["alignment"])
        if rss <= FLAT_PREDICTOR_RSS_RATIO * float(centered["alignment"]):
            failed.append("flat_predictor")
    elif constant:
        failed.append("flat_predictor")
    design = np.column_stack([np.ones(len(eligible)), alignment, neutral, helpful, gradient_l1])
    gram = design.T @ design
    condition = float(np.linalg.cond(gram))
    rank = int(np.linalg.matrix_rank(design))
    if rank < N_PARAMETERS or not math.isfinite(condition) or condition > DESIGN_CONDITION_LIMIT:
        failed.append("singular_design")
    return failed, {
        "centered_sum_squares": centered,
        "constant_columns": constant,
        "alignment_residual_rss_ratio": residual_ratio,
        "rank": rank,
        "condition_number": condition if math.isfinite(condition) else None,
        "flat_predictor_rss_ratio": FLAT_PREDICTOR_RSS_RATIO,
        "condition_limit": DESIGN_CONDITION_LIMIT,
    }


def _estimate(eligible: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    ordered = sorted(eligible, key=lambda row: row["scene_id"])
    design = np.column_stack(
        [
            np.ones(len(ordered)),
            np.asarray([row["alignment"] for row in ordered], dtype=np.float64),
            np.asarray([row["neutral_margin"] for row in ordered], dtype=np.float64),
            np.asarray([row["helpful_margin"] for row in ordered], dtype=np.float64),
            np.asarray([row["neutral_gradient_l1"] for row in ordered], dtype=np.float64),
        ]
    )
    outcome = np.asarray([row["T"] for row in ordered], dtype=np.float64)
    clusters = np.asarray([row["counterfactual_group_id"] for row in ordered])
    fitted = fit_linear_design(
        design,
        outcome,
        clusters,
        n_boot=WILD_DRAWS,
        seed=BOOTSTRAP_SEED,
        alpha=RATE_ALPHA,
        check_focal_flat=True,
        focal_index=1,
    )
    if fitted.get("status") != "interval_returned" or not fitted.get("intervals"):
        return None, str(fitted.get("status") or "not_estimated")
    named = {}
    for name, interval in zip(COLUMN_NAMES, fitted["intervals"]):
        named[name] = {
            "estimate": float(interval["estimate"]),
            "ci_low": float(interval["ci_low"]),
            "ci_high": float(interval["ci_high"]),
        }
    beta = named["alignment"]
    logistic = _logistic(design, outcome, clusters)
    return (
        {
            "status": "interval_returned",
            "beta_a": beta["estimate"],
            "ci_low": beta["ci_low"],
            "ci_high": beta["ci_high"],
            "directional_reading_holds": interval_entirely_above_zero(beta["ci_low"], beta["ci_high"]),
            "confirmatory_supported": False,
            "columns": named,
            "zero_residual_fit": bool(fitted["zero_residual_fit"]),
            "n_rows": int(fitted["n_rows"]),
            "n_groups": int(fitted["n_groups"]),
            "n_parameters": int(fitted["n_parameters"]),
            "n_boot": WILD_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "alpha": RATE_ALPHA,
            "weights": "webb6",
            "quantile_method": "linear",
            "unit": "counterfactual_group_id",
            "logistic": logistic,
        },
        None,
    )


def _logistic(design: np.ndarray, outcome: np.ndarray, clusters: np.ndarray) -> dict[str, Any]:
    point, status = _fit_logistic(design, outcome)
    if point is None:
        return {"status": status, "interval": None, "estimate": None}
    samples, n_undefined = _pairs_cluster_logistic(
        design,
        outcome,
        clusters,
        n_boot=WILD_DRAWS,
        seed=BOOTSTRAP_SEED,
        focal_index=1,
    )
    needed = math.ceil(0.95 * WILD_DRAWS)
    if len(samples) < needed:
        return {
            "status": "unstable_resample",
            "interval": None,
            "estimate": float(point[1]),
            "n_defined_replicates": len(samples),
            "n_undefined_replicates": n_undefined,
        }
    low, high = _percentile(np.asarray(samples, dtype=np.float64), RATE_ALPHA)
    return {
        "status": "interval_returned",
        "estimate": float(point[1]),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_defined_replicates": len(samples),
        "n_undefined_replicates": n_undefined,
        "link": "logit",
    }


def _rate_intervals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intervals: dict[str, Any] = {}
    for config_rows in _grouped(rows):
        name = str(config_rows[0]["method"]) + "|" + str(config_rows[0]["epsilon"]) + "|" + str(config_rows[0]["steps"])
        intervals[name] = {}
        for kind in RATE_KINDS:
            indicators, cluster = _indicators(config_rows, kind)
            if len(set(cluster.values())) != N_GROUPS:
                intervals[name][kind] = {
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "undefined_reason": "the resample did not contain 48 counterfactual groups",
                }
                continue
            stat = scene_bootstrap_rate(
                indicators,
                n_boot=RATE_DRAWS,
                seed=BOOTSTRAP_SEED,
                alpha=RATE_ALPHA,
                cluster_of=cluster,
            )
            intervals[name][kind] = {
                "estimate": stat.get("estimate"),
                "ci_low": stat.get("ci_low"),
                "ci_high": stat.get("ci_high"),
                "undefined_reason": stat.get("undefined_reason"),
                "n_boot": RATE_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "n_undefined_replicates": stat.get("n_undefined_replicates"),
                "unit": "counterfactual_group_id",
                "quantile_method": "linear",
            }
    return intervals


def _grouped(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[object, object, object], list[dict[str, Any]]] = {}
    order: list[tuple[object, object, object]] = []
    for row in rows:
        key = (row.get("method"), row.get("epsilon"), row.get("steps"))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    return [buckets[key] for key in order]


def _indicators(rows: list[dict[str, Any]], kind: str) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    indicators: dict[str, tuple[int, int]] = {}
    cluster: dict[str, str] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        cluster[scene_id] = str(row["counterfactual_group_id"])
        flags = scene_flags(str(row["answer"]), row.get("clean") or {}, row.get("perturbed") or {})
        if kind == "source_success":
            if flags["source_clean_correct"] is True and flags["source_flip"] is not None:
                indicators[scene_id] = (1, 1 if flags["source_flip"] else 0)
            else:
                indicators[scene_id] = (0, 0)
        elif kind == "conditional_transfer":
            if flags["transfer"] is True:
                indicators[scene_id] = (1, 1)
            elif flags["transfer"] is False:
                indicators[scene_id] = (1, 0)
            else:
                indicators[scene_id] = (0, 0)
        elif kind == "target_attack_success":
            if flags["target_clean_correct"] is True and flags["target_flip"] is not None:
                indicators[scene_id] = (1, 1 if flags["target_flip"] else 0)
            else:
                indicators[scene_id] = (0, 0)
        else:
            raise ValueError(kind)
    return indicators, cluster


def _public_counts(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public = []
    for block in blocks:
        public.append(
            {
                "configuration": block["configuration"],
                "role": block["role"],
                "method": block["method"],
                "epsilon": block["epsilon"],
                "steps": block["steps"],
                "n_scenes": block["n_scenes"],
                "n_source_clean_correct": block["n_source_clean_correct"],
                "n_source_not_clean": block["n_source_not_clean"],
                "n_source_clean_missing": block["n_source_clean_missing"],
                "n_source_success": block["n_source_success"],
                "n_source_failure": block["n_source_failure"],
                "n_source_missing": block["n_source_missing"],
                "n_source_success_denominator": block["n_source_success_denominator"],
                "n_conditional_transfer": block["n_conditional_transfer"],
                "n_conditional_transfer_failure": block["n_conditional_transfer_failure"],
                "n_conditional_transfer_denominator": block["n_conditional_transfer_denominator"],
                "n_excluded_target_not_clean": block["n_excluded_target_not_clean"],
                "n_transfer_missing": block["n_transfer_missing"],
                "n_target_clean_correct": block["n_target_clean_correct"],
                "n_target_flip": block["n_target_flip"],
                "n_target_failure": block.get("n_target_failure"),
                "n_target_flip_denominator": block["n_target_flip_denominator"],
                "n_target_missing": block["n_target_missing"],
            }
        )
    return public


def _gradient_comparison(scenes: Mapping[str, Any], scalars: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["scene_id"]): row for row in scalars}
    differences = []
    missing = 0
    worst = None
    for scene_id, scene in scenes.items():
        if not isinstance(scene, dict):
            continue
        recomputed = _finite(scene.get("recomputed_neutral_gradient_l1"))
        stored = _finite((by_id.get(str(scene_id)) or {}).get("neutral_gradient_l1"))
        if recomputed is None or stored is None:
            missing += 1
            continue
        gap = abs(recomputed - stored)
        differences.append(gap)
        if worst is None or gap > worst[0]:
            worst = (gap, str(scene_id), recomputed, stored)
    return {
        "covariate": "stored Checkpoint 1.8 neutral gradient L1",
        "n_compared": len(differences),
        "n_missing": missing,
        "max_abs": None if not differences else float(max(differences)),
        "median_abs": None if not differences else float(np.median(np.asarray(differences))),
        "worst_scene_id": None if worst is None else worst[1],
        "worst_recomputed": None if worst is None else worst[2],
        "worst_stored": None if worst is None else worst[3],
    }


def _margin_comparison(scenes: Mapping[str, Any], scalars: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["scene_id"]): row for row in scalars}
    report: dict[str, Any] = {}
    for context, field in ((SOURCE_CONTEXT, "neutral_margin"), (TARGET_CONTEXT, "helpful_margin")):
        differences = []
        missing = 0
        for scene_id, scene in scenes.items():
            if not isinstance(scene, dict):
                continue
            clean = (scene.get("clean") or {}).get(context) or {}
            recomputed = _finite(clean.get("margin") if isinstance(clean, dict) else None)
            stored = _finite((by_id.get(str(scene_id)) or {}).get(field))
            if recomputed is None or stored is None:
                missing += 1
                continue
            differences.append(abs(recomputed - stored))
        report[field] = {
            "n_compared": len(differences),
            "n_missing": missing,
            "max_abs": None if not differences else float(max(differences)),
            "median_abs": None if not differences else float(np.median(np.asarray(differences))),
        }
    return report


def _decision(view: object) -> str | None:
    if not isinstance(view, Mapping):
        return None
    decision = view.get("decision")
    if isinstance(decision, str) and decision in COLORS:
        return decision
    return None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _sha256(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_ready(value.item())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return value


def main() -> None:
    root = _repo_root(Path(__file__).resolve())
    payload = apply_analysis(root)
    print(
        json.dumps(
            {
                "h_star_estimated": payload["h_star_estimated"],
                "stop_reason": payload["stop_reason"],
                "h_star_statement": payload["h_star_statement"],
                "failed_checks": payload["failed_checks"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _repo_root(start: Path) -> Path:
    from .protocol import PROTOCOL_RELATIVE

    for candidate in (start, *start.parents):
        if (candidate / PROTOCOL_RELATIVE).is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 2.1 protocol is not above {start}")


if __name__ == "__main__":
    main()
