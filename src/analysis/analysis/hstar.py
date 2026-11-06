"""H* estimator on fabricated rows.

H* is an association: among scenes that are clean-correct under both
``neutral`` and ``helpful_nonrevealing``, the sign alignment ``A(s, t)``
between those contexts predicts strict-error transfer of an attack crafted
under one context and scored under the other.

This module scores rows the caller already fabricated. It does not load a
model, does not differentiate an image, does not run an attack, and does not
read the Checkpoint 1 development scenes. A fit of those scenes is refused.
Every returned object sets ``not_an_empirical_finding`` and refuses a
confirmatory claim.

The primary interval is an unrestricted wild cluster bootstrap of the linear
probability model, resampled at ``counterfactual_group_id``. Both ordered
directions are rows. They are not averaged and they are not independent
scenes. A logistic fit with the same covariates is a sensitivity. Invalid
target outputs and valid-wrong target outputs are separate outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import SchemaError, epsilon_key

DATA_ORIGIN = "fabricated"
PRIMARY_CONTEXTS = frozenset({"neutral", "helpful_nonrevealing"})
TASK_FAMILIES = ("attribute", "counting", "spatial")
REFERENCE_TASK = "attribute"
METHODS = frozenset({"pgd", "fgsm", "noise"})
PRIMARY_METHOD = "pgd"
CONTROL_METHODS = ("fgsm", "noise")

# Algebra: the cluster-score matrix has at most G rows. A k-parameter
# covariance needs G >= k + 1 before this module returns an interval.
# For the three-family primary design, k = 4 (intercept, alignment, two
# task indicators), so the minimum is 5 groups.

NEAR_FLAT_LOW = 0.05
NEAR_FLAT_HIGH = 0.95
FLAT_PREDICTOR_RSS_RATIO = 1e-10
DESIGN_CONDITION_LIMIT = 1e12
LOGISTIC_MAX_ABS_COEF = 15.0
LOGISTIC_MIN_WEIGHT = 1e-12
LOGISTIC_MAX_ITER = 25
LOGISTIC_ETA_LIMIT = 30.0
LOGISTIC_DEFINED_FRACTION = 0.95
HIGH_CORRELATION = 0.95
ZERO_RESIDUAL_TOLERANCE = 1e-8

# Cited inclusion count from docs/checkpoint-1/PILOT_REPORT.md.
# This module does not recompute it from model output.
CITED_CHECKPOINT1_ELIGIBLE_GROUPS = 3

# Development scene and group ids. Fitting them would be an empirical estimate.
DEVELOPMENT_SCENE_IDS = frozenset(
    {
        "dev-attribute-00-a",
        "dev-attribute-00-b",
        "dev-attribute-01-a",
        "dev-attribute-01-b",
        "dev-attribute-02-a",
        "dev-attribute-02-b",
        "dev-attribute-03-a",
        "dev-attribute-03-b",
        "dev-counting-00-a",
        "dev-counting-00-b",
        "dev-counting-01-a",
        "dev-counting-01-b",
        "dev-counting-02-a",
        "dev-counting-02-b",
        "dev-counting-03-a",
        "dev-counting-03-b",
        "dev-spatial-00-a",
        "dev-spatial-00-b",
        "dev-spatial-01-a",
        "dev-spatial-01-b",
        "dev-spatial-02-a",
        "dev-spatial-02-b",
        "dev-spatial-03-a",
        "dev-spatial-03-b",
    }
)
DEVELOPMENT_GROUP_IDS = frozenset(
    {
        "cf-attribute-00",
        "cf-attribute-01",
        "cf-attribute-02",
        "cf-attribute-03",
        "cf-counting-00",
        "cf-counting-01",
        "cf-counting-02",
        "cf-counting-03",
        "cf-spatial-00",
        "cf-spatial-01",
        "cf-spatial-02",
        "cf-spatial-03",
    }
)

HSTAR_FIELDS = (
    "data_origin",
    "scene_id",
    "counterfactual_group_id",
    "task_family",
    "source_context",
    "target_context",
    "method",
    "epsilon",
    "step_budget",
    "alignment_a",
    "clean_correct_source",
    "clean_correct_target",
    "source_reveals_answer",
    "target_reveals_answer",
    "source_contains_answer",
    "target_contains_answer",
    "primary_contexts_invariant",
    "ambiguity_flagged",
    "gradients_finite",
    "gradient_source_l1",
    "gradient_target_l1",
    "clean_confidence",
    "strict_error",
    "invalid_target",
    "valid_wrong_target",
    "source_strict_error",
)

BOOL_FIELDS = (
    "clean_correct_source",
    "clean_correct_target",
    "source_reveals_answer",
    "target_reveals_answer",
    "source_contains_answer",
    "target_contains_answer",
    "primary_contexts_invariant",
    "ambiguity_flagged",
    "gradients_finite",
    "strict_error",
    "invalid_target",
    "valid_wrong_target",
    "source_strict_error",
)

WEBB_WEIGHTS = np.array(
    [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)],
    dtype=np.float64,
)

EXCLUSION_REASONS = (
    "context_outside_primary",
    "not_clean_correct_both",
    "answer_leakage",
    "contexts_not_group_invariant",
    "ambiguity_flagged",
    "gradient_not_finite_or_zero",
    "undefined_alignment",
)

INTERVAL_METHOD = "wild_cluster_bootstrap"
WEIGHT_NAME = "webb6"
QUANTILE_METHOD = "linear"
CLUSTER_UNIT = "counterfactual_group_id"
LOGISTIC_INTERVAL_METHOD = "pairs_cluster_bootstrap"


class HStarRefused(ValueError):
    """The packet is not fabricated, or it names a development scene."""


def minimum_groups(n_parameters: int) -> int:
    """Smallest group count for which a cluster interval is returned.

    The meat of a cluster covariance has rank at most ``G``. This module
    returns an interval only when ``G >= k + 1``.
    """
    if isinstance(n_parameters, bool) or not isinstance(n_parameters, int) or n_parameters < 1:
        raise ValueError("n_parameters must be an integer >= 1")
    return n_parameters + 1


def interval_entirely_above_zero(low: float | None, high: float | None) -> bool:
    """True when both ends exist and the lower end is strictly above 0."""
    if low is None or high is None:
        return False
    return float(low) > 0.0 and float(high) >= float(low)


def interval_entirely_below_zero(low: float | None, high: float | None) -> bool:
    """True when both ends exist and the upper end is strictly below 0."""
    if low is None or high is None:
        return False
    return float(high) < 0.0 and float(high) >= float(low)


def interval_covers_zero(low: float | None, high: float | None) -> bool:
    if low is None or high is None:
        return False
    return float(low) <= 0.0 <= float(high)


def mechanical_reading(
    *,
    identifiable: bool,
    pgd: tuple[float, float] | None,
    fgsm: tuple[float, float] | None,
    noise: tuple[float, float] | None,
    invalid_only_pattern: bool,
    sign_disagreement: bool,
) -> str:
    """Map intervals to the pre-specified reading labels.

    The labels describe the intervals. They are not a confirmatory claim.
    ``pgd``, ``fgsm``, and ``noise`` are ``(ci_low, ci_high)`` or None when
    that interval was not returned.
    """
    if not identifiable:
        return "non_identifiable"
    if fgsm is None or noise is None:
        return "controls_not_supplied"
    if interval_entirely_above_zero(noise[0], noise[1]):
        return "alignment_confounded_with_difficulty"
    if sign_disagreement:
        return "pooled_coefficient_is_not_one_mechanism"
    if invalid_only_pattern:
        return "association_in_invalid_outputs"
    if pgd is None:
        return "non_identifiable"
    if interval_entirely_above_zero(fgsm[0], fgsm[1]) and interval_covers_zero(pgd[0], pgd[1]):
        return "fgsm_above_pgd_covers_zero"
    if interval_entirely_above_zero(pgd[0], pgd[1]):
        return "interval_above_zero"
    if interval_covers_zero(pgd[0], pgd[1]):
        return "interval_covers_zero"
    if interval_entirely_below_zero(pgd[0], pgd[1]):
        return "interval_below_zero"
    return "undetermined"


def fit_hstar(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = 999,
    seed: int = 20261107,
    alpha: float = 0.05,
    reference_task: str = REFERENCE_TASK,
) -> dict[str, Any]:
    """Fit H* on fabricated rows and return a JSON-ready document.

    ``confirmatory_supported`` is always false. Fabricated rows cannot
    support the hypothesis.
    """
    _validate_fit_args(n_boot=n_boot, seed=seed, alpha=alpha, reference_task=reference_task)
    validated = validate_hstar_rows(rows)
    included, exclusion_counts, exclusion_examples = partition_rows(validated)
    by_method = {method: [row for row in included if row["method"] == method] for method in METHODS}
    pgd_rows = by_method[PRIMARY_METHOD]
    budgets = sorted(
        {(epsilon_key(row["epsilon"]), int(row["step_budget"])) for row in pgd_rows}
    )
    multiple_budgets = len(budgets) > 1
    primary = _fit_method_rows(
        pgd_rows,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="strict_error",
        check_focal_flat=True,
    )
    logistic = _fit_logistic_sensitivity(
        pgd_rows,
        n_boot=n_boot,
        seed=seed + 1,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
    )
    invalid_fit = _fit_method_rows(
        pgd_rows,
        n_boot=n_boot,
        seed=seed + 2,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="invalid_target",
        check_focal_flat=True,
    )
    valid_wrong_fit = _fit_method_rows(
        pgd_rows,
        n_boot=n_boot,
        seed=seed + 3,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="valid_wrong_target",
        check_focal_flat=True,
    )
    invalid_only = _invalid_only_pattern(invalid_fit, valid_wrong_fit)
    comparison = _predictor_comparison(
        pgd_rows,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
    )
    stratification = _source_stratification(
        pgd_rows,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=multiple_budgets,
    )
    controls = {
        method: _fit_method_rows(
            by_method[method],
            n_boot=n_boot,
            seed=seed + 9 + offset,
            alpha=alpha,
            reference_task=reference_task,
            multiple_budgets=len(
                {(epsilon_key(row["epsilon"]), int(row["step_budget"])) for row in by_method[method]}
            )
            > 1,
            focal_field="alignment_a",
            focal_name="alignment_a",
            outcome_field="strict_error",
            check_focal_flat=True,
        )
        for offset, method in enumerate(CONTROL_METHODS)
    }
    primary_interval = _focal_ends(primary, "alignment_a")
    identifiable = bool(primary.get("identifiable"))
    reading = mechanical_reading(
        identifiable=identifiable,
        pgd=primary_interval,
        fgsm=_focal_ends(controls["fgsm"], "alignment_a"),
        noise=_focal_ends(controls["noise"], "alignment_a"),
        invalid_only_pattern=invalid_only,
        sign_disagreement=bool(stratification["sign_disagreement"]),
    )
    directions = Counter(_direction_name(row) for row in pgd_rows)
    scenes: dict[str, set[str]] = {}
    for row in pgd_rows:
        scenes.setdefault(row["scene_id"], set()).add(_direction_name(row))
    return {
        "data_origin": DATA_ORIGIN,
        "not_an_empirical_finding": True,
        "empirical_claim": False,
        "confirmatory_supported": False,
        "confirmatory_blocked_reason": "fabricated_data_cannot_support_h_star",
        "hypothesis": "beta_A > 0 on the linear probability model",
        "estimand": (
            "strict-error transfer on alignment and task-family indicators, "
            "clustered by counterfactual_group_id, both ordered directions kept"
        ),
        "reference_task": reference_task,
        "cited_checkpoint1_eligible_groups": CITED_CHECKPOINT1_ELIGIBLE_GROUPS,
        "cited_checkpoint1_note": (
            "The pilot report states that three counterfactual groups were "
            "clean-correct under both primary contexts and that no coefficient "
            "was fit. This number is that citation. It is not recomputed here."
        ),
        "inclusion": {
            "n_rows_input": len(validated),
            "n_included": len(included),
            "n_included_by_method": {method: len(by_method[method]) for method in sorted(METHODS)},
            "n_groups_pgd": len({row["counterfactual_group_id"] for row in pgd_rows}),
            "n_scenes_pgd": len({row["scene_id"] for row in pgd_rows}),
            "n_scenes_with_both_directions": sum(len(names) == 2 for names in scenes.values()),
            "n_scenes_with_one_direction": sum(len(names) == 1 for names in scenes.values()),
            "direction_counts": {
                "neutral_to_helpful": int(directions["neutral_to_helpful"]),
                "helpful_to_neutral": int(directions["helpful_to_neutral"]),
            },
            "pgd_budgets": [
                {"epsilon_key": budget[0], "step_budget": budget[1]} for budget in budgets
            ],
            "multiple_budgets": multiple_budgets,
            "exclusion_counts": {reason: int(exclusion_counts[reason]) for reason in EXCLUSION_REASONS},
            "n_rows_excluded": len(validated) - len(included),
            "exclusion_reason_incidences": int(sum(exclusion_counts.values())),
            "exclusion_examples": exclusion_examples,
        },
        "primary": primary,
        "logistic_sensitivity": logistic,
        "decomposition": {
            "strict_error": primary,
            "invalid_target": invalid_fit,
            "valid_wrong_target": valid_wrong_fit,
            "invalid_only_pattern": invalid_only,
            "note": (
                "invalid_only_pattern is true when the invalid-outcome interval "
                "lies entirely above 0 and the valid-wrong interval does not. "
                "That pattern means a positive association is carried by "
                "unparseable outputs. It is a label on these rows, not a "
                "model measurement."
            ),
        },
        "predictor_comparison": comparison,
        "source_stratification": stratification,
        "controls": controls,
        "mechanical_reading": reading,
        "reading_note": (
            "mechanical_reading applies the pre-specified interval rules to "
            "this fit. confirmatory_supported stays false because the rows "
            "are fabricated."
        ),
        "procedure": {
            "primary_interval": INTERVAL_METHOD,
            "weights": WEIGHT_NAME,
            "logistic_interval": LOGISTIC_INTERVAL_METHOD,
            "quantile_method": QUANTILE_METHOD,
            "unit": CLUSTER_UNIT,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "alpha": float(alpha),
            "seed_offsets": {
                "primary_linear": 0,
                "logistic": 1,
                "invalid_target": 2,
                "valid_wrong_target": 3,
                "clean_confidence": 4,
                "gradient_magnitude": 5,
                "joint_diagnostic": 6,
                "source_success_slice": 7,
                "source_failure_slice": 8,
                "fgsm": 9,
                "noise": 10,
            },
        },
    }


def validate_hstar_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``rows`` or raise if the packet is not allowed."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise SchemaError("rows must be a sequence of objects")
    validated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SchemaError(f"row {index} must be an object")
        extra = sorted(set(row) - set(HSTAR_FIELDS))
        missing = sorted(set(HSTAR_FIELDS) - set(row))
        if extra or missing:
            raise SchemaError(f"row {index} fields extra={extra} missing={missing}")
        checked = _check_row(index, row)
        identity = (
            checked["scene_id"],
            checked["source_context"],
            checked["target_context"],
            checked["method"],
            epsilon_key(checked["epsilon"]),
            checked["step_budget"],
        )
        if identity in seen:
            raise SchemaError(f"row {index} duplicates {identity!r}")
        seen.add(identity)
        validated.append(checked)
    return validated


def exclusion_reasons(row: Mapping[str, Any]) -> list[str]:
    """Inclusion failures for one validated row.

    The outcome fields are not read. Changing strict error, invalid output,
    or valid-wrong does not change this list.
    """
    reasons: list[str] = []
    pair = {row["source_context"], row["target_context"]}
    if pair != PRIMARY_CONTEXTS or row["source_context"] == row["target_context"]:
        reasons.append("context_outside_primary")
    if not (row["clean_correct_source"] and row["clean_correct_target"]):
        reasons.append("not_clean_correct_both")
    if (
        row["source_reveals_answer"]
        or row["target_reveals_answer"]
        or row["source_contains_answer"]
        or row["target_contains_answer"]
    ):
        reasons.append("answer_leakage")
    if not row["primary_contexts_invariant"]:
        reasons.append("contexts_not_group_invariant")
    if row["ambiguity_flagged"]:
        reasons.append("ambiguity_flagged")
    if (
        (not row["gradients_finite"])
        or float(row["gradient_source_l1"]) <= 0.0
        or float(row["gradient_target_l1"]) <= 0.0
    ):
        reasons.append("gradient_not_finite_or_zero")
    if row["alignment_a"] is None:
        reasons.append("undefined_alignment")
    return reasons


def partition_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    """Split validated rows into included rows and exclusion counts.

    A row that fails two gates contributes to both counts. The included list
    preserves input order.
    """
    included: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        reasons = exclusion_reasons(row)
        if reasons:
            for reason in reasons:
                counts[reason] += 1
            if len(examples) < 24:
                examples.append(
                    {
                        "scene_id": row["scene_id"],
                        "method": row["method"],
                        "reasons": list(reasons),
                    }
                )
        else:
            included.append(dict(row))
    for reason in EXCLUSION_REASONS:
        counts.setdefault(reason, 0)
    return included, counts, examples


def fit_linear_design(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    check_focal_flat: bool = True,
    focal_index: int = 1,
) -> dict[str, Any]:
    """Linear probability fit with a wild cluster-bootstrap interval.

    Column ``focal_index`` is the predictor the flat-predictor diagnostic
    refers to. Callers put that predictor in column 1. The interval resamples
    whole clusters with Webb six-point weights. A zero residual vector makes
    every replicate equal the point estimate; ``zero_residual_fit`` records
    that collapse.
    """
    _validate_boot_args(n_boot, seed, alpha)
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    cluster_array = np.asarray(clusters)
    if x_array.ndim != 2 or y_array.ndim != 1 or x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x and y shapes do not match")
    if cluster_array.shape[0] != y_array.shape[0]:
        raise ValueError("clusters must have one label per row")
    if x_array.shape[0] == 0:
        return _empty_design("no_included_rows")
    n_parameters = int(x_array.shape[1])
    if focal_index < 0 or focal_index >= n_parameters:
        raise ValueError("focal_index is outside the design")
    n_groups = int(len(np.unique(cluster_array)))
    needed = minimum_groups(n_parameters)
    outcome_mean = float(np.mean(y_array))
    near_flat = bool(outcome_mean < NEAR_FLAT_LOW or outcome_mean > NEAR_FLAT_HIGH)
    point = _ols_strict(x_array, y_array)
    flat_outcome = bool(np.max(y_array) - np.min(y_array) == 0.0)
    flat_predictor = bool(
        check_focal_flat and _focal_column_is_flat(x_array, focal_index)
    )
    base = {
        "n_rows": int(x_array.shape[0]),
        "n_groups": n_groups,
        "n_parameters": n_parameters,
        "minimum_groups": needed,
        "outcome_mean": outcome_mean,
        "near_flat_outcome": near_flat,
        "flat_outcome": flat_outcome,
        "flat_predictor": flat_predictor,
        "interval_method": INTERVAL_METHOD,
        "weights": WEIGHT_NAME,
        "quantile_method": QUANTILE_METHOD,
        "unit": CLUSTER_UNIT,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
    }
    diagnostic = None if point is None else [float(value) for value in point]
    # A constant outcome, or a focal column with no residual variation, is
    # reported before the singular-design status. A constant alignment column
    # is collinear with the intercept, so the algebra is singular and flat.
    if flat_outcome:
        return _refused(base, "flat_outcome", diagnostic)
    if flat_predictor:
        return _refused(base, "flat_predictor", diagnostic)
    if point is None:
        return {
            **base,
            "status": "singular_design",
            "diagnostic_point_estimate": None,
            "intervals": None,
            "identifiable": False,
            "zero_residual_fit": False,
        }
    if n_groups < needed:
        return _refused(base, "too_few_groups", diagnostic)
    replicated = _wild_replicates(x_array, y_array, cluster_array, point, n_boot=n_boot, seed=seed)
    if replicated is None:
        return {
            **base,
            "status": "singular_design",
            "diagnostic_point_estimate": diagnostic,
            "intervals": None,
            "identifiable": False,
            "zero_residual_fit": False,
        }
    replicates = replicated
    intervals = []
    for index in range(n_parameters):
        low, high = _percentile(replicates[:, index], alpha)
        intervals.append(
            {
                "index": index,
                "estimate": float(point[index]),
                "ci_low": low,
                "ci_high": high,
            }
        )
    residuals = y_array - x_array @ point
    zero_residual = bool(np.max(np.abs(residuals)) <= ZERO_RESIDUAL_TOLERANCE)
    identifiable = not near_flat
    return {
        **base,
        "status": "interval_returned",
        "diagnostic_point_estimate": diagnostic,
        "intervals": intervals,
        "identifiable": identifiable,
        "zero_residual_fit": zero_residual,
    }


def packet_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash of the canonical JSON for a fabricated packet."""
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_method_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    reference_task: str,
    multiple_budgets: bool,
    focal_field: str,
    focal_name: str,
    outcome_field: str,
    check_focal_flat: bool,
) -> dict[str, Any]:
    if not rows:
        result = _named_empty("no_included_rows", focal_name)
        return result
    if multiple_budgets:
        result = _named_empty("multiple_budgets", focal_name)
        result["n_rows"] = len(rows)
        result["n_groups"] = len({row["counterfactual_group_id"] for row in rows})
        return result
    if any(row[focal_field] is None for row in rows):
        result = _named_empty("predictor_not_supplied", focal_name)
        return result
    design = _design_matrix(
        rows,
        focal_field=focal_field,
        outcome_field=outcome_field,
        reference_task=reference_task,
    )
    fitted = fit_linear_design(
        design["x"],
        design["y"],
        design["clusters"],
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        check_focal_flat=check_focal_flat,
        focal_index=1,
    )
    return _attach_names(fitted, design["columns"], focal_name)


def _fit_logistic_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    reference_task: str,
    multiple_budgets: bool,
) -> dict[str, Any]:
    """Logistic coefficient on alignment, with a pairs cluster bootstrap.

    A separated or non-convergent fit returns a null interval. The logistic
    coefficient is a log-odds change. It is not the primary estimand.
    """
    note = (
        "Sensitivity only. The primary estimand is the linear probability "
        "coefficient. A null logistic interval means the fit was separated, "
        "flat, or unstable under cluster resampling."
    )
    if not rows:
        return {"status": "no_included_rows", "interval": None, "note": note}
    if multiple_budgets:
        return {"status": "multiple_budgets", "interval": None, "note": note}
    design = _design_matrix(
        rows,
        focal_field="alignment_a",
        outcome_field="strict_error",
        reference_task=reference_task,
    )
    y_array = design["y"]
    if float(np.max(y_array) - np.min(y_array)) == 0.0:
        return {
            "status": "flat_outcome",
            "interval": None,
            "n_groups": int(len(np.unique(design["clusters"]))),
            "n_parameters": len(design["columns"]),
            "note": note,
        }
    point, status = _fit_logistic(design["x"], y_array)
    if point is None:
        return {
            "status": status,
            "interval": None,
            "n_groups": int(len(np.unique(design["clusters"]))),
            "n_parameters": len(design["columns"]),
            "columns": design["columns"],
            "note": note,
        }
    samples, n_undefined = _pairs_cluster_logistic(
        design["x"],
        y_array,
        design["clusters"],
        n_boot=n_boot,
        seed=seed,
        focal_index=1,
    )
    needed = math.ceil(LOGISTIC_DEFINED_FRACTION * n_boot)
    if len(samples) < needed:
        return {
            "status": "unstable_resample",
            "interval": None,
            "estimate": float(point[1]),
            "n_defined_replicates": len(samples),
            "n_undefined_replicates": n_undefined,
            "n_boot": int(n_boot),
            "columns": design["columns"],
            "note": note,
        }
    low, high = _percentile(np.asarray(samples, dtype=np.float64), alpha)
    return {
        "status": "interval_returned",
        "interval": {
            "alignment_a": {
                "estimate": float(point[1]),
                "ci_low": low,
                "ci_high": high,
            }
        },
        "estimate": float(point[1]),
        "n_defined_replicates": len(samples),
        "n_undefined_replicates": n_undefined,
        "n_groups": int(len(np.unique(design["clusters"]))),
        "n_parameters": len(design["columns"]),
        "columns": design["columns"],
        "interval_method": LOGISTIC_INTERVAL_METHOD,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "alpha": float(alpha),
        "quantile_method": QUANTILE_METHOD,
        "unit": CLUSTER_UNIT,
        "link": "logit",
        "note": note,
    }


def _predictor_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    reference_task: str,
    multiple_budgets: bool,
) -> dict[str, Any]:
    """Separate cluster intervals for alignment, confidence, and magnitude.

    Magnitude is the target gradient L1 already stored on the row. Confidence
    is the caller-supplied clean probability of the true answer. The three
    coefficients use different units. None of the alternatives is added to
    the primary model.
    """
    note = (
        "Each predictor is fit in its own linear model with the same task "
        "indicators and the same clusters. Coefficients are not in the same "
        "units. A larger coefficient is not a better predictor. The joint "
        "fit is a collinearity diagnostic and is not a primary estimand."
    )
    if not rows or multiple_budgets:
        return {
            "status": "no_included_rows" if not rows else "multiple_budgets",
            "note": note,
        }
    alignment = _fit_method_rows(
        rows,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=False,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="strict_error",
        check_focal_flat=True,
    )
    confidence_rows = [row for row in rows if row["clean_confidence"] is not None]
    if len(confidence_rows) != len(rows):
        confidence: dict[str, Any] = {"status": "predictor_not_supplied", "interval": None}
    else:
        confidence = _fit_method_rows(
            rows,
            n_boot=n_boot,
            seed=seed + 4,
            alpha=alpha,
            reference_task=reference_task,
            multiple_budgets=False,
            focal_field="clean_confidence",
            focal_name="clean_confidence",
            outcome_field="strict_error",
            check_focal_flat=True,
        )
    magnitude = _fit_method_rows(
        rows,
        n_boot=n_boot,
        seed=seed + 5,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=False,
        focal_field="gradient_target_l1",
        focal_name="gradient_magnitude",
        outcome_field="strict_error",
        check_focal_flat=True,
    )
    alignment_values = np.asarray([float(row["alignment_a"]) for row in rows], dtype=np.float64)
    magnitude_values = np.asarray(
        [float(row["gradient_target_l1"]) for row in rows], dtype=np.float64
    )
    if len(confidence_rows) == len(rows):
        confidence_values = np.asarray(
            [float(row["clean_confidence"]) for row in rows], dtype=np.float64
        )
        corr_confidence = _pearson(alignment_values, confidence_values)
    else:
        corr_confidence = None
    corr_magnitude = _pearson(alignment_values, magnitude_values)
    high = []
    if corr_confidence is not None and abs(corr_confidence) >= HIGH_CORRELATION:
        high.append("clean_confidence")
    if corr_magnitude is not None and abs(corr_magnitude) >= HIGH_CORRELATION:
        high.append("gradient_magnitude")
    joint = _joint_diagnostic(
        rows,
        n_boot=n_boot,
        seed=seed + 6,
        alpha=alpha,
        reference_task=reference_task,
        confidence_supplied=len(confidence_rows) == len(rows),
    )
    return {
        "status": "compared",
        "alignment_a": alignment,
        "clean_confidence": confidence,
        "gradient_magnitude": magnitude,
        "pearson_alignment_confidence": corr_confidence,
        "pearson_alignment_magnitude": corr_magnitude,
        "high_correlation": high,
        "high_correlation_threshold": HIGH_CORRELATION,
        "joint_diagnostic": joint,
        "note": note,
    }


def _joint_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    reference_task: str,
    confidence_supplied: bool,
) -> dict[str, Any]:
    if not confidence_supplied:
        return {"status": "predictor_not_supplied", "not_a_primary_estimand": True}
    families, baseline, indicators = _task_coding(rows, reference_task)
    columns = [
        "intercept",
        "alignment_a",
        "clean_confidence",
        "gradient_magnitude",
        *[f"task_{family}" for family in indicators],
    ]
    x_matrix = np.zeros((len(rows), len(columns)), dtype=np.float64)
    x_matrix[:, 0] = 1.0
    x_matrix[:, 1] = [float(row["alignment_a"]) for row in rows]
    x_matrix[:, 2] = [float(row["clean_confidence"]) for row in rows]
    x_matrix[:, 3] = [float(row["gradient_target_l1"]) for row in rows]
    for offset, family in enumerate(indicators):
        x_matrix[:, 4 + offset] = [
            1.0 if row["task_family"] == family else 0.0 for row in rows
        ]
    y_array = np.asarray([1.0 if row["strict_error"] else 0.0 for row in rows], dtype=np.float64)
    clusters = np.asarray([row["counterfactual_group_id"] for row in rows])
    fitted = fit_linear_design(
        x_matrix,
        y_array,
        clusters,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        check_focal_flat=False,
        focal_index=1,
    )
    named = _attach_names(fitted, columns, "alignment_a")
    named["not_a_primary_estimand"] = True
    named["task_baseline"] = baseline
    named["families_present"] = families
    return named


def _source_stratification(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    reference_task: str,
    multiple_budgets: bool,
) -> dict[str, Any]:
    """Fit β_A inside the source-success slice and the source-failure slice.

    Source success is strict error under the source context. It is not a
    covariate in the pooled model. Disagreeing signs mean the pooled
    coefficient is not one mechanism.
    """
    note = (
        "source_strict_error is not a regressor. The two slices are separate "
        "fits. A sign disagreement blocks reading the pooled coefficient as "
        "one mechanism."
    )
    if not rows or multiple_budgets:
        return {
            "status": "no_included_rows" if not rows else "multiple_budgets",
            "sign_disagreement": False,
            "note": note,
        }
    success = [row for row in rows if row["source_strict_error"]]
    failure = [row for row in rows if not row["source_strict_error"]]
    success_fit = _fit_method_rows(
        success,
        n_boot=n_boot,
        seed=seed + 7,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=False,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="strict_error",
        check_focal_flat=True,
    )
    failure_fit = _fit_method_rows(
        failure,
        n_boot=n_boot,
        seed=seed + 8,
        alpha=alpha,
        reference_task=reference_task,
        multiple_budgets=False,
        focal_field="alignment_a",
        focal_name="alignment_a",
        outcome_field="strict_error",
        check_focal_flat=True,
    )
    success_estimate = _returned_focal_point(success_fit, "alignment_a")
    failure_estimate = _returned_focal_point(failure_fit, "alignment_a")
    disagreement = _signs_disagree(success_estimate, failure_estimate)
    if success_estimate is None or failure_estimate is None:
        status = "incomplete"
    elif disagreement:
        status = "sign_disagreement"
    else:
        status = "signs_do_not_disagree"
    return {
        "status": status,
        "source_success": success_fit,
        "source_failure": failure_fit,
        "sign_disagreement": disagreement,
        "note": note,
    }


def _design_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    focal_field: str,
    outcome_field: str,
    reference_task: str,
) -> dict[str, Any]:
    _families, baseline, indicators = _task_coding(rows, reference_task)
    focal_name = "gradient_magnitude" if focal_field == "gradient_target_l1" else focal_field
    columns = ["intercept", focal_name, *[f"task_{family}" for family in indicators]]
    x_matrix = np.zeros((len(rows), len(columns)), dtype=np.float64)
    x_matrix[:, 0] = 1.0
    x_matrix[:, 1] = [float(row[focal_field]) for row in rows]
    for offset, family in enumerate(indicators):
        x_matrix[:, 2 + offset] = [
            1.0 if row["task_family"] == family else 0.0 for row in rows
        ]
    y_array = np.asarray([1.0 if row[outcome_field] else 0.0 for row in rows], dtype=np.float64)
    clusters = np.asarray([row["counterfactual_group_id"] for row in rows])
    return {
        "x": x_matrix,
        "y": y_array,
        "clusters": clusters,
        "columns": columns,
        "baseline": baseline,
    }


def _task_coding(
    rows: Sequence[Mapping[str, Any]], reference_task: str
) -> tuple[list[str], str, list[str]]:
    present = [family for family in TASK_FAMILIES if any(row["task_family"] == family for row in rows)]
    if reference_task in present:
        baseline = reference_task
    elif present:
        baseline = present[0]
    else:
        baseline = reference_task
    indicators = [family for family in present if family != baseline]
    return present, baseline, indicators


def _wild_replicates(
    x_array: np.ndarray,
    y_array: np.ndarray,
    clusters: np.ndarray,
    point: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray | None:
    """Unrestricted wild cluster bootstrap replicates of the OLS coefficient.

    ``β* = β + (X'X)^{-1} Σ_g w_g X_g' u_g`` with ``w_g`` drawn from the Webb
    six-point set. Weights are constant inside a cluster. Clusters are taken
    in sorted order, and the seed draws weights in that order.
    """
    xtx = x_array.T @ x_array
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None
    residuals = y_array - x_array @ point
    unique, inverse = np.unique(clusters, return_inverse=True)
    n_groups = int(unique.shape[0])
    n_parameters = int(x_array.shape[1])
    scores = np.zeros((n_groups, n_parameters), dtype=np.float64)
    for index in range(n_groups):
        mask = inverse == index
        scores[index] = x_array[mask].T @ residuals[mask]
    draw = np.random.default_rng(seed).integers(0, WEBB_WEIGHTS.shape[0], size=(n_boot, n_groups))
    weights = WEBB_WEIGHTS[draw]
    delta = weights @ scores
    return point + delta @ xtx_inv.T


def _fit_logistic(x_array: np.ndarray, y_array: np.ndarray) -> tuple[np.ndarray | None, str]:
    """IRLS logistic fit. Returns ``(None, status)`` on separation or failure."""
    beta = np.zeros(x_array.shape[1], dtype=np.float64)
    for _ in range(LOGISTIC_MAX_ITER):
        eta = x_array @ beta
        if float(np.max(np.abs(eta))) > LOGISTIC_ETA_LIMIT:
            return None, "separation"
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = probability * (1.0 - probability)
        if float(np.min(weight)) < LOGISTIC_MIN_WEIGHT:
            return None, "separation"
        working = eta + (y_array - probability) / weight
        scaled = np.sqrt(weight)
        nxt = _ols_strict(x_array * scaled[:, None], working * scaled)
        if nxt is None:
            return None, "singular"
        if float(np.max(np.abs(nxt))) > LOGISTIC_MAX_ABS_COEF:
            return None, "separation"
        if float(np.max(np.abs(nxt - beta))) < 1e-8:
            return nxt, "converged"
        beta = nxt
    return None, "not_converged"


def _pairs_cluster_logistic(
    x_array: np.ndarray,
    y_array: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    focal_index: int,
) -> tuple[list[float], int]:
    unique, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(unique.shape[0])]
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(n_boot, len(groups)))
    samples: list[float] = []
    n_undefined = 0
    for replicate in range(n_boot):
        index = np.concatenate([groups[int(choice)] for choice in draws[replicate]])
        beta, status = _fit_logistic(x_array[index], y_array[index])
        if beta is None or status != "converged":
            n_undefined += 1
            continue
        samples.append(float(beta[focal_index]))
    return samples, n_undefined


def _ols_strict(x_array: np.ndarray, y_array: np.ndarray) -> np.ndarray | None:
    xtx = x_array.T @ x_array
    condition = float(np.linalg.cond(xtx))
    if not math.isfinite(condition) or condition > DESIGN_CONDITION_LIMIT:
        return None
    try:
        return np.linalg.solve(xtx, x_array.T @ y_array)
    except np.linalg.LinAlgError:
        return None


def _focal_column_is_flat(x_array: np.ndarray, focal_index: int) -> bool:
    focal = x_array[:, focal_index]
    centered = focal - float(np.mean(focal))
    total = float(np.dot(centered, centered))
    if total <= 1e-18:
        return True
    others = np.concatenate(
        [x_array[:, :focal_index], x_array[:, focal_index + 1 :]],
        axis=1,
    )
    if others.shape[1] == 0:
        return False
    beta, _, _, _ = np.linalg.lstsq(others, focal, rcond=None)
    residual = focal - others @ beta
    rss = float(np.dot(residual, residual))
    return rss <= FLAT_PREDICTOR_RSS_RATIO * total


def _percentile(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], method=QUANTILE_METHOD)
    return float(low), float(high)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _attach_names(fitted: dict[str, Any], columns: list[str], focal_name: str) -> dict[str, Any]:
    estimates = fitted.get("diagnostic_point_estimate")
    named_estimate = None
    if estimates is not None:
        named_estimate = {column: estimates[index] for index, column in enumerate(columns)}
    intervals = fitted.get("intervals")
    named_intervals = None
    if intervals is not None:
        named_intervals = {}
        for column, interval in zip(columns, intervals):
            named_intervals[column] = {
                "estimate": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
            }
    return {
        "status": fitted["status"],
        "focal": focal_name,
        "columns": list(columns),
        "n_rows": fitted["n_rows"],
        "n_groups": fitted["n_groups"],
        "n_parameters": fitted["n_parameters"],
        "minimum_groups": fitted["minimum_groups"],
        "outcome_mean": fitted["outcome_mean"],
        "near_flat_outcome": fitted["near_flat_outcome"],
        "flat_outcome": fitted["flat_outcome"],
        "flat_predictor": fitted["flat_predictor"],
        "diagnostic_point_estimate": named_estimate,
        "interval": named_intervals,
        "identifiable": fitted["identifiable"] and fitted["status"] == "interval_returned",
        "zero_residual_fit": fitted["zero_residual_fit"],
        "interval_method": fitted["interval_method"],
        "weights": fitted["weights"],
        "quantile_method": fitted["quantile_method"],
        "unit": fitted["unit"],
        "n_boot": fitted["n_boot"],
        "seed": fitted["seed"],
        "alpha": fitted["alpha"],
        "evidence": False,
    }


def _refused(base: dict[str, Any], status: str, diagnostic: list[float]) -> dict[str, Any]:
    return {
        **base,
        "status": status,
        "diagnostic_point_estimate": diagnostic,
        "intervals": None,
        "identifiable": False,
        "zero_residual_fit": False,
    }


def _empty_design(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "n_rows": 0,
        "n_groups": 0,
        "n_parameters": 0,
        "minimum_groups": None,
        "outcome_mean": None,
        "near_flat_outcome": False,
        "flat_outcome": False,
        "flat_predictor": False,
        "diagnostic_point_estimate": None,
        "intervals": None,
        "identifiable": False,
        "zero_residual_fit": False,
        "interval_method": INTERVAL_METHOD,
        "weights": WEIGHT_NAME,
        "quantile_method": QUANTILE_METHOD,
        "unit": CLUSTER_UNIT,
        "n_boot": None,
        "seed": None,
        "alpha": None,
    }


def _named_empty(status: str, focal_name: str) -> dict[str, Any]:
    empty = _empty_design(status)
    empty["focal"] = focal_name
    empty["columns"] = []
    empty["interval"] = None
    empty["evidence"] = False
    empty.pop("intervals", None)
    return empty


def _focal_ends(fit: Mapping[str, Any], name: str) -> tuple[float, float] | None:
    interval = fit.get("interval")
    if not isinstance(interval, dict) or name not in interval:
        return None
    block = interval[name]
    if fit.get("status") != "interval_returned":
        return None
    if not fit.get("identifiable", False):
        return None
    return float(block["ci_low"]), float(block["ci_high"])


def _focal_point(fit: Mapping[str, Any], name: str) -> float | None:
    estimate = fit.get("diagnostic_point_estimate")
    if not isinstance(estimate, dict) or name not in estimate:
        return None
    if fit.get("status") in {"singular_design", "no_included_rows", "multiple_budgets"}:
        return None
    return float(estimate[name])


def _returned_focal_point(fit: Mapping[str, Any], name: str) -> float | None:
    """Point estimate used for the source-slice sign check.

    A slice that did not return an interval does not contribute a sign.
    Algebra from a refused fit is not a second mechanism.
    """
    if fit.get("status") != "interval_returned":
        return None
    return _focal_point(fit, name)


def _signs_disagree(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    if left == 0.0 or right == 0.0:
        return False
    return (left > 0.0) != (right > 0.0)


def _invalid_only_pattern(invalid_fit: Mapping[str, Any], valid_wrong_fit: Mapping[str, Any]) -> bool:
    invalid_ends = _focal_ends(invalid_fit, "alignment_a")
    valid_ends = _focal_ends(valid_wrong_fit, "alignment_a")
    invalid_above = (
        invalid_ends is not None and interval_entirely_above_zero(invalid_ends[0], invalid_ends[1])
    )
    valid_above = (
        valid_ends is not None and interval_entirely_above_zero(valid_ends[0], valid_ends[1])
    )
    return bool(invalid_above and not valid_above)


def _direction_name(row: Mapping[str, Any]) -> str:
    if row["source_context"] == "neutral" and row["target_context"] == "helpful_nonrevealing":
        return "neutral_to_helpful"
    if row["source_context"] == "helpful_nonrevealing" and row["target_context"] == "neutral":
        return "helpful_to_neutral"
    return "other"


def _check_row(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    origin = row["data_origin"]
    if origin != DATA_ORIGIN:
        raise HStarRefused(
            f"row {index} data_origin must be {DATA_ORIGIN!r}; empirical rows are refused"
        )
    scene_id = _nonempty_str(row["scene_id"], f"row {index} scene_id")
    group_id = _nonempty_str(
        row["counterfactual_group_id"], f"row {index} counterfactual_group_id"
    )
    if scene_id in DEVELOPMENT_SCENE_IDS or scene_id.startswith("dev-"):
        raise HStarRefused(
            f"row {index} scene_id {scene_id!r} is a development scene; H* is not fit to it"
        )
    if group_id in DEVELOPMENT_GROUP_IDS:
        raise HStarRefused(
            f"row {index} counterfactual_group_id {group_id!r} is a development group"
        )
    family = row["task_family"]
    if family not in TASK_FAMILIES:
        raise SchemaError(f"row {index} task_family {family!r} is not one of {TASK_FAMILIES}")
    for context_field in ("source_context", "target_context"):
        if row[context_field] not in {"neutral", "helpful_nonrevealing", "contradictory", "answer_revealing"}:
            raise SchemaError(f"row {index} {context_field} is not a known context")
    if row["source_context"] == row["target_context"]:
        raise SchemaError(f"row {index} source and target contexts must differ")
    method = row["method"]
    if method not in METHODS:
        raise SchemaError(f"row {index} method {method!r} is outside pgd, fgsm, and noise")
    epsilon = _finite_number(row["epsilon"], f"row {index} epsilon")
    if epsilon < 0:
        raise SchemaError(f"row {index} epsilon must be >= 0")
    step_budget = row["step_budget"]
    if isinstance(step_budget, bool) or not isinstance(step_budget, int) or step_budget < 1:
        raise SchemaError(f"row {index} step_budget must be an integer >= 1")
    for field in BOOL_FIELDS:
        if not isinstance(row[field], bool):
            raise SchemaError(f"row {index} {field} must be a boolean")
    alignment = row["alignment_a"]
    if alignment is not None:
        alignment = _finite_number(alignment, f"row {index} alignment_a")
        if alignment < -1.0 - 1e-9 or alignment > 1.0 + 1e-9:
            raise SchemaError(f"row {index} alignment_a must lie in [-1, 1] or be null")
    for field in ("gradient_source_l1", "gradient_target_l1"):
        value = _finite_number(row[field], f"row {index} {field}")
        if value < 0:
            raise SchemaError(f"row {index} {field} must be >= 0")
    confidence = row["clean_confidence"]
    if confidence is not None:
        confidence = _finite_number(confidence, f"row {index} clean_confidence")
        if confidence <= 0.0 or confidence > 1.0:
            raise SchemaError(f"row {index} clean_confidence must lie in (0, 1] or be null")
    strict = row["strict_error"]
    invalid = row["invalid_target"]
    valid_wrong = row["valid_wrong_target"]
    if strict:
        if invalid == valid_wrong:
            raise SchemaError(
                f"row {index} strict error must be exactly one of invalid_target or valid_wrong_target"
            )
    elif invalid or valid_wrong:
        raise SchemaError(f"row {index} a correct target cannot be invalid or valid-wrong")
    checked = dict(row)
    checked["scene_id"] = scene_id
    checked["counterfactual_group_id"] = group_id
    checked["epsilon"] = epsilon
    checked["alignment_a"] = alignment
    checked["clean_confidence"] = confidence
    checked["gradient_source_l1"] = float(row["gradient_source_l1"])
    checked["gradient_target_l1"] = float(row["gradient_target_l1"])
    return checked


def _validate_fit_args(*, n_boot: int, seed: int, alpha: float, reference_task: str) -> None:
    _validate_boot_args(n_boot, seed, alpha)
    if reference_task not in TASK_FAMILIES:
        raise ValueError(f"reference_task must be one of {TASK_FAMILIES}")


def _validate_boot_args(n_boot: int, seed: int, alpha: float) -> None:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")


def _nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or value == "":
        raise SchemaError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"{label} must be a finite number")
    return number
