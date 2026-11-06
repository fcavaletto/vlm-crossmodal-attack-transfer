"""Precision of a cluster interval under a fabricated one-family design.

The draws are not model outputs and they are not the development scenes.
One uniform number is drawn for each counterfactual group and compared with
every row probability, so sibling scenes and both ordered directions move
together. Coverage of the known coefficient and the share of intervals whose
lower endpoint lies above zero are different counts. The second count is the
one the H* reading uses when it asks whether the interval lies above zero.

This module refuses a row whose scene id or group id looks like a development
id. It does not read the Checkpoint 1.6 predictions.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

DATA_ORIGIN = "fabricated_dgp"
BETA_0 = 0.50
BETA_A = 0.25
ALIGN_LOW = -0.5
ALIGN_HIGH = 0.5
SCENES_PER_GROUP = 2
N_DIRECTIONS = 2
N_PARAMETERS = 2
ALGEBRAIC_MINIMUM_GROUPS = N_PARAMETERS + 1
NEAR_FLAT_LOW = 0.05
NEAR_FLAT_HIGH = 0.95
DEFAULT_GRID = (9, 12, 24, 48)
DEFAULT_REPLICATIONS = 200
DEFAULT_BOOT = 399
DEFAULT_SEED = 20261108
WEBB_WEIGHTS = np.array(
    [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)],
    dtype=np.float64,
)


class RealDataRefused(ValueError):
    """The row names a development scene or is not marked fabricated."""


def simulate_precision(
    *,
    n_groups_grid: Sequence[int] = DEFAULT_GRID,
    n_replications: int = DEFAULT_REPLICATIONS,
    n_boot: int = DEFAULT_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    scenes_per_group: int = SCENES_PER_GROUP,
) -> dict[str, Any]:
    """Replicate the one-family cluster interval at several group counts."""

    _check_args(n_groups_grid, n_replications, n_boot, seed, alpha, scenes_per_group)
    data_rng = np.random.default_rng(seed)
    by_n = []
    for n_groups in n_groups_grid:
        by_n.append(
            _one_size(
                data_rng,
                n_groups=int(n_groups),
                n_replications=n_replications,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
                scenes_per_group=scenes_per_group,
            )
        )
    return {
        "data_origin": DATA_ORIGIN,
        "empirical_claim": False,
        "not_an_empirical_finding": True,
        "h_star_fitted_on_real_data": False,
        "model_inference": False,
        "known_beta_a": BETA_A,
        "known_intercept": BETA_0,
        "design": "one_family_linear_probability",
        "n_parameters": N_PARAMETERS,
        "columns": ["intercept", "alignment_a"],
        "algebraic_minimum_groups": ALGEBRAIC_MINIMUM_GROUPS,
        "scenes_per_group": scenes_per_group,
        "directions_per_scene": N_DIRECTIONS,
        "sampling_unit": "counterfactual_group",
        "dependence": (
            "One Uniform(0, 1) draw per counterfactual group is compared with "
            "each row probability. Sibling scenes and both ordered directions share that draw."
        ),
        "interval_method": "wild_cluster_bootstrap",
        "weights": "webb6",
        "quantile_method": "linear",
        "excludes_zero_rule": "lower endpoint strictly greater than 0",
        "seed": seed,
        "n_replications": n_replications,
        "n_boot": n_boot,
        "alpha": alpha,
        "by_n_groups": by_n,
        "twenty_four_groups_automatically_enough": False,
        "reading": _reading(by_n),
    }


def fit_fabricated_rows(
    rows: Sequence[dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit the one-family interval on rows the caller marks fabricated.

    Development scene ids and development group ids are refused before any
    coefficient is computed.
    """

    if not rows:
        raise ValueError("rows are empty")
    for row in rows:
        _refuse_real_row(row)
    alignment = np.asarray([float(row["alignment_a"]) for row in rows], dtype=np.float64)
    outcome = np.asarray([1.0 if row["strict_error"] else 0.0 for row in rows], dtype=np.float64)
    clusters = np.asarray([row["counterfactual_group_id"] for row in rows])
    design = np.column_stack([np.ones(len(rows)), alignment])
    return fit_design(design, outcome, clusters, n_boot=n_boot, seed=seed, alpha=alpha)


def fit_design(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Wild cluster interval for the alignment column. Fabricated numbers only."""

    n_groups = int(np.unique(clusters).shape[0])
    if n_groups < ALGEBRAIC_MINIMUM_GROUPS:
        return {
            "status": "too_few_groups",
            "interval": None,
            "evidence": False,
            "n_groups": n_groups,
            "n_parameters": N_PARAMETERS,
            "algebraic_minimum_groups": ALGEBRAIC_MINIMUM_GROUPS,
            "identifiable": False,
        }
    if float(np.max(outcome) - np.min(outcome)) == 0.0:
        return {
            "status": "flat_outcome",
            "interval": None,
            "evidence": False,
            "n_groups": n_groups,
            "n_parameters": N_PARAMETERS,
            "identifiable": False,
        }
    if float(np.std(design[:, 1])) == 0.0:
        return {
            "status": "flat_predictor",
            "interval": None,
            "evidence": False,
            "n_groups": n_groups,
            "n_parameters": N_PARAMETERS,
            "identifiable": False,
        }
    point = _ols(design, outcome)
    if point is None:
        return {
            "status": "singular_design",
            "interval": None,
            "evidence": False,
            "n_groups": n_groups,
            "n_parameters": N_PARAMETERS,
            "identifiable": False,
        }
    replicates = _wild_replicates(design, outcome, clusters, point, n_boot=n_boot, seed=seed)
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    residual = outcome - design @ point
    zero_residual = bool(np.max(np.abs(residual)) <= 1e-8)
    mean_outcome = float(np.mean(outcome))
    near_flat = mean_outcome < NEAR_FLAT_LOW or mean_outcome > NEAR_FLAT_HIGH
    return {
        "status": "interval_returned",
        "interval": {"estimate": float(point[1]), "ci_low": float(low), "ci_high": float(high)},
        "evidence": False,
        "n_groups": n_groups,
        "n_parameters": N_PARAMETERS,
        "identifiable": not near_flat,
        "near_flat_outcome": near_flat,
        "zero_residual_fit": zero_residual,
        "excludes_zero": lower_endpoint_excludes_zero(float(low)),
        "h_star_fitted_on_real_data": False,
    }


def lower_endpoint_excludes_zero(ci_low: float) -> bool:
    """The H* reading counts an interval as above zero only when the endpoint is strictly greater."""

    return float(ci_low) > 0.0


def worked_zero_residual_example(*, n_boot: int = 40, seed: int = 7) -> dict[str, Any]:
    """Six fabricated rows on the line strict_error = 1/2 + alignment.

    Residuals are zero, so the interval collapses to the point. The collapse
    checks the arithmetic. It is not a picture of sampling uncertainty.
    """

    rows = []
    for group in range(3):
        for arm, alignment in enumerate((-0.5, 0.5)):
            rows.append(
                {
                    "data_origin": "fabricated",
                    "scene_id": f"fab-g{group}-s{arm}",
                    "counterfactual_group_id": f"fab-g{group}",
                    "alignment_a": alignment,
                    "strict_error": alignment > 0,
                }
            )
    fitted = fit_fabricated_rows(rows, n_boot=n_boot, seed=seed)
    return fitted


def _one_size(
    data_rng: np.random.Generator,
    *,
    n_groups: int,
    n_replications: int,
    n_boot: int,
    seed: int,
    alpha: float,
    scenes_per_group: int,
) -> dict[str, Any]:
    n_returned = 0
    n_covered = 0
    n_excludes = 0
    n_row_returned = 0
    n_row_covered = 0
    n_row_excludes = 0
    n_near_flat = 0
    widths: list[float] = []
    row_widths: list[float] = []
    n_rows = n_groups * scenes_per_group * N_DIRECTIONS
    for replication in range(n_replications):
        design, outcome, clusters = _draw(data_rng, n_groups=n_groups, scenes_per_group=scenes_per_group)
        fit_seed = seed + 1 + replication
        cluster_fit = fit_design(design, outcome, clusters, n_boot=n_boot, seed=fit_seed, alpha=alpha)
        if cluster_fit["status"] == "interval_returned":
            n_returned += 1
            interval = cluster_fit["interval"]
            widths.append(interval["ci_high"] - interval["ci_low"])
            if interval["ci_low"] <= BETA_A <= interval["ci_high"]:
                n_covered += 1
            if lower_endpoint_excludes_zero(interval["ci_low"]):
                n_excludes += 1
            if cluster_fit["near_flat_outcome"]:
                n_near_flat += 1
        row_fit = fit_design(
            design,
            outcome,
            np.arange(n_rows),
            n_boot=n_boot,
            seed=fit_seed,
            alpha=alpha,
        )
        if row_fit["status"] == "interval_returned":
            n_row_returned += 1
            interval = row_fit["interval"]
            row_widths.append(interval["ci_high"] - interval["ci_low"])
            if interval["ci_low"] <= BETA_A <= interval["ci_high"]:
                n_row_covered += 1
            if lower_endpoint_excludes_zero(interval["ci_low"]):
                n_row_excludes += 1
    return {
        "n_groups": n_groups,
        "n_scenes": n_groups * scenes_per_group,
        "n_rows": n_rows,
        "n_replications": n_replications,
        "n_interval_returned": n_returned,
        "n_covered_known_coefficient": n_covered,
        "n_lower_endpoint_above_zero": n_excludes,
        "n_near_flat_outcome": n_near_flat,
        "fraction_interval_returned": n_returned / n_replications,
        "fraction_covered_among_returned": None if n_returned == 0 else n_covered / n_returned,
        "fraction_excludes_zero_among_returned": None if n_returned == 0 else n_excludes / n_returned,
        "median_width": None if not widths else float(np.median(np.asarray(widths))),
        "n_row_interval_returned": n_row_returned,
        "fraction_row_covered_among_returned": None if n_row_returned == 0 else n_row_covered / n_row_returned,
        "fraction_row_excludes_zero_among_returned": (
            None if n_row_returned == 0 else n_row_excludes / n_row_returned
        ),
        "median_width_row": None if not row_widths else float(np.median(np.asarray(row_widths))),
    }


def _draw(
    data_rng: np.random.Generator,
    *,
    n_groups: int,
    scenes_per_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_per_group = scenes_per_group * N_DIRECTIONS
    n_rows = n_groups * rows_per_group
    clusters = np.repeat(np.arange(n_groups), rows_per_group)
    alignment = data_rng.uniform(ALIGN_LOW, ALIGN_HIGH, size=n_rows)
    latent = data_rng.random(n_groups)
    probability = BETA_0 + BETA_A * alignment
    outcome = (latent[clusters] < probability).astype(np.float64)
    design = np.column_stack([np.ones(n_rows), alignment])
    return design, outcome, clusters


def _wild_replicates(
    design: np.ndarray,
    outcome: np.ndarray,
    clusters: np.ndarray,
    point: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    xtx = design.T @ design
    xtx_inv = np.linalg.inv(xtx)
    residuals = outcome - design @ point
    unique, inverse = np.unique(clusters, return_inverse=True)
    n_groups = int(unique.shape[0])
    scores = np.zeros((n_groups, N_PARAMETERS), dtype=np.float64)
    for index in range(n_groups):
        mask = inverse == index
        scores[index] = design[mask].T @ residuals[mask]
    draw = np.random.default_rng(seed).integers(0, WEBB_WEIGHTS.shape[0], size=(n_boot, n_groups))
    weights = WEBB_WEIGHTS[draw]
    delta = weights @ scores
    replicates = point + delta @ xtx_inv.T
    return replicates[:, 1]


def _ols(design: np.ndarray, outcome: np.ndarray) -> np.ndarray | None:
    xtx = design.T @ design
    condition = float(np.linalg.cond(xtx))
    if not math.isfinite(condition) or condition > 1e12:
        return None
    try:
        return np.linalg.solve(xtx, design.T @ outcome)
    except np.linalg.LinAlgError:
        return None


def _refuse_real_row(row: dict[str, Any]) -> None:
    if row.get("data_origin") != "fabricated":
        raise RealDataRefused("data_origin must be fabricated")
    scene_id = str(row.get("scene_id", ""))
    group_id = str(row.get("counterfactual_group_id", ""))
    if scene_id.startswith(("dev", "cf")) or group_id.startswith(("dev", "cf")):
        raise RealDataRefused(f"refusing development id {scene_id!r} / {group_id!r}")
    if "row-" in scene_id or "row-" in group_id:
        raise RealDataRefused(f"refusing development-like id {scene_id!r} / {group_id!r}")


def _reading(by_n: Sequence[dict[str, Any]]) -> str:
    parts = []
    for row in by_n:
        share = row["fraction_excludes_zero_among_returned"]
        share_text = "undefined" if share is None else f"{share:.3f}"
        parts.append(
            f"{row['n_groups']} groups: returned {row['n_interval_returned']}/{row['n_replications']}, "
            f"lower endpoint above zero on {share_text} of returned intervals"
        )
    body = "; ".join(parts)
    return (
        "The known alignment coefficient is 0.25. An interval establishes the "
        "directional reading only when its lower endpoint is strictly above zero. "
        "Covering 0.25 is a different event. Twenty-four groups is not automatically enough. "
        + body
        + ". These fractions describe the fabricated design. They are not a measurement of a vision-language model."
    )


def _check_args(
    n_groups_grid: Sequence[int],
    n_replications: int,
    n_boot: int,
    seed: int,
    alpha: float,
    scenes_per_group: int,
) -> None:
    if not n_groups_grid:
        raise ValueError("n_groups_grid is empty")
    for n_groups in n_groups_grid:
        if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
            raise ValueError("each group count must be a positive integer")
    for name, value in (("n_replications", n_replications), ("n_boot", n_boot), ("scenes_per_group", scenes_per_group)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
