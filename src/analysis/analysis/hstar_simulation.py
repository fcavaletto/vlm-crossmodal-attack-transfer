"""Precision of the H* cluster interval under a fabricated dependent design.

The draws are not model outputs. One uniform number is drawn for each
counterfactual group and compared with every row's probability, so the two
ordered directions and the sibling scenes in that group succeed or fail
together more often than independent rows would. Coverage and interval width
describe this design and this interval. They do not promise that an interval
on a vision-language model will exclude zero, and they are not a significance
calculation.

The data stream is one ``numpy.random.default_rng(seed)``. Grid points are
visited in the order given, and each point uses ``n_replications`` draws.
Replication ``r`` fits its interval with seed ``seed + 1 + r``. That
bootstrap seed repeats at every grid point. The row-level interval is the
same wild cluster bootstrap with each row treated as its own cluster.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .hstar import (
    CITED_CHECKPOINT1_ELIGIBLE_GROUPS,
    fit_linear_design,
    minimum_groups,
)

READING = (
    "Coverage and width describe this fabricated design and this cluster "
    "interval. They are not a measurement of a vision-language model. "
    "A group count that meets the planning bar does not promise that an "
    "interval will exclude zero."
)

# Pre-specified design. Probabilities stay inside (0.335, 0.665):
# 0.50 ± 0.25 * 0.50 ± 0.04.
BETA_0 = 0.50
BETA_A = 0.25
TASK_EFFECTS = {"attribute": 0.0, "counting": -0.04, "spatial": 0.04}
ALIGN_LOW = -0.50
ALIGN_HIGH = 0.50
SCENES_PER_GROUP = 2
FAMILY_CYCLE = ("attribute", "counting", "spatial")
PRIMARY_DESIGN_PARAMETERS = 4
PLANNING_COVERAGE_BAR = 0.90
DEFAULT_GROUP_GRID = (6, 12, 24, 48, 96)


def simulate_hstar_precision(
    *,
    n_groups_grid: Sequence[int] = DEFAULT_GROUP_GRID,
    n_replications: int = 200,
    n_boot: int = 399,
    seed: int = 20261107,
    alpha: float = 0.05,
    scenes_per_group: int = SCENES_PER_GROUP,
) -> dict[str, Any]:
    """Replicate the cluster interval for β_A at several group counts."""
    _validate_simulation_args(
        n_groups_grid=n_groups_grid,
        n_replications=n_replications,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        scenes_per_group=scenes_per_group,
    )
    data_rng = np.random.default_rng(seed)
    by_n = []
    for n_groups in n_groups_grid:
        if int(n_groups) % len(FAMILY_CYCLE) != 0:
            raise ValueError(
                "each group count must be a multiple of 3 so the simulated "
                "design keeps all three task indicators"
            )
        cluster_widths = []
        row_widths = []
        n_cluster_returned = 0
        n_row_returned = 0
        n_cluster_covered = 0
        n_row_covered = 0
        cluster_statuses: Counter[str] = Counter()
        for replication in range(n_replications):
            x_matrix, outcome, clusters = _draw_replication(
                data_rng,
                n_groups=int(n_groups),
                scenes_per_group=scenes_per_group,
            )
            fit_seed = seed + 1 + replication
            cluster_fit = fit_linear_design(
                x_matrix,
                outcome,
                clusters,
                n_boot=n_boot,
                seed=fit_seed,
                alpha=alpha,
            )
            row_fit = fit_linear_design(
                x_matrix,
                outcome,
                np.arange(outcome.shape[0]),
                n_boot=n_boot,
                seed=fit_seed,
                alpha=alpha,
            )
            cluster_statuses[_status_label(cluster_fit)] += 1
            if _returned(cluster_fit):
                n_cluster_returned += 1
                low = cluster_fit["intervals"][1]["ci_low"]
                high = cluster_fit["intervals"][1]["ci_high"]
                cluster_widths.append(float(high - low))
                if low <= BETA_A <= high:
                    n_cluster_covered += 1
            if _returned(row_fit):
                n_row_returned += 1
                low = row_fit["intervals"][1]["ci_low"]
                high = row_fit["intervals"][1]["ci_high"]
                row_widths.append(float(high - low))
                if low <= BETA_A <= high:
                    n_row_covered += 1
        by_n.append(
            _grid_row(
                n_groups=int(n_groups),
                n_replications=n_replications,
                scenes_per_group=scenes_per_group,
                n_cluster_returned=n_cluster_returned,
                n_cluster_covered=n_cluster_covered,
                cluster_widths=cluster_widths,
                n_row_returned=n_row_returned,
                n_row_covered=n_row_covered,
                row_widths=row_widths,
                cluster_statuses=cluster_statuses,
            )
        )
    payload = {
        "data_origin": "fabricated_dgp",
        "not_an_empirical_finding": True,
        "empirical_claim": False,
        "reading": READING,
        "estimand": "linear probability coefficient on alignment, clustered by counterfactual group",
        "known_beta_a": BETA_A,
        "dgp": {
            "name": "shared_uniform_linear_probability",
            "beta_0": BETA_0,
            "beta_a": BETA_A,
            "task_effects": dict(TASK_EFFECTS),
            "alignment_low": ALIGN_LOW,
            "alignment_high": ALIGN_HIGH,
            "scenes_per_group": int(scenes_per_group),
            "directions_per_scene": 2,
            "dependence": (
                "One Uniform(0, 1) draw per counterfactual group is compared "
                "with each row probability. Sibling scenes and both ordered "
                "directions share that draw."
            ),
            "probability_bound_note": (
                "With these constants every row probability lies in "
                "[0.335, 0.665] before the shared uniform is applied."
            ),
        },
        "procedure": {
            "name": "wild_cluster_bootstrap",
            "weights": "webb6",
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "quantile_method": "linear",
            "data_seed": int(seed),
            "bootstrap_seed_rule": "data_seed + 1 + replication_index",
            "row_level_contrast": "same bootstrap with one cluster per row",
            "primary_design_parameters": PRIMARY_DESIGN_PARAMETERS,
            "algebraic_minimum_groups": minimum_groups(PRIMARY_DESIGN_PARAMETERS),
        },
        "by_n_groups": by_n,
    }
    payload["recommendation"] = recommend_eligible_groups(payload)
    return payload


def recommend_eligible_groups(simulation: Mapping[str, Any]) -> dict[str, Any]:
    """Planning rule for eligible groups. It does not promise significance.

    The algebraic minimum is ``k + 1`` groups. For the three-family primary
    design that is 5. The planning bar, applied only to this fabricated grid,
    asks for a returned interval in every replication and coverage of the
    known coefficient of at least 0.90. The recommended count is the smallest
    grid point that meets the bar. If none does, the recommendation is that
    this grid does not supply a count. Neither outcome says the interval will
    exclude zero.
    """
    rows = list(simulation["by_n_groups"])
    meeting = [
        row
        for row in rows
        if row["fraction_interval_returned"] == 1.0
        and row["coverage_cluster_among_returned"] is not None
        and row["coverage_cluster_among_returned"] >= PLANNING_COVERAGE_BAR
    ]
    algebraic = minimum_groups(PRIMARY_DESIGN_PARAMETERS)
    if meeting:
        chosen = min(meeting, key=lambda row: int(row["n_groups"]))
        recommended = int(chosen["n_groups"])
        status = "smallest_grid_point_meeting_planning_bar"
        sentence = (
            f"Under this fabricated design, the smallest grid point at which "
            f"every replication returned a cluster interval and at least "
            f"{PLANNING_COVERAGE_BAR:.0%} of those intervals covered the known "
            f"coefficient is {recommended} eligible groups. That count does not "
            f"promise that an interval will exclude zero. The algebraic minimum "
            f"for returning any interval on the three-family design is "
            f"{algebraic} groups. A cited Checkpoint 1 inclusion count of "
            f"{CITED_CHECKPOINT1_ELIGIBLE_GROUPS} groups is below that minimum. "
            f"This is not a measurement of a vision-language model."
        )
    else:
        recommended = None
        status = "no_grid_point_met_the_planning_bar"
        sentence = (
            "No point on this fabricated grid met the planning bar of a "
            "returned interval in every replication and "
            f"{PLANNING_COVERAGE_BAR:.0%} coverage of the known coefficient. "
            "That result does not promise that a larger sample will exclude "
            "zero. The algebraic minimum for returning any interval on the "
            f"three-family design is {algebraic} groups. A cited Checkpoint 1 "
            f"inclusion count of {CITED_CHECKPOINT1_ELIGIBLE_GROUPS} groups is "
            "below that minimum. This is not a measurement of a vision-language "
            "model."
        )
    return {
        "algebraic_rule": "Return a cluster interval only when G >= k + 1.",
        "primary_design_parameters": PRIMARY_DESIGN_PARAMETERS,
        "algebraic_minimum_groups": algebraic,
        "planning_coverage_bar": PLANNING_COVERAGE_BAR,
        "planning_bar_note": (
            "The bar is a description of coverage under the fabricated design. "
            "It is not a promise that an interval will exclude zero and it is "
            "not a significance calculation."
        ),
        "recommended_minimum_eligible_groups": recommended,
        "status": status,
        "does_not_promise_significance": True,
        "cited_checkpoint1_eligible_groups": CITED_CHECKPOINT1_ELIGIBLE_GROUPS,
        "cited_checkpoint1_below_algebraic_minimum": CITED_CHECKPOINT1_ELIGIBLE_GROUPS
        < algebraic,
        "sentence": sentence,
    }


def _draw_replication(
    rng: np.random.Generator,
    *,
    n_groups: int,
    scenes_per_group: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = n_groups * scenes_per_group * 2
    x_matrix = np.zeros((n_rows, PRIMARY_DESIGN_PARAMETERS), dtype=np.float64)
    outcome = np.zeros(n_rows, dtype=np.float64)
    clusters = np.empty(n_rows, dtype=np.int64)
    cursor = 0
    for group in range(n_groups):
        family = FAMILY_CYCLE[group % len(FAMILY_CYCLE)]
        shared = float(rng.random())
        for _scene in range(scenes_per_group):
            for _direction in range(2):
                alignment = float(rng.uniform(ALIGN_LOW, ALIGN_HIGH))
                probability = BETA_0 + BETA_A * alignment + TASK_EFFECTS[family]
                if not 0.0 < probability < 1.0:
                    raise RuntimeError("fabricated probability left (0, 1)")
                x_matrix[cursor, 0] = 1.0
                x_matrix[cursor, 1] = alignment
                x_matrix[cursor, 2] = 1.0 if family == "counting" else 0.0
                x_matrix[cursor, 3] = 1.0 if family == "spatial" else 0.0
                outcome[cursor] = 1.0 if shared < probability else 0.0
                clusters[cursor] = group
                cursor += 1
    return x_matrix, outcome, clusters


def _returned(fitted: Mapping[str, Any]) -> bool:
    return fitted["status"] == "interval_returned" and bool(fitted["identifiable"])


def _grid_row(
    *,
    n_groups: int,
    n_replications: int,
    scenes_per_group: int,
    n_cluster_returned: int,
    n_cluster_covered: int,
    cluster_widths: list[float],
    n_row_returned: int,
    n_row_covered: int,
    row_widths: list[float],
    cluster_statuses: Counter[str],
) -> dict[str, Any]:
    return {
        "n_groups": n_groups,
        "n_replications": n_replications,
        "n_rows": n_groups * scenes_per_group * 2,
        "fraction_interval_returned": n_cluster_returned / n_replications,
        "n_interval_returned": n_cluster_returned,
        "coverage_cluster_among_returned": _ratio(n_cluster_covered, n_cluster_returned),
        "coverage_cluster_counting_unreturned_as_not_covered": n_cluster_covered / n_replications,
        "median_ci_width_cluster": _median(cluster_widths),
        "fraction_row_interval_returned": n_row_returned / n_replications,
        "coverage_row_among_returned": _ratio(n_row_covered, n_row_returned),
        "median_ci_width_row": _median(row_widths),
        "cluster_status_counts": dict(sorted(cluster_statuses.items())),
    }


def _status_label(fitted: Mapping[str, Any]) -> str:
    if fitted["status"] == "interval_returned" and not fitted["identifiable"]:
        return "near_flat_outcome"
    return str(fitted["status"])


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _validate_simulation_args(
    *,
    n_groups_grid: Sequence[int],
    n_replications: int,
    n_boot: int,
    seed: int,
    alpha: float,
    scenes_per_group: int,
) -> None:
    if len(tuple(n_groups_grid)) == 0:
        raise ValueError("n_groups_grid must not be empty")
    for n_groups in n_groups_grid:
        if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
            raise ValueError("each group count must be an integer >= 1")
    if isinstance(n_replications, bool) or not isinstance(n_replications, int) or n_replications < 1:
        raise ValueError("n_replications must be an integer >= 1")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(scenes_per_group, bool) or not isinstance(scenes_per_group, int) or scenes_per_group < 1:
        raise ValueError("scenes_per_group must be an integer >= 1")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, nargs="+", default=list(DEFAULT_GROUP_GRID))
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--boot", type=int, default=399)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    payload = simulate_hstar_precision(
        n_groups_grid=args.groups,
        n_replications=args.replications,
        n_boot=args.boot,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
