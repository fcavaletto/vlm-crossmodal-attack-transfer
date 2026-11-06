"""Sample-size sketch on a fabricated distribution.

This is a planning tool. The draws are not model outputs. Coverage and
interval width describe the scene-level percentile bootstrap under the
stated three-point distribution. They do not choose the study's sample
size and they are not a measurement of a vision-language model.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

import numpy as np

from .bootstrap import QUANTILE_METHOD, scene_bootstrap_mean

READING = (
    "Widths and coverage describe this fabricated distribution and this "
    "bootstrap. They are not a measurement of a vision-language model and "
    "they do not choose the study sample size."
)


def simulate_paired_difference_precision(
    *,
    n_scenes_grid: Sequence[int] = (12, 24, 48),
    n_replications: int = 100,
    n_boot: int = 400,
    seed: int = 20261106,
    p_plus: float = 0.25,
    p_minus: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """Replicate the paired-mean interval at several scene counts.

    Each scene contributes an i.i.d. difference D in {+1, -1, 0} with
    P(D = +1) = p_plus and P(D = -1) = p_minus. The true mean is
    p_plus - p_minus. Replication r draws its scenes from
    ``numpy.random.default_rng(seed)``. That stream continues across the grid
    in the order given. Replication r draws its bootstrap from seed
    ``seed + 1 + r`` (the bootstrap seed repeats for each grid point).
    """
    _validate_simulation_args(
        n_scenes_grid=n_scenes_grid,
        n_replications=n_replications,
        n_boot=n_boot,
        seed=seed,
        p_plus=p_plus,
        p_minus=p_minus,
        alpha=alpha,
    )
    probabilities = _categorical_probabilities(p_plus, p_minus)
    mean = float(p_plus - p_minus)
    data_rng = np.random.default_rng(seed)
    by_n = []
    for n_scenes in n_scenes_grid:
        widths = []
        n_covered = 0
        for replication in range(n_replications):
            draws = data_rng.choice(
                np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
                size=n_scenes,
                p=probabilities["used"],
            )
            values = {f"s{index:04d}": float(draws[index]) for index in range(n_scenes)}
            interval = scene_bootstrap_mean(
                values,
                n_boot=n_boot,
                seed=seed + 1 + replication,
                alpha=alpha,
            )
            widths.append(float(interval["ci_high"] - interval["ci_low"]))
            if interval["ci_low"] <= mean <= interval["ci_high"]:
                n_covered += 1
        width_array = np.asarray(widths, dtype=np.float64)
        by_n.append(
            {
                "n_scenes": int(n_scenes),
                "n_replications": int(n_replications),
                "median_ci_width": float(np.median(width_array)),
                "mean_ci_width": float(np.mean(width_array)),
                "n_covered": int(n_covered),
                "coverage": n_covered / n_replications,
            }
        )
    return {
        "data_origin": "fabricated_dgp",
        "not_an_empirical_finding": True,
        "reading": READING,
        "estimand": "mean of i.i.d. scene-level paired differences",
        "dgp": {
            "name": "three_point_paired_difference",
            "support": [-1, 0, 1],
            "p_plus": float(p_plus),
            "p_minus": float(p_minus),
            "p_zero": float(probabilities["p_zero"]),
            "mean": mean,
        },
        "procedure": {
            "name": "scene_percentile_bootstrap",
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "quantile_method": QUANTILE_METHOD,
            "data_seed": int(seed),
            "bootstrap_seed_rule": "data_seed + 1 + replication_index",
        },
        "by_n": by_n,
    }


def _categorical_probabilities(p_plus: float, p_minus: float) -> dict:
    p_zero = 1.0 - p_plus - p_minus
    raw = np.asarray([p_minus, p_zero, p_plus], dtype=np.float64)
    total = float(raw.sum())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise ValueError("p_plus and p_minus must be non-negative and sum to at most 1")
    return {"p_zero": float(p_zero), "used": raw / total}


def _validate_simulation_args(
    *,
    n_scenes_grid: Sequence[int],
    n_replications: int,
    n_boot: int,
    seed: int,
    p_plus: float,
    p_minus: float,
    alpha: float,
) -> None:
    if len(tuple(n_scenes_grid)) == 0:
        raise ValueError("n_scenes_grid must not be empty")
    for n_scenes in n_scenes_grid:
        if isinstance(n_scenes, bool) or not isinstance(n_scenes, int) or n_scenes < 1:
            raise ValueError("each scene count must be an integer >= 1")
    if isinstance(n_replications, bool) or not isinstance(n_replications, int) or n_replications < 1:
        raise ValueError("n_replications must be an integer >= 1")
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    for name, value in (("p_plus", p_plus), ("p_minus", p_minus)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a probability")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be a non-negative finite probability")
    if float(p_plus) + float(p_minus) > 1 + 1e-12:
        raise ValueError("p_plus + p_minus must be <= 1")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be strictly between 0 and 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--boot", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--p-plus", type=float, default=0.25)
    parser.add_argument("--p-minus", type=float, default=0.05)
    args = parser.parse_args()
    payload = simulate_paired_difference_precision(
        n_scenes_grid=args.n,
        n_replications=args.replications,
        n_boot=args.boot,
        seed=args.seed,
        p_plus=args.p_plus,
        p_minus=args.p_minus,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
