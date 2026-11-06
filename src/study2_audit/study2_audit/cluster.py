"""Percentile intervals that resample counterfactual groups.

A draw takes whole groups. Every scene in a drawn group enters the mean, so
the estimand stays the scene mean and sibling scenes stay together. This is
a description of a table that already exists. It is not a fit of H*.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

METHOD = "percentile_bootstrap"
QUANTILE_METHOD = "linear"
UNIT = "counterfactual_group"


def descriptive_cluster_mean(
    values: Mapping[str, float],
    cluster_of: Mapping[str, str],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | str | None]:
    """Equal-tailed interval for the mean of one number per scene.

    ``cluster_of`` maps each scene id to its counterfactual group. The point
    estimate is the mean over scenes. The bootstrap redraws groups.
    """

    _check_boot(n_boot, seed, alpha)
    scene_ids = sorted(values)
    if not scene_ids:
        raise ValueError("no scenes")
    if set(scene_ids) != set(cluster_of):
        raise ValueError("cluster_of must name the same scenes as values")
    observed = np.asarray([float(values[scene_id]) for scene_id in scene_ids], dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("values must be finite")
    groups = _groups(scene_ids, observed, cluster_of)
    point = float(observed.mean())
    replicates = _replicates(groups, n_boot=n_boot, seed=seed)
    low, high = _quantile(replicates, alpha)
    return _payload(
        estimate=point,
        ci_low=low,
        ci_high=high,
        n_scenes=len(scene_ids),
        n_groups=len(groups),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        unit=UNIT,
    )


def descriptive_group_rate(
    group_values: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | str | None]:
    """Interval for a mean that already has one number per group."""

    _check_boot(n_boot, seed, alpha)
    group_ids = sorted(group_values)
    if not group_ids:
        raise ValueError("no groups")
    observed = np.asarray([float(group_values[group_id]) for group_id in group_ids], dtype=np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("values must be finite")
    point = float(observed.mean())
    draws = np.random.default_rng(seed).integers(0, len(group_ids), size=(n_boot, len(group_ids)))
    replicates = observed[draws].mean(axis=1)
    low, high = _quantile(replicates, alpha)
    return _payload(
        estimate=point,
        ci_low=low,
        ci_high=high,
        n_scenes=len(group_ids),
        n_groups=len(group_ids),
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        unit=UNIT,
    )


def scene_iid_mean(
    values: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float | int | str | None]:
    """The same mean, redrawing scenes as if they were independent.

    This interval is a sensitivity. The study's sampling unit is the group.
    """

    cluster_of = {scene_id: scene_id for scene_id in values}
    result = descriptive_cluster_mean(values, cluster_of, n_boot=n_boot, seed=seed, alpha=alpha)
    result["unit"] = "scene_treated_as_independent"
    result["h_star_fitted"] = False
    return result


def _groups(
    scene_ids: Sequence[str],
    observed: np.ndarray,
    cluster_of: Mapping[str, str],
) -> list[np.ndarray]:
    order = sorted(set(cluster_of[scene_id] for scene_id in scene_ids))
    grouped = {group_id: [] for group_id in order}
    for scene_id, value in zip(scene_ids, observed):
        grouped[cluster_of[scene_id]].append(float(value))
    return [np.asarray(grouped[group_id], dtype=np.float64) for group_id in order]


def _replicates(groups: Sequence[np.ndarray], *, n_boot: int, seed: int) -> np.ndarray:
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(n_boot, len(groups)))
    replicates = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        replicates[index] = np.concatenate([groups[int(choice)] for choice in draws[index]]).mean()
    return replicates


def _quantile(replicates: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method=QUANTILE_METHOD)
    return float(low), float(high)


def _payload(
    *,
    estimate: float,
    ci_low: float,
    ci_high: float,
    n_scenes: int,
    n_groups: int,
    n_boot: int,
    seed: int,
    alpha: float,
    unit: str,
) -> dict[str, float | int | str | None]:
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "width": ci_high - ci_low,
        "n_scenes": n_scenes,
        "n_groups": n_groups,
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "method": METHOD,
        "quantile_method": QUANTILE_METHOD,
        "unit": unit,
        "h_star_fitted": False,
        "empirical_claim_about_a_model": False,
    }


def _check_boot(n_boot: int, seed: int, alpha: float) -> None:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
