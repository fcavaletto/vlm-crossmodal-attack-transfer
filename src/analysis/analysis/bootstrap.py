"""Scene-level percentile intervals.

The resampled unit is the scene. Context rows that share a scene are not
drawn independently. If the caller passes ``cluster_of``, the unit becomes
that cluster (for example a future counterfactual group). A drawn cluster
contributes each of its scenes, so a larger cluster weighs more than a
smaller one. Drawing a cluster twice counts its scenes twice. The interval
is not a mean of cluster-level means.

The interval is the equal-tailed percentile interval of the bootstrap
replicates, using NumPy's linear quantile. It is not a normal approximation
and it is not a p-value.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

METHOD_NAME = "scene_percentile_bootstrap"
QUANTILE_METHOD = "linear"


def scene_bootstrap_mean(
    values: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    cluster_of: Mapping[str, str] | None = None,
    return_replicates: bool = False,
) -> dict:
    """Percentile interval for the mean of one number per scene."""
    _validate_boot_args(n_boot, seed, alpha)
    scene_ids = sorted(values)
    observed = [_as_float(values[scene_id], scene_id) for scene_id in scene_ids]
    if not scene_ids:
        result = _empty_interval(
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            reason="no scenes",
            unit=_unit_name(cluster_of),
        )
        return result
    point = float(np.mean(np.asarray(observed, dtype=np.float64)))
    replicates = _resample_means(
        scene_ids,
        observed,
        n_boot=n_boot,
        seed=seed,
        cluster_of=cluster_of,
    )
    low, high = _percentile_interval(replicates, alpha)
    result = _interval_dict(
        n_scenes=len(scene_ids),
        estimate=point,
        ci_low=low,
        ci_high=high,
        undefined_reason=None,
        n_boot=n_boot,
        n_undefined_replicates=0,
        seed=seed,
        alpha=alpha,
        unit=_unit_name(cluster_of),
    )
    if return_replicates:
        result["replicates"] = [float(value) for value in replicates]
    return result


def scene_bootstrap_rate(
    indicators: Mapping[str, tuple[int, int]],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    cluster_of: Mapping[str, str] | None = None,
    return_replicates: bool = False,
) -> dict:
    """Percentile interval for a scene-level ratio.

    Each scene contributes ``(denominator, numerator)`` with the numerator
    equal to 0 or 1 and no larger than the denominator. Scenes with
    denominator 0 stay in the resample. A replicate whose denominator is 0
    is undefined and is left out of the percentile interval. The point
    estimate is undefined when the full sample has denominator 0, and then
    no interval is reported.
    """
    _validate_boot_args(n_boot, seed, alpha)
    scene_ids = sorted(indicators)
    denominators = []
    numerators = []
    for scene_id in scene_ids:
        denominator, numerator = indicators[scene_id]
        if isinstance(denominator, bool) or isinstance(numerator, bool):
            raise ValueError("rate indicators must be integers")
        if denominator not in (0, 1) or numerator not in (0, 1):
            raise ValueError("rate indicators must be 0 or 1")
        if numerator > denominator:
            raise ValueError("numerator cannot exceed denominator")
        denominators.append(int(denominator))
        numerators.append(int(numerator))
    if not scene_ids:
        return _empty_interval(
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            reason="no scenes",
            unit=_unit_name(cluster_of),
        )
    point_den = sum(denominators)
    point_num = sum(numerators)
    if point_den == 0:
        result = _empty_interval(
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            reason="no scene was in the denominator",
            unit=_unit_name(cluster_of),
        )
        result["n_scenes"] = len(scene_ids)
        return result
    point = point_num / point_den
    replicates, n_undefined = _resample_rates(
        scene_ids,
        denominators,
        numerators,
        n_boot=n_boot,
        seed=seed,
        cluster_of=cluster_of,
    )
    if len(replicates) == 0:
        result = _empty_interval(
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            reason="every bootstrap replicate had a zero denominator",
            unit=_unit_name(cluster_of),
        )
        result["n_scenes"] = len(scene_ids)
        result["n_undefined_replicates"] = n_undefined
        return result
    low, high = _percentile_interval(np.asarray(replicates, dtype=np.float64), alpha)
    result = _interval_dict(
        n_scenes=len(scene_ids),
        estimate=point,
        ci_low=low,
        ci_high=high,
        undefined_reason=None,
        n_boot=n_boot,
        n_undefined_replicates=n_undefined,
        seed=seed,
        alpha=alpha,
        unit=_unit_name(cluster_of),
    )
    if return_replicates:
        result["replicates"] = [float(value) for value in replicates]
    return result


def paired_mean_difference(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
    cluster_of: Mapping[str, str] | None = None,
    return_replicates: bool = False,
) -> dict:
    """Mean of scene-wise ``left - right``, with a scene bootstrap interval.

    Scenes present on only one side are excluded from the mean and counted.
    The subtraction happens before the resample, so a scene's two numbers
    stay paired.
    """
    shared = sorted(set(left) & set(right))
    differences = {
        scene_id: _as_float(left[scene_id], scene_id) - _as_float(right[scene_id], scene_id)
        for scene_id in shared
    }
    cluster = None
    if cluster_of is not None:
        missing = [scene_id for scene_id in shared if scene_id not in cluster_of]
        if missing:
            raise ValueError(f"cluster_of is missing scenes {missing}")
        cluster = {scene_id: cluster_of[scene_id] for scene_id in shared}
    stat = scene_bootstrap_mean(
        differences,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
        cluster_of=cluster,
        return_replicates=return_replicates,
    )
    stat["n_only_left"] = len(set(left) - set(right))
    stat["n_only_right"] = len(set(right) - set(left))
    stat["contrast"] = "mean(left - right) on scenes present in both maps"
    return stat


def _resample_means(
    scene_ids: Sequence[str],
    observed: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    cluster_of: Mapping[str, str] | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(observed, dtype=np.float64)
    if cluster_of is None:
        draws = rng.integers(0, len(scene_ids), size=(n_boot, len(scene_ids)))
        return values[draws].mean(axis=1)
    _require_cluster_keys(scene_ids, cluster_of)
    clusters, members = _cluster_members(scene_ids, cluster_of)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    means = np.empty(n_boot, dtype=np.float64)
    bundled = [values[np.asarray(members[cluster], dtype=int)] for cluster in clusters]
    for replicate in range(n_boot):
        parts = [bundled[index] for index in draws[replicate]]
        means[replicate] = np.concatenate(parts).mean()
    return means


def _resample_rates(
    scene_ids: Sequence[str],
    denominators: Sequence[int],
    numerators: Sequence[int],
    *,
    n_boot: int,
    seed: int,
    cluster_of: Mapping[str, str] | None,
) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    den = np.asarray(denominators, dtype=np.int64)
    num = np.asarray(numerators, dtype=np.int64)
    if cluster_of is None:
        draws = rng.integers(0, len(scene_ids), size=(n_boot, len(scene_ids)))
        boot_den = den[draws].sum(axis=1)
        boot_num = num[draws].sum(axis=1)
        defined = boot_den > 0
        values = [
            float(boot_num[index] / boot_den[index])
            for index in range(n_boot)
            if defined[index]
        ]
        return values, int((~defined).sum())
    _require_cluster_keys(scene_ids, cluster_of)
    clusters, members = _cluster_members(scene_ids, cluster_of)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    bundled_den = [den[np.asarray(members[cluster], dtype=int)] for cluster in clusters]
    bundled_num = [num[np.asarray(members[cluster], dtype=int)] for cluster in clusters]
    values = []
    n_undefined = 0
    for replicate in range(n_boot):
        total_den = 0
        total_num = 0
        for index in draws[replicate]:
            total_den += int(bundled_den[index].sum())
            total_num += int(bundled_num[index].sum())
        if total_den == 0:
            n_undefined += 1
        else:
            values.append(total_num / total_den)
    return values, n_undefined


def _cluster_members(
    scene_ids: Sequence[str], cluster_of: Mapping[str, str]
) -> tuple[list[str], dict[str, list[int]]]:
    clusters = sorted({cluster_of[scene_id] for scene_id in scene_ids})
    index_of = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    members = {
        cluster: [index_of[scene_id] for scene_id in scene_ids if cluster_of[scene_id] == cluster]
        for cluster in clusters
    }
    return clusters, members


def _require_cluster_keys(scene_ids: Sequence[str], cluster_of: Mapping[str, str]) -> None:
    if set(cluster_of) != set(scene_ids):
        raise ValueError("cluster_of keys must match the scene ids being resampled")
    for scene_id, cluster in cluster_of.items():
        if not isinstance(cluster, str) or cluster == "":
            raise ValueError(f"cluster id for {scene_id} must be a non-empty string")


def _percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2], method=QUANTILE_METHOD)
    return float(low), float(high)


def _interval_dict(
    *,
    n_scenes: int,
    estimate: float | None,
    ci_low: float | None,
    ci_high: float | None,
    undefined_reason: str | None,
    n_boot: int,
    n_undefined_replicates: int,
    seed: int,
    alpha: float,
    unit: str,
) -> dict:
    return {
        "n_scenes": n_scenes,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "undefined_reason": undefined_reason,
        "n_boot": n_boot,
        "n_undefined_replicates": n_undefined_replicates,
        "seed": seed,
        "alpha": alpha,
        "method": METHOD_NAME,
        "quantile_method": QUANTILE_METHOD,
        "unit": unit,
    }


def _empty_interval(
    *, n_boot: int, seed: int, alpha: float, reason: str, unit: str
) -> dict:
    return _interval_dict(
        n_scenes=0,
        estimate=None,
        ci_low=None,
        ci_high=None,
        undefined_reason=reason,
        n_boot=n_boot,
        n_undefined_replicates=0,
        seed=seed,
        alpha=alpha,
        unit=unit,
    )


def _unit_name(cluster_of: Mapping[str, str] | None) -> str:
    if cluster_of is None:
        return "scene_id"
    return "cluster"


def _validate_boot_args(n_boot: int, seed: int, alpha: float) -> None:
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError("n_boot must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number strictly between 0 and 1")
    if not math.isfinite(float(alpha)) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must be a number strictly between 0 and 1")


def _as_float(value: float, scene_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"scene {scene_id} value must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"scene {scene_id} value must be finite")
    return number
