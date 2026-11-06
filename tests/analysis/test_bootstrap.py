"""Scene resampling keeps paired contexts together."""

from __future__ import annotations

import pytest

from analysis.bootstrap import paired_mean_difference, scene_bootstrap_mean, scene_bootstrap_rate


def test_constant_sample_has_a_degenerate_interval():
    stat = scene_bootstrap_mean({"a": 3, "b": 3}, n_boot=30, seed=2)
    assert stat["estimate"] == 3
    assert stat["ci_low"] == 3
    assert stat["ci_high"] == 3
    assert stat["unit"] == "scene_id"
    assert stat["method"] == "scene_percentile_bootstrap"
    assert "replicates" not in stat


def test_empty_sample_is_undefined():
    stat = scene_bootstrap_mean({}, n_boot=10, seed=1)
    assert stat["estimate"] is None
    assert stat["n_scenes"] == 0
    assert stat["undefined_reason"] == "no scenes"


def test_paired_difference_stays_inside_the_scene():
    left = {f"s{index}": float(index) for index in range(10)}
    right = dict(left)
    stat = paired_mean_difference(left, right, n_boot=100, seed=1)
    assert stat["estimate"] == 0
    assert stat["ci_low"] == 0
    assert stat["ci_high"] == 0
    assert stat["n_only_left"] == 0
    assert stat["n_scenes"] == 10


def test_scenes_missing_one_side_are_counted_and_excluded():
    stat = paired_mean_difference(
        {"a": 1.0, "b": 2.0},
        {"b": 0.0, "c": 5.0},
        n_boot=20,
        seed=1,
    )
    assert stat["estimate"] == 2
    assert stat["n_scenes"] == 1
    assert stat["n_only_left"] == 1
    assert stat["n_only_right"] == 1


def test_cluster_resample_keeps_sibling_scenes_together():
    values = {"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0}
    clusters = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    scene = scene_bootstrap_mean(values, n_boot=300, seed=1, return_replicates=True)
    grouped = scene_bootstrap_mean(
        values,
        n_boot=300,
        seed=1,
        cluster_of=clusters,
        return_replicates=True,
    )
    assert scene["estimate"] == 0.5
    assert grouped["estimate"] == 0.5
    assert grouped["unit"] == "cluster"
    assert any(value not in (0.0, 0.5, 1.0) for value in scene["replicates"])
    assert all(value in (0.0, 0.5, 1.0) for value in grouped["replicates"])


def test_rate_interval_is_undefined_when_the_denominator_is_zero():
    stat = scene_bootstrap_rate({"a": (0, 0), "b": (0, 0)}, n_boot=10, seed=1)
    assert stat["estimate"] is None
    assert stat["n_scenes"] == 2
    certain = scene_bootstrap_rate({"a": (1, 1), "b": (1, 1)}, n_boot=20, seed=1)
    assert certain["estimate"] == 1
    assert certain["ci_low"] == 1
    assert certain["ci_high"] == 1


def test_bootstrap_rejects_bad_arguments():
    with pytest.raises(ValueError):
        scene_bootstrap_mean({"a": 1}, n_boot=0, seed=1)
    with pytest.raises(ValueError):
        scene_bootstrap_mean({"a": True}, n_boot=5, seed=1)
    with pytest.raises(ValueError):
        scene_bootstrap_mean({"a": 1}, n_boot=5, seed=1, alpha=1)
    with pytest.raises(ValueError):
        scene_bootstrap_rate({"a": (1, 0), "b": (0, 1)}, n_boot=5, seed=1)
