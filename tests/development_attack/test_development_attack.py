"""Checkpoint 2.1 rules with no model load."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from attack_calibration.protocol import FROZEN_SCENES, epsilon as calibration_epsilon
from attack_feasibility.steps import pgd_update, project_signed_step, scene_seed
from dev_pilot.fixtures import load_manifest, verify_checksums
from attack_calibration.prior import describe_scenes, load_prior

from development_attack.analyze import analyze_run
from development_attack.protocol import (
    HSTAR_UNESTIMATED,
    MIN_EVENTS,
    MIN_GROUPS,
    N_GROUPS,
    N_SCENES,
    SEED_PREFIX,
    STARTED_FROM_COMMIT,
    assert_protocol_committed,
    configurations,
    epsilon,
    step_size,
)
from development_attack.report import render_report
from development_attack.run import _ancestor_pids, _is_other_model_command
from development_attack.select import select_development
from development_attack.verify import (
    assert_memory_lines,
    assert_preserved,
    bound_errors,
    preserved_changes,
)

ROOT = Path(__file__).resolve().parents[2]
VERIFICATION = {"passed": True}


def _predictors(index: int, transfer: bool) -> tuple[float, float, float, float]:
    alignment = -0.35 + (index % 11) * 0.05 + (0.25 if transfer else 0.0)
    neutral = 1.0 + (index % 13) * 0.17
    helpful = 2.0 + (index % 9) * 0.13
    gradient_l1 = 15.0 + (index % 17) * 4.0
    return alignment, neutral, helpful, gradient_l1


def _view(color: str) -> dict[str, object]:
    return {"decision": color, "margin": 1.0, "nll": 0.1, "error": None}


def _run(specs: list[tuple[str, str]], bad_indexes: set[int] | None = None):
    bad_indexes = bad_indexes or set()
    scenes: dict[str, dict] = {}
    scalars = []
    scene_ids = []
    for index, (kind, group) in enumerate(specs):
        scene_id = f"s{index:03d}"
        scene_ids.append(scene_id)
        label = "red"
        other = "blue"
        if kind == "out":
            clean_neutral = other
            clean_helpful = label
            perturbed_neutral = label
            perturbed_helpful = label
        elif kind == "transfer":
            clean_neutral = label
            clean_helpful = label
            perturbed_neutral = other
            perturbed_helpful = other
        elif kind == "failure":
            clean_neutral = label
            clean_helpful = label
            perturbed_neutral = other
            perturbed_helpful = label
        elif kind == "excluded":
            clean_neutral = label
            clean_helpful = other
            perturbed_neutral = other
            perturbed_helpful = label
        else:
            raise AssertionError(kind)
        block = {
            "perturbed": {
                "neutral": _view(perturbed_neutral),
                "helpful_nonrevealing": _view(perturbed_helpful),
            },
            "linf_before_round_trip": epsilon(),
            "linf_after_round_trip": epsilon(),
            "transfer_status": kind,
        }
        alignment, neutral, helpful, gradient_l1 = _predictors(index, kind == "transfer")
        if index in bad_indexes:
            alignment = None
        scenes[scene_id] = {
            "scene_id": scene_id,
            "counterfactual_group_id": group,
            "answer": label,
            "finished": True,
            "held_out_opened": False,
            "recomputed_neutral_gradient_l1": gradient_l1,
            "clean": {
                "neutral": _view(clean_neutral),
                "helpful_nonrevealing": _view(clean_helpful),
            },
            "epsilons": {"1/255": {"pgd": {"3": block, "7": block}, "fgsm": block, "random": block}},
        }
        scalars.append(
            {
                "scene_id": scene_id,
                "counterfactual_group_id": group,
                "answer": label,
                "alignment_neutral_to_helpful": alignment,
                "neutral_margin": neutral,
                "helpful_margin": helpful,
                "neutral_gradient_l1": gradient_l1,
            }
        )
    progress = {
        "status": "completed",
        "n_finished": N_SCENES,
        "n_scenes": N_SCENES,
        "scene_ids": scene_ids,
        "scenes": scenes,
        "held_out_scenes_opened": False,
        "float32_loaded": False,
        "source_context": "neutral",
        "epsilon": "1/255",
        "pgd_steps": 7,
        "primary_steps": 3,
        "seed_prefix": SEED_PREFIX,
        "protocol_commit": STARTED_FROM_COMMIT,
        "git_head_at_start": "test",
        "h_star_fitted": False,
    }
    return progress, scalars


def _full(pattern: list[str], groups_for_pattern: int) -> list[tuple[str, str]]:
    specs = []
    cursor = 0
    for group_index in range(groups_for_pattern):
        for _arm in range(2):
            specs.append((pattern[cursor], f"g{group_index:02d}"))
            cursor += 1
    for group_index in range(groups_for_pattern, N_GROUPS):
        specs.append(("out", f"g{group_index:02d}"))
        specs.append(("out", f"g{group_index:02d}"))
    assert len(specs) == N_SCENES
    assert cursor == len(pattern)
    return specs


def _assert_no_coefficient(payload: dict) -> None:
    assert payload["h_star_estimated"] is False
    assert payload["h_star"] is None
    assert payload["confirmatory_supported"] is False
    assert payload["h_star_statement"] == HSTAR_UNESTIMATED
    dumped = json.dumps(payload)
    assert "beta_a" not in dumped
    assert "diagnostic_point_estimate" not in dumped


def test_manifest_selects_every_development_scene_and_the_calibration_seed():
    records = load_manifest(ROOT)
    selected = select_development(records)
    assert selected["ok"] is True
    assert selected["n_groups"] == N_GROUPS
    assert selected["n_scenes"] == N_SCENES
    assert len(selected["selected_group_ids"]) == N_GROUPS
    by_group: dict[str, list[str]] = {}
    for row in selected["scenes"]:
        by_group.setdefault(row["group_id"], []).append(row["scene_id"])
    assert all(len(arms) == 2 for arms in by_group.values())
    seeds = {row["scene_id"]: row["seed"] for row in selected["scenes"]}
    manifest_seeds = {row["scene_id"]: row["seed"] for row in records}
    for scene_id, _group, answer, seed in FROZEN_SCENES:
        assert seeds[scene_id] == seed
        assert seeds[scene_id] == scene_seed(SEED_PREFIX, scene_id)
        assert manifest_seeds[scene_id] != seed
        matched = next(row for row in selected["scenes"] if row["scene_id"] == scene_id)
        assert matched["answer"] == answer
    assert epsilon() == calibration_epsilon(1.0)
    assert step_size() == calibration_epsilon(1.0) / 4.0
    assert [row["method"] for row in configurations()] == ["pgd", "fgsm", "random", "pgd"]
    assert {row["epsilon"] for row in configurations()} == {"1/255"}
    assert [row["steps"] for row in configurations()] == [3, 1, 1, 7]
    assert verify_checksums(ROOT)["files"] > 90
    assert_memory_lines(ROOT)
    assert preserved_changes(ROOT) == []
    assert_preserved(ROOT)
    assert assert_protocol_committed(ROOT) == STARTED_FROM_COMMIT
    scalars = describe_scenes(load_prior(ROOT), [row["scene_id"] for row in selected["scenes"]])
    assert len(scalars) == N_SCENES
    assert all(
        row["alignment_neutral_to_helpful"] is not None
        and -1.0 <= row["alignment_neutral_to_helpful"] <= 1.0
        and row["neutral_margin"] is not None
        and row["helpful_margin"] is not None
        and row["neutral_gradient_l1"] is not None
        and row["neutral_gradient_l1"] > 0.0
        for row in scalars
    )


def test_package_keeps_the_frozen_attack():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "development_attack" / "development_attack").glob("*.py")
    )
    assert "2/255" not in text
    assert "fixtures/held" not in text
    assert "fixtures/sealed" not in text
    assert 'load_model_after_budget(decisions["bfloat16"], "bfloat16")' in text
    assert 'load_model_after_budget(decisions["float32"]' not in text
    assert MIN_EVENTS == 40
    assert MIN_GROUPS == 6


def test_projection_stays_inside_the_ball_and_the_round_trip_slack_is_half_a_level():
    image = torch.zeros(3, 2, 2)
    gradient = torch.ones(3, 2, 2)
    radius = epsilon()
    current = image.clone()
    for _step in range(7):
        current = pgd_update(current, image, gradient, radius / 4.0, radius)
    distance = float((current - image).abs().max())
    assert distance <= radius + 1e-5
    assert distance == radius or math.isclose(distance, radius, rel_tol=0.0, abs_tol=1e-5)
    fgsm = project_signed_step(image, gradient, radius, "positive")
    assert bound_errors(float((fgsm - image).abs().max()), radius, radius) == []
    assert bound_errors(radius + 1e-4, radius, radius)
    assert bound_errors(radius, radius + (0.5 / 255.0) + 1e-4, radius)
    assert bound_errors(radius, radius + (0.5 / 255.0), radius) == []


def test_three_transfer_failures_do_not_estimate_h_star(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("H* was fit")

    monkeypatch.setattr("development_attack.analyze.fit_linear_design", boom)
    monkeypatch.setattr("development_attack.analyze._fit_logistic", boom)
    pattern = ["transfer"] * 13 + ["failure"] * 3
    progress, scalars = _run(_full(pattern, 8))
    result = analyze_run(progress, scalars, VERIFICATION)
    _assert_no_coefficient(result)
    assert result["stop_reason"] == "effective_sample"
    assert "effective_sample" in result["failed_checks"]
    assert result["eligible"]["n_fail"] == 3
    assert result["eligible"]["n_transfer"] == 13
    primary = next(row for row in result["counts"] if row["role"] == "primary")
    assert primary["n_source_success"] == 16
    assert primary["n_source_success_denominator"] == 16
    assert primary["n_conditional_transfer"] == 13
    assert primary["n_conditional_transfer_denominator"] == 16
    assert primary["n_conditional_transfer_failure"] == 3
    text = render_report(result, progress, scalars)
    assert HSTAR_UNESTIMATED in text
    assert "No coefficient was computed." in text
    assert "beta_a" not in text


def test_a_predictor_exclusion_is_not_counted_as_an_eligible_failure(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("H* was fit")

    monkeypatch.setattr("development_attack.analyze.fit_linear_design", boom)
    pattern = ["transfer", "failure"] * 40
    specs = _full(pattern, 40)
    failure_indexes = [index for index, (kind, _group) in enumerate(specs) if kind == "failure"]
    progress, scalars = _run(specs, bad_indexes={failure_indexes[0]})
    result = analyze_run(progress, scalars, VERIFICATION)
    _assert_no_coefficient(result)
    assert result["stop_reason"] == "effective_sample"
    assert result["eligible"]["n_fail"] == 39
    assert result["eligible"]["n_transfer"] == 40
    assert result["predictor_exclusions"]["alignment_undefined"] == 1
    assert result["predictor_exclusions"]["n_scenes"] == 1


def test_passing_sample_estimates_and_leaves_confirmation_false():
    pattern = ["transfer", "failure"] * 40
    progress, scalars = _run(_full(pattern, 40))
    result = analyze_run(progress, scalars, VERIFICATION)
    assert result["failed_checks"] == []
    assert result["h_star_estimated"] is True
    assert result["stop_reason"] == "estimated"
    assert result["confirmatory_supported"] is False
    assert result["h_star"]["confirmatory_supported"] is False
    assert math.isfinite(result["h_star"]["beta_a"])
    assert result["h_star"]["ci_low"] <= result["h_star"]["beta_a"] <= result["h_star"]["ci_high"]
    assert result["eligible"]["n_fail"] == 40
    assert result["eligible"]["n_transfer"] == 40
    assert result["eligible"]["n_groups"] >= MIN_GROUPS
    text = render_report(result, progress, scalars)
    assert "β_A" in text
    assert "confirmatory_supported" in text


def test_verification_failure_does_not_estimate(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("H* was fit")

    monkeypatch.setattr("development_attack.analyze.fit_linear_design", boom)
    pattern = ["transfer", "failure"] * 40
    progress, scalars = _run(_full(pattern, 40))
    result = analyze_run(progress, scalars, {"passed": False})
    _assert_no_coefficient(result)
    assert result["stop_reason"] == "verification_failed"


def test_constant_alignment_stops_before_a_fit(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("H* was fit")

    monkeypatch.setattr("development_attack.analyze.fit_linear_design", boom)
    pattern = ["transfer", "failure"] * 40
    progress, scalars = _run(_full(pattern, 40))
    for row in scalars:
        row["alignment_neutral_to_helpful"] = 0.2
    result = analyze_run(progress, scalars, VERIFICATION)
    _assert_no_coefficient(result)
    assert result["stop_reason"] == "flat_predictor"


def test_an_unfinished_run_does_not_apply_the_floor():
    progress = {
        "status": "stopped_budget",
        "n_finished": 0,
        "n_scenes": N_SCENES,
        "scene_ids": [],
        "scenes": {},
        "stopped_reason": "memory",
        "protocol_commit": STARTED_FROM_COMMIT,
    }
    result = analyze_run(progress, [], VERIFICATION)
    assert result["effective_sample_applied"] is False
    assert result["h_star"] is None
    assert result["h_star_estimated"] is False
    assert result["stop_reason"] == "stopped_budget"
    assert "did not finish" in result["h_star_statement"]


def test_ancestor_pids_include_this_process():
    import os

    assert os.getpid() in _ancestor_pids()


def test_caffeinate_wrapper_is_not_another_model():
    wrapped = (
        "caffeinate -dimsu -- /Users/fedecava/Development/_c0-work/venv-vlm/bin/python "
        "-u -m development_attack"
    )
    assert _is_other_model_command(wrapped) is False
    assert _is_other_model_command(
        "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -u -m development_attack"
    )
    assert _is_other_model_command(
        "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m attack_calibration"
    )


def test_fewer_than_six_groups_stops(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("H* was fit")

    monkeypatch.setattr("development_attack.analyze.fit_linear_design", boom)
    specs = []
    for index in range(80):
        kind = "transfer" if index < 40 else "failure"
        specs.append((kind, f"g{index // 16:02d}"))
    for index in range(16):
        specs.append(("out", f"g{index + 5:02d}"))
    progress, scalars = _run(specs)
    result = analyze_run(progress, scalars, VERIFICATION)
    _assert_no_coefficient(result)
    assert result["eligible"]["n_fail"] == 40
    assert result["eligible"]["n_groups"] < MIN_GROUPS
    assert result["stop_reason"] == "too_few_groups"
