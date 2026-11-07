"""Calibration rules with no model load."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from analysis.metrics import conditional_transfer
from attack_feasibility.steps import pgd_update, project_signed_step, scene_seed
from dev_pilot.fixtures import load_manifest
from qwen_rgb.constraints import round_trip_uint8

from attack_calibration.counts import (
    analysis_rows,
    configuration_counts,
    edge_position,
    flatten_scenes,
    scene_flags,
)
from attack_calibration.prior import describe_scenes, load_prior
from attack_calibration.protocol import (
    BUDGETS,
    FROZEN_SCENES,
    GROUP_INDEXES,
    PGD_STEPS,
    RECORDED_STEPS,
    SEED_PREFIX,
    assert_protocol_committed,
    configurations,
    epsilon,
)
from attack_calibration.report import render_report
from attack_calibration.scene import exact_linf, run_scene
from attack_calibration.select import select_calibration

ROOT = Path(__file__).resolve().parents[2]


def _records(n_groups: int) -> list[dict]:
    rows = []
    for index in range(n_groups):
        group = f"g{index:02d}"
        for arm, answer in (("a", "red"), ("b", "blue")):
            rows.append(
                {
                    "counterfactual_group_id": group,
                    "scene_id": f"s{index:02d}-{arm}",
                    "answer": answer,
                }
            )
    return rows


def test_development_manifest_matches_the_protocol_table():
    selected = select_calibration(load_manifest(ROOT))
    assert selected["ok"] is True
    assert selected["n_groups"] == 48
    assert selected["selected_group_ids"] == [row[1] for row in FROZEN_SCENES][::2]
    got = [(row["scene_id"], row["group_id"], row["answer"], row["seed"]) for row in selected["scenes"]]
    assert got == list(FROZEN_SCENES)
    assert [scene_seed(SEED_PREFIX, row[0]) for row in FROZEN_SCENES] == [row[3] for row in FROZEN_SCENES]
    text = (ROOT / "docs/checkpoint-2.0/CALIBRATION_PROTOCOL.md").read_text(encoding="utf-8")
    assert "counterfactual-group-stride-3" in text
    assert "If the group count is not 48" in text
    assert "Near the floor means the numerator is 0 or 1." in text
    assert "Near the ceiling means the numerator is the denominator or one less." in text
    assert "It is not a population attack-success rate." in text
    cursor = 0
    for scene_id, _group, _answer, seed in FROZEN_SCENES:
        found = text.find(f"`{scene_id}`", cursor)
        assert found > cursor
        assert str(seed) in text[found : found + 80]
        cursor = found
    assert_protocol_committed(ROOT)


def test_a_group_count_other_than_48_does_not_invent_a_shorter_stride():
    selected = select_calibration(_records(47))
    assert selected["ok"] is False
    assert selected["n_groups"] == 47
    assert selected["scenes"] == []
    assert "not 48" in selected["reason"]


def test_the_recorded_grid_is_only_the_listed_budgets_and_steps():
    assert list(GROUP_INDEXES) == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
    assert [numerator for _key, numerator in BUDGETS] == [1.0, 2.0]
    assert RECORDED_STEPS == (2, 3, 4, 5, 7)
    assert PGD_STEPS == 7
    grid = configurations()
    assert len(grid) == 14
    assert {row["epsilon"] for row in grid} == {"1/255", "2/255"}
    assert {row["steps"] for row in grid if row["method"] == "pgd"} == {2, 3, 4, 5, 7}
    assert {row["method"] for row in grid} == {"pgd", "fgsm", "random"}
    assert epsilon(1) == 1 / 255
    assert epsilon(2) / 4 == 2 / 255 / 4


def test_pgd_and_the_positive_step_use_the_checkpoint_1_9_projection():
    clean = torch.full((3, 2, 2), 0.5)
    gradient = torch.ones_like(clean)
    eps = epsilon(2)
    updated = pgd_update(clean, clean, gradient, eps / 4.0, eps)
    assert torch.allclose(updated, clean + eps / 4.0)
    assert float(exact_linf(updated, clean)) <= eps + 1e-12
    stored = round_trip_uint8(updated)
    assert exact_linf(stored, clean) > 0.0
    half = project_signed_step(torch.zeros((3, 2, 2)), torch.ones((3, 2, 2)), epsilon(0.5), "positive")
    assert exact_linf(round_trip_uint8(half), torch.zeros((3, 2, 2))) == 0.0


def test_the_attack_function_does_not_take_alignment_or_the_manifest_seed():
    assert "alignment" not in inspect.signature(run_scene).parameters
    source = inspect.getsource(run_scene)
    assert 'record["seed"]' not in source
    assert "hstar" not in source.lower()
    package = ROOT / "src" / "attack_calibration"
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "linregress" not in text
        assert "corrcoef" not in text
        assert "hstar" not in text.lower()


def test_missing_decisions_stay_out_of_both_counts():
    rows = [
        _flat("s0", "pgd", "1/255", 2, "red", "red", "red", "blue", "blue"),
        _flat("s1", "pgd", "1/255", 2, "red", "red", "red", "blue", "red"),
        _flat("s2", "pgd", "1/255", 2, "red", "red", "blue", "green", "green"),
        _flat("s3", "pgd", "1/255", 2, "red", "red", "red", "red", "red"),
        _flat("s4", "pgd", "1/255", 2, "red", "blue", "blue", "green", "green"),
        _flat("s5", "pgd", "1/255", 2, "red", "red", "red", None, "red"),
    ]
    block = configuration_counts(rows)[0]
    assert block["configuration"] == "pgd|1/255|2"
    assert block["n_source_success"] == 3
    assert block["n_source_success_denominator"] == 4
    assert block["n_source_missing"] == 1
    assert block["n_excluded_target_not_clean"] == 1
    assert block["n_conditional_transfer"] == 1
    assert block["n_conditional_transfer_denominator"] == 2
    assert block["n_transfer_missing"] == 0
    assert block["adequate_source_and_transfer_variation"] is False
    assert edge_position(0, 32) == "near_floor"
    assert edge_position(1, 32) == "near_floor"
    assert edge_position(2, 32) == "interior"
    assert edge_position(30, 32) == "interior"
    assert edge_position(31, 32) == "near_ceiling"
    assert edge_position(32, 32) == "near_ceiling"
    assert edge_position(1, 1) == "undefined"
    compared = conditional_transfer(
        analysis_rows(rows[:5]),
        method="pgd",
        epsilon=1 / 255,
        step_budget=2,
        source_context="neutral",
        target_context="helpful_nonrevealing",
    )
    assert compared["n_source_success"] == 3
    assert compared["n_excluded_target_not_correct_before"] == 1
    assert compared["n_eligible"] == 2
    assert compared["n_target_success"] == 1


def test_checkpoint_1_8_scalars_are_stored_and_not_a_selection_input():
    description = describe_scenes(load_prior(ROOT), ["dev4-ref-00-a"])
    row = description[0]
    assert row["alignment_neutral_to_helpful"] == 0.9110217380928537
    assert row["neutral_margin"] == 8.75000016298145
    assert row["helpful_margin"] == 6.999999950174242
    assert row["neutral_gradient_l1"] == 72.17303015444587
    assert row["used_to_choose_the_sample"] is False
    assert row["used_to_choose_a_configuration"] is False
    flags = scene_flags(
        "red",
        {"neutral": {"decision": "red", "margin": 1.0, "nll": 0.1}, "helpful_nonrevealing": {"decision": "red", "margin": 1.0, "nll": 0.2}},
        {"neutral": {"decision": "blue", "margin": -1.0, "nll": 2.0}, "helpful_nonrevealing": {"decision": "blue", "margin": -1.0, "nll": 2.0}},
    )
    assert flags["source_flip"] is True
    assert flags["transfer"] is True


def test_the_report_lists_counts_and_does_not_rank_by_alignment():
    progress = {
        "status": "completed",
        "stopped_reason": "The listed budgets and recorded steps finished. Epsilon was not raised and no step was added.",
        "protocol_commit": "abc",
        "selection_rule": "counterfactual-group-stride-3",
        "n_groups": 48,
        "n_finished": 4,
        "n_scenes": 4,
        "configurations": configuration_counts(
            [
                _flat(scene_id, "pgd", "1/255", step, "red", clean_neutral, clean_helpful, pert_neutral, pert_helpful)
                for step in (2, 3, 4, 5, 7)
                for scene_id, clean_neutral, clean_helpful, pert_neutral, pert_helpful in (
                    ("s0", "red", "red", "blue", "blue"),
                    ("s1", "red", "red", "blue", "blue"),
                    ("s2", "red", "red", "blue", "red"),
                    ("s3", "red", "red", "blue", "red"),
                    ("s4", "red", "red", "red", "red"),
                    ("s5", "red", "red", "red", "red"),
                )
            ]
        ),
        "loaded_model": {"device": "mps", "dtype": "torch.bfloat16"},
        "available_bytes_entering_decision": 12000000000,
        "available_bytes_at_decision": 12000000000,
    }
    # The synthetic rows above only fill pgd at 1/255. Other grid rows have empty counts.
    text = render_report(progress, [])
    assert "pgd|1/255|2" in text
    assert "4/6 source success, 2/4 conditional transfer" in text
    assert "alignment was not used" in text
    assert "Epsilon stayed inside {1/255, 2/255}" in text
    assert "hstar" not in text.lower()
    empty = {
        "status": "completed",
        "configurations": [],
        "n_groups": 48,
        "n_finished": 0,
        "n_scenes": 32,
    }
    assert "No configuration was recorded." in render_report(empty, [])
    finished = flatten_scenes(
        {
            "s0": {
                "finished": True,
                "answer": "red",
                "counterfactual_group_id": "g",
                "clean": {
                    "neutral": {"decision": "red", "margin": 1.0, "nll": 0.2},
                    "helpful_nonrevealing": {"decision": "red", "margin": 1.0, "nll": 0.3},
                },
                "epsilons": {},
                "error": "the clean gradient was missing, non-finite, or zero",
            }
        },
        ["s0"],
    )
    assert len(finished) == 14
    assert all(row["perturbed"]["neutral"]["decision"] is None for row in finished)
    assert configuration_counts(finished)[0]["n_source_missing"] == 1
    assert configuration_counts(finished)[0]["n_source_success_denominator"] == 0


def _flat(scene_id, method, eps, steps, answer, clean_neutral, clean_helpful, pert_neutral, pert_helpful):
    def view(decision, nll):
        if decision is None:
            return {"decision": None, "margin": None, "nll": None}
        return {"decision": decision, "margin": 0.5, "nll": nll}

    return {
        "scene_id": scene_id,
        "answer": answer,
        "method": method,
        "epsilon": eps,
        "epsilon_rgb": epsilon(1 if eps == "1/255" else 2),
        "steps": steps,
        "clean": {
            "neutral": view(clean_neutral, 0.1),
            "helpful_nonrevealing": view(clean_helpful, 0.2),
        },
        "perturbed": {
            "neutral": view(pert_neutral, 1.0),
            "helpful_nonrevealing": view(pert_helpful, 1.2),
        },
    }
