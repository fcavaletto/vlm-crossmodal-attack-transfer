"""The scene rule and the stop rule, with no model load."""

from __future__ import annotations

from pathlib import Path

import torch

from attack_feasibility.judge import judge_stage_a, judge_stage_b
from attack_feasibility.protocol import (
    STAGE_A_SCENES,
    STAGE_B_SCENE_IDS,
    epsilon,
)
from attack_feasibility.select import assert_frozen_selection, select_stage_a, select_stage_b
from attack_feasibility.steps import (
    matched_rademacher,
    pgd_update,
    project_signed_step,
    round_trip_metrics,
    scene_seed,
)
from qwen_rgb.constraints import linf_distance

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_scenes_match_the_checkpoint_1_8_record_and_the_protocol_text():
    selected = assert_frozen_selection(ROOT)
    assert [row["scene_id"] for row in selected["stage_a"]] == [row[0] for row in STAGE_A_SCENES]
    assert selected["stage_b"] == list(STAGE_B_SCENE_IDS)
    text = (ROOT / "docs/checkpoint-1.9/ATTACK_PROTOCOL.md").read_text(encoding="utf-8")
    assert "per-color-extreme-alignment-exploratory" in text
    assert "scene-id-stride-6" in text
    assert "cannot support an unbiased population attack-success rate" in text
    cursor = 0
    for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES:
        found = text.find(f"`{scene_id}`", cursor)
        assert found > cursor
        cursor = found
    assert "A missing observed change is not imputed as zero" in text


def test_stage_a_uses_alignment_extremes_and_stage_b_uses_only_scene_id_order():
    progress = {
        "status": "completed",
        "attacks_run": False,
        "scenes": {},
    }
    colors = ("red", "blue", "green", "yellow")
    for index in range(96):
        scene_id = f"dev4-ref-{index // 2:02d}-{'a' if index % 2 == 0 else 'b'}"
        color = colors[index // 24]
        progress["scenes"][scene_id] = {
            "answer": color,
            "alignment_neutral_to_helpful": (index % 24) / 23,
        }
    # The real record is what the frozen table locks. This synthetic check only
    # confirms the rule: two extremes per color, and stride 6 on sorted ids.
    stage_a = select_stage_a(progress)
    assert len(stage_a) == 8
    assert [row["answer"] for row in stage_a] == ["red", "red", "blue", "blue", "green", "green", "yellow", "yellow"]
    assert [row["stratum"] for row in stage_a] == ["lowest_alignment_within_color", "highest_alignment_within_color"] * 4
    assert select_stage_b(progress)[0] == "dev4-ref-00-a"
    assert select_stage_b(progress)[1] == "dev4-ref-03-a"
    assert len(select_stage_b(progress)) == 16


def test_positive_step_stays_in_the_ball_and_does_not_point_against_the_gradient():
    clean = torch.full((3, 4, 4), 0.4)
    clean[0, 0, 0] = 0.0
    clean[1, 0, 1] = 1.0
    gradient = torch.zeros_like(clean)
    gradient[0] = 0.2
    gradient[1] = -0.5
    gradient[2, 0, 0] = 0.0
    eps = epsilon(4)
    projected = project_signed_step(clean, gradient, eps, "positive")
    assert float(projected.min()) >= 0.0
    assert float(projected.max()) <= 1.0
    assert float(linf_distance(projected, clean)) <= eps + 1e-8
    delta = projected - clean
    assert float((delta * gradient).sum()) >= -1e-8
    assert float(delta[2, 0, 0]) == 0.0
    negative = project_signed_step(clean, gradient, eps, "negative")
    assert float(((negative - clean) * gradient).sum()) <= 1e-8


def test_half_level_step_from_zero_rounds_back_to_zero():
    clean = torch.zeros((3, 2, 2))
    gradient = torch.ones_like(clean)
    projected = project_signed_step(clean, gradient, epsilon(0.5), "positive")
    metrics = round_trip_metrics(clean, projected)
    stored = metrics.pop("stored")
    assert torch.allclose(projected, clean + epsilon(0.5))
    assert torch.equal(stored, clean)
    assert metrics["fraction_pixels_changed"] == 0.0
    assert metrics["linf_stored_vs_clean"] == 0.0


def test_matched_random_signs_share_the_gradient_support_and_the_seed():
    gradient = torch.tensor([[[0.0, 1.0], [-2.0, 0.0]]])
    signs = matched_rademacher(gradient, scene_seed("checkpoint-1.9-stage-a", "dev4-ref-01-a"))
    assert torch.equal(signs == 0, gradient == 0)
    assert torch.all(signs.abs().eq(0) | signs.abs().eq(1))
    again = matched_rademacher(gradient, scene_seed("checkpoint-1.9-stage-a", "dev4-ref-01-a"))
    assert torch.equal(signs, again)
    assert scene_seed("checkpoint-1.9-stage-a", "dev4-ref-01-a") != scene_seed(
        "checkpoint-1.9-stage-b", "dev4-ref-01-a"
    )


def test_pgd_step_uses_a_shorter_step_inside_the_listed_ball():
    clean = torch.full((3, 2, 2), 0.5)
    current = clean.clone()
    gradient = torch.ones_like(clean)
    eps = epsilon(4)
    updated = pgd_update(current, clean, gradient, eps / 4.0, eps)
    assert torch.allclose(updated, clean + eps / 4.0)
    assert float(linf_distance(updated, clean)) <= eps + 1e-8


def _scene(scene_id: str, positive: float, negative: float, random: float, valid: bool = True) -> dict:
    def step(value: float) -> dict:
        return {"observed_change": value, "loss_identity_agreed": True}

    budgets = {}
    for key in ("0.5/255", "1/255", "2/255", "4/255"):
        budgets[key] = {
            "valid": valid,
            "positive": step(positive),
            "negative": step(negative),
            "random": step(random),
        }
    return {"budgets": budgets, "scene_id": scene_id}


def test_stage_a_passes_only_when_every_budget_beats_both_controls():
    scenes = {
        scene_id: _scene(scene_id, 1.0, -1.0, 0.0)
        for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES
    }
    assert judge_stage_a(scenes)["passed"] is True
    failing = scenes["dev4-ref-01-a"]["budgets"]["4/255"]
    failing["positive"]["observed_change"] = -1.0
    # One scene is not enough to drop a budget below 6 of 8.
    assert judge_stage_a(scenes)["budgets"]["4/255"]["n_raised"] == 7
    for scene_id in ("dev4-ref-01-a", "dev4-ref-09-b", "dev4-ref-25-a"):
        scenes[scene_id]["budgets"]["4/255"]["positive"]["observed_change"] = -1.0
    failed = judge_stage_a(scenes)
    assert failed["passed"] is False
    assert failed["stage_b_allowed"] is False
    assert failed["budgets"]["0.5/255"]["passed"] is True
    assert failed["budgets"]["4/255"]["passed"] is False


def test_a_missing_stage_a_change_fails_the_budget_and_is_not_zero():
    scenes = {
        scene_id: _scene(scene_id, 1.0, -1.0, 0.0)
        for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES
    }
    scenes["dev4-ref-17-b"]["budgets"]["1/255"]["valid"] = False
    judgment = judge_stage_a(scenes)
    assert judgment["passed"] is False
    assert judgment["budgets"]["1/255"]["n_valid"] == 7
    tied = {
        scene_id: _scene(scene_id, 1.0, 1.0, 0.0)
        for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES
    }
    assert judge_stage_a(tied)["passed"] is False


def test_stage_b_ineffective_rule_does_not_treat_a_tie_as_a_win():
    def scene(flip: bool, pgd: float, fgsm: float, random: float) -> dict:
        chosen = "blue" if flip else "red"

        def arm(increase: float, decision: str) -> dict:
            return {"uint8": {"neutral": {"chosen": decision, "labeled_nll_increase": increase}}}

        return {
            "answer": "red",
            "scene_id": "dev4-ref-00-a",
            "clean": {"neutral": {"chosen": "red"}},
            "epsilons": {
                "1/255": {
                    "pgd": arm(pgd, chosen),
                    "fgsm": arm(fgsm, "red"),
                    "random": arm(random, "red"),
                },
                "2/255": {
                    "pgd": arm(0.0, "red"),
                    "fgsm": arm(0.0, "red"),
                    "random": arm(0.0, "red"),
                },
                "4/255": {
                    "pgd": arm(0.0, "red"),
                    "fgsm": arm(0.0, "red"),
                    "random": arm(0.0, "red"),
                },
            },
        }

    ineffective = judge_stage_b([scene(False, 1.0, 1.0, 0.0), scene(False, 1.0, 1.0, 0.0)])
    assert ineffective["ineffective_at_every_listed_budget"] is True
    assert ineffective["raise_epsilon"] is False
    effective = judge_stage_b([scene(True, 0.0, 1.0, 1.0)])
    assert effective["budgets"]["1/255"]["effective"] is True
    assert effective["budgets"]["1/255"]["source_flips"] == 1
    assert effective["budgets"]["1/255"]["source_eligible"] == 1
