"""Eligible-group count from compact records. No model and no H* fit."""

from __future__ import annotations

import pytest

from qwen_rgb_loss.summarize import build_summary


def _records(n_groups: int = 48) -> list[dict]:
    rows = []
    for group in range(n_groups):
        for arm, answer in (("a", "red"), ("b", "blue")):
            rows.append(
                {
                    "scene_id": f"dev4-ref-{group:02d}-{arm}",
                    "counterfactual_group_id": f"cf4-ref-{group:02d}",
                    "answer": answer,
                    "question": "What color is the small circle in the top row?",
                    "ambiguity_flags": [],
                    "contexts": {
                        "neutral": {"text": "Several simple objects appear on a plain background.", "reveals_answer": False},
                        "helpful_nonrevealing": {
                            "text": "Use the object words in the question together. An object that matches only part of that description is a different object.",
                            "reveals_answer": False,
                        },
                    },
                }
            )
    return rows


def _predictions(records: list[dict], *, wrong: set[str] | None = None) -> list[dict]:
    wrong = wrong or set()
    rows = []
    for record in records:
        for context in ("neutral", "helpful_nonrevealing"):
            correct = record["scene_id"] not in wrong
            rows.append(
                {
                    "scene_id": record["scene_id"],
                    "context": context,
                    "free_generation": {"correct": correct, "valid_but_wrong": not correct, "invalid": False},
                }
            )
    return rows


def _scenes(records: list[dict], *, drop: set[str] | None = None) -> dict:
    drop = drop or set()
    scenes = {}
    for record in records:
        scene_id = record["scene_id"]
        present = scene_id not in drop
        gradient = (
            {"present": True, "finite": True, "nonzero": True, "l1": 1.25, "l2": 0.5}
            if present
            else {"present": False, "finite": None, "nonzero": None, "l1": None, "l2": None, "reason": "not computed"}
        )
        scenes[scene_id] = {
            "scene_id": scene_id,
            "neutral": {"gradient": gradient, "margin": 1.0 if present else None},
            "helpful_nonrevealing": {"gradient": gradient, "margin": 1.0 if present else None},
            "alignment_neutral_to_helpful": 0.25 if present else None,
            "alignment_helpful_to_neutral": -0.1 if present else None,
            "eligibility_alignment": 0.25 if present else None,
            "undefined_reason": None if present else "a gradient is missing",
        }
    return scenes


def test_complete_pass_counts_groups_and_incomplete_pass_stays_unset():
    records = _records()
    scene_ids = [record["scene_id"] for record in records]
    summary = build_summary(
        scenes=_scenes(records),
        records=records,
        rows=_predictions(records),
        scene_ids=scene_ids,
        gate={"passed": True},
    )
    assert summary["measurement_complete"] is True
    assert summary["n_groups_eligible"] == 48
    assert summary["any_gradient_missing"] is False
    assert summary["h_star_fitted"] is False
    assert summary["alignment"]["neutral_to_helpful"]["mean"] == 0.25
    assert summary["alignment"]["helpful_to_neutral"]["mean"] == pytest.approx(-0.1)
    assert summary["alignment"]["h_star_fitted"] is False

    partial = build_summary(
        scenes=_scenes(records, drop={"dev4-ref-00-a", "dev4-ref-00-b"}),
        records=records,
        rows=_predictions(records),
        scene_ids=scene_ids,
        gate={"passed": True},
    )
    assert partial["measurement_complete"] is True
    assert partial["n_groups_eligible"] == 47
    assert partial["any_gradient_missing"] is True
    assert partial["missing_gradients"][0]["reason"] == "not computed"
    assert all(item["reason"] != "0" for item in partial["missing_gradients"])

    incomplete_scenes = _scenes(records)
    incomplete_scenes.pop("dev4-ref-47-b")
    unset = build_summary(
        scenes=incomplete_scenes,
        records=records,
        rows=_predictions(records),
        scene_ids=scene_ids,
        gate={"passed": False},
    )
    assert unset["measurement_complete"] is False
    assert unset["n_groups_eligible"] is None
    assert unset["eligible"]["h_star_fitted"] is False
