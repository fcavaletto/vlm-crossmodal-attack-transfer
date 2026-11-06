"""Schema checks on hand-built rows. No model output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.schemas import (
    CONTEXTS,
    SchemaError,
    epsilon_key,
    load_jsonl,
    validate_attacks,
    validate_gradient_summaries,
    validate_gradient_vectors,
    validate_losses,
    validate_predictions,
)

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "src" / "analysis" / "analysis" / "fixtures" / "toy"


def _prediction(**overrides):
    row = {
        "scene_id": "toy-01",
        "context": "neutral",
        "parsed_answer": "alpha",
        "invalid": False,
        "correct": True,
    }
    row.update(overrides)
    return row


def _attack(**overrides):
    row = {
        "scene_id": "toy-01",
        "source_context": "neutral",
        "target_context": "neutral",
        "method": "pgd",
        "epsilon": 0.01,
        "step_budget": 1,
        "correct_before": True,
        "correct_after": False,
        "invalid_after": False,
    }
    row.update(overrides)
    return row


def test_toy_prediction_file_loads_and_uses_only_toy_scenes():
    rows = validate_predictions(load_jsonl(TOY / "predictions.jsonl"))
    assert len(rows) == 24
    assert {row["scene_id"] for row in rows} == {f"toy-{index:02d}" for index in range(1, 7)}
    assert {row["context"] for row in rows} == set(CONTEXTS)


def test_unknown_field_and_missing_field_are_rejected():
    with pytest.raises(SchemaError, match="unknown"):
        validate_predictions([(1, _prediction(extra=1))])
    broken = _prediction()
    del broken["correct"]
    with pytest.raises(SchemaError, match="missing"):
        validate_predictions([(1, broken)])


def test_invalid_and_correct_cannot_both_be_true():
    with pytest.raises(SchemaError, match="invalid"):
        validate_predictions([(1, _prediction(invalid=True, correct=True, parsed_answer=""))])


def test_boolean_fields_reject_integers():
    with pytest.raises(SchemaError, match="boolean"):
        validate_predictions([(1, _prediction(correct=1))])
    with pytest.raises(SchemaError, match="integer"):
        validate_attacks([(1, _attack(step_budget=True))])


def test_duplicate_scene_context_is_rejected():
    row = _prediction()
    with pytest.raises(SchemaError, match="duplicate"):
        validate_predictions([(1, row), (2, dict(row))])


def test_context_and_method_labels():
    with pytest.raises(SchemaError, match="context"):
        validate_predictions([(1, _prediction(context="supportive"))])
    with pytest.raises(SchemaError, match="method"):
        validate_attacks([(1, _attack(method="adam"))])
    with pytest.raises(SchemaError, match="together"):
        validate_attacks([(1, _attack(method="pgd", source_context="multi"))])
    with pytest.raises(SchemaError, match="together"):
        validate_attacks([(1, _attack(method="pgd_multi", source_context="neutral"))])
    with pytest.raises(SchemaError, match="target_context"):
        validate_attacks([(1, _attack(target_context="multi"))])
    accepted = validate_attacks(
        [(1, _attack(method="pgd_multi", source_context="multi", target_context="neutral"))]
    )
    assert accepted[0]["method"] == "pgd_multi"


def test_loss_and_alignment_ranges():
    with pytest.raises(SchemaError, match="finite"):
        validate_losses([(1, {"scene_id": "toy-01", "context": "neutral", "loss": float("nan")})])
    with pytest.raises(SchemaError, match="\\[-1, 1\\]"):
        validate_gradient_summaries(
            [
                (
                    1,
                    {
                        "scene_id": "toy-01",
                        "source_context": "neutral",
                        "target_context": "neutral",
                        "alignment_a": 2,
                    },
                )
            ]
        )
    rows = validate_gradient_summaries(
        [
            (
                1,
                {
                    "scene_id": "toy-01",
                    "source_context": "neutral",
                    "target_context": "neutral",
                    "alignment_a": None,
                },
            )
        ]
    )
    assert rows[0]["alignment_a"] is None


def test_empty_gradient_vector_is_rejected():
    with pytest.raises(SchemaError, match="non-empty"):
        validate_gradient_vectors(
            [
                (
                    1,
                    {
                        "scene_id": "toy-01",
                        "source_context": "neutral",
                        "target_context": "neutral",
                        "g_source": [],
                        "g_target": [1.0],
                    },
                )
            ]
        )


def test_epsilon_key_groups_at_eight_decimals():
    assert epsilon_key(0.01) == "0.01000000"
    assert epsilon_key(0.02) == "0.02000000"
    with pytest.raises(SchemaError):
        epsilon_key(-0.1)


def test_attack_invalid_and_correct_is_rejected():
    with pytest.raises(SchemaError, match="invalid"):
        validate_attacks([(1, _attack(correct_after=True, invalid_after=True))])


def test_jsonl_loader_skips_blank_lines_and_rejects_non_objects(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\n{"scene_id": "toy-01"}\n\n', encoding="utf-8")
    loaded = load_jsonl(path)
    assert loaded == [(2, {"scene_id": "toy-01"})]
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="JSON object"):
        load_jsonl(path)
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_jsonl(path)
    unused = json.dumps({"ok": True})
    assert unused.startswith("{")
