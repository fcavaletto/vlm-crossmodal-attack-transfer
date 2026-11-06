"""Gate, alignment, and loss identity. No model load."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from analysis.metrics import sign_alignment
from dev_pilot.fixtures import load_manifest

from qwen_rgb_loss.gate import gate_scene_ids, judge_gate
from qwen_rgb_loss.loss_check import same_row_identity
from qwen_rgb_loss.protocol import GIB_BYTES, TOKEN_IDS, assert_unchanged_lines
from qwen_rgb_loss.records import (
    clean_margin,
    directed_alignments,
    distribution,
    gradient_record,
    stability_report,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "qwen_rgb_loss" / "qwen_rgb_loss"


def test_gate_ids_follow_scene_id_and_ignore_gradient_fields():
    records = [
        {"scene_id": "b", "answer": "red", "l1": 100.0, "gradient": [1.0]},
        {"scene_id": "a", "answer": "red", "l1": 0.0},
        {"scene_id": "d", "answer": "blue", "alignment": -1},
        {"scene_id": "c", "answer": "blue", "alignment": 1},
        {"scene_id": "f", "answer": "green"},
        {"scene_id": "e", "answer": "green", "nonzero": False},
        {"scene_id": "h", "answer": "yellow"},
        {"scene_id": "g", "answer": "yellow", "abs_sum": 9},
    ]
    assert gate_scene_ids(records) == ["a", "c", "e", "g"]


def test_locked_manifest_selects_one_scene_per_color():
    records = load_manifest(ROOT)
    assert gate_scene_ids(records) == [
        "dev4-ref-00-a",
        "dev4-ref-00-b",
        "dev4-ref-01-b",
        "dev4-ref-14-b",
    ]


def test_gate_passes_only_for_finite_nonzero_gradients():
    scenes = ["s0", "s1", "s2", "s3"]
    good = {
        scene: {
            "neutral": {"gradient": {"present": True, "finite": True, "nonzero": True, "l1": 1.0}},
            "helpful_nonrevealing": {"gradient": {"present": True, "finite": True, "nonzero": True, "l1": 1.0}},
        }
        for scene in scenes
    }
    assert judge_gate(good, scenes)["passed"] is True
    good["s1"]["neutral"]["gradient"] = {
        "present": True,
        "finite": True,
        "nonzero": False,
        "l1": 0.0,
        "reason": "gradient is zero",
    }
    failed = judge_gate(good, scenes)
    assert failed["passed"] is False
    assert failed["failures"][0]["reason"] == "gradient is zero"
    good["s1"]["neutral"]["gradient"] = {"present": False, "l1": 0.0, "reason": "budget refused"}
    missing = judge_gate(good, scenes)
    assert missing["passed"] is False
    assert missing["failures"][0]["reason"] == "budget refused"


def test_missing_gradient_is_not_stored_as_zero():
    missing = gradient_record(None, missing_reason="the pre-load budget refused the load")
    assert missing["present"] is False
    assert missing["l1"] is None
    assert missing["l2"] is None
    zero = gradient_record(np.zeros((3, 2, 2), dtype=np.float32))
    assert zero["present"] is True
    assert zero["finite"] is True
    assert zero["nonzero"] is False
    assert zero["l1"] == 0.0


def test_sign_alignment_matches_the_worked_example_and_both_directions():
    source = np.array([1.0, -2.0, 0.0, 0.5])
    target = np.array([3.0, -1.0, 4.0, -2.0])
    assert sign_alignment(source, target)["alignment_a"] == pytest.approx(0.2)
    pair = directed_alignments(source, target)
    assert pair["alignment_neutral_to_helpful"] == pytest.approx(0.2)
    assert pair["alignment_helpful_to_neutral"] == pytest.approx(2.5 / 3.5)
    assert pair["eligibility_alignment"] == pytest.approx(0.2)
    flat = directed_alignments(source, np.zeros(4))
    assert flat["alignment_neutral_to_helpful"] is None
    assert flat["eligibility_alignment"] is None
    assert flat["undefined_reason"]


def test_distribution_excludes_nulls_from_the_mean():
    summary = distribution([1.0, None, -1.0], undefined=0, missing=1)
    assert summary["n_defined"] == 2
    assert summary["mean"] == 0.0
    assert summary["n_missing"] == 1
    assert summary["nulls_imputed_as_zero"] is False
    assert summary["histogram_counts"][0] == 1
    assert summary["histogram_counts"][-1] == 1


def test_stability_distinguishes_a_numerical_miss_from_an_invalid_gradient():
    base = np.ones((2, 2), dtype=np.float32)
    confirmed = stability_report(base, base + 1e-6)
    assert confirmed["numerical_check"] == "confirmed"
    assert confirmed["gradient_invalid"] is False
    shifted = stability_report(base, -base)
    assert shifted["numerical_check"] == "failed"
    assert shifted["gradient_invalid"] is False
    invalid = stability_report(base, np.zeros_like(base))
    assert invalid["gradient_invalid"] is True
    assert invalid["numerical_check"] == "not_applied"


def test_clean_margin_is_the_gap_above_the_labeled_loss():
    assert clean_margin({"red": 0.2, "blue": 1.5, "green": 0.4, "yellow": 3.0}, "red") == pytest.approx(0.2)


def test_same_row_loss_formulas_agree_and_backpropagate():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 8, dtype=torch.float32, requires_grad=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    identity = same_row_identity(logits, input_ids, prompt_length=5, token_id=3)
    assert identity["agreed"] is True
    assert identity["max_abs"] <= 1e-5
    identity["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad[0, -1].abs().sum()) > 0.0
    assert float(logits.grad[0, 0].abs().sum()) == 0.0


def test_locked_lines_match_checkpoint_1_7():
    assert_unchanged_lines()
    assert GIB_BYTES == 6_442_450_944
    assert TOKEN_IDS["red"] == (1151,)
    assert TOKEN_IDS["yellow"] == (27869,)


def test_package_has_one_model_load_and_no_attack_step():
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(PACKAGE.glob("*.py")))
    assert "from_pretrained(" not in source
    assert "project_linf" not in source
    assert "FGSM" not in source
    assert "PGD" not in source
    assert source.count("load_model_after_budget(") == 1
    tree = ast.parse((PACKAGE / "run.py").read_text(encoding="utf-8"))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("load_model_after_budget") == 1


def test_stored_qwen_rows_still_match_the_locked_rule():
    from qwen_rgb_loss.offline import verify_locked_inputs

    report = verify_locked_inputs(ROOT)
    assert report["primary_rows"] == 192
    assert report["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert report["predictions_rewritten"] is False


def test_protocol_names_the_rule_and_the_byte_floor():
    text = (ROOT / "docs" / "checkpoint-1.8" / "MEASUREMENT_PROTOCOL.md").read_text(encoding="utf-8")
    manifest_ids = gate_scene_ids(load_manifest(ROOT))
    for scene_id in manifest_ids:
        assert scene_id in text
    assert "lexicographic-first-scene-per-labeled-color" in text
    assert "6,442,450,944" in text
    assert "H* is not fit" in text
    assert "No attack is run" in text
    packet = (ROOT / "docs" / "checkpoint-1.7" / "REVIEW_PACKET.md").read_text(encoding="utf-8")
    assert "6373998592" in packet
    assert "7759183872" in packet
    assert "6,442,450,944" in packet
    assert "also above 6 GiB" not in packet
    assert "6,078.71875 MiB" in packet
