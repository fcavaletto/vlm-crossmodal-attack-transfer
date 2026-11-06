"""The assembled audit reads Checkpoint 1.6 and does not rewrite it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from study2_audit.audit import build_audit
from study2_audit.cluster import descriptive_cluster_mean, descriptive_group_rate
from study2_audit.qwen_record import QWEN_SUCCESS_SHA256

ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "docs/checkpoint-1.6/qwen_feasibility.json"


def test_cluster_mean_uses_groups_and_does_not_fit_h_star():
    values = {"a": 0.0, "b": 0.0, "c": 1.0}
    clusters = {"a": "g1", "b": "g1", "c": "g2"}
    first = descriptive_cluster_mean(values, clusters, n_boot=300, seed=5)
    second = descriptive_cluster_mean(values, clusters, n_boot=300, seed=5)
    assert first == second
    assert first["estimate"] == pytest.approx(1.0 / 3.0)
    assert first["unit"] == "counterfactual_group"
    assert first["h_star_fitted"] is False
    assert first["n_groups"] == 2
    flat = descriptive_group_rate({"g1": 0.0, "g2": 0.0}, n_boot=50, seed=1)
    assert flat["estimate"] == 0.0
    assert flat["width"] == 0.0
    assert flat["h_star_fitted"] is False


def test_build_audit_leaves_the_qwen_file_and_refuses_support():
    before = hashlib.sha256(QWEN.read_bytes()).hexdigest()
    report = build_audit(ROOT, n_replications=20, n_boot=49, n_groups_grid=(9, 48), n_boot_descriptive=200)
    after = hashlib.sha256(QWEN.read_bytes()).hexdigest()
    assert before == after == QWEN_SUCCESS_SHA256
    assert report["original_qwen_json_rewritten"] is False
    assert report["h_star_fitted"] is False
    assert report["model_inference"] is False
    assert report["attacks_run"] is False
    assert report["reproducibility_ok"] is True
    assert report["qwen_records_match_pins"] is True
    assert report["eligibility"]["n_groups_eligible"] is None
    assert report["eligibility"]["establishes_h_star"] is False
    assert report["published_three_family_precision"]["refit"] is False
    assert report["published_three_family_precision"]["recommended_minimum_eligible_groups"] == 24
    assert report["published_three_family_precision"]["sentence_says_does_not_promise"] is True
    assert report["decision"]["decision"] == "no-go"
    assert report["decision"]["supports_h_star"] is False
    assert report["precision"]["h_star_fitted_on_real_data"] is False
    neutral = report["descriptive_intervals"]["by_context"]["neutral"]["cluster"]
    assert neutral["estimate"] == 14 / 24
    assert neutral["h_star_fitted"] is False
    revealing = report["ceilings"]["parser_ceiling"]
    assert revealing["strict_correct"] == 0
    assert revealing["invalid"] == 24
    assert revealing["class_argmin_correct"] == 22
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "study2_audit").rglob("*.py"))
    assert "import torch" not in source
    assert "from_pretrained(" not in source
