"""Hashes, parses, labels, token-id maps, and revisions recompute from the files."""

from __future__ import annotations

from pathlib import Path

from study2_audit.reproducibility import check_reproducibility

ROOT = Path(__file__).resolve().parents[2]


def _check(name: str) -> dict:
    report = check_reproducibility(ROOT)
    return next(item for item in report["checks"] if item["name"] == name)


def test_every_committed_check_matches():
    report = check_reproducibility(ROOT)
    assert report["model_inference"] is False
    assert report["failed"] == []
    assert report["ok"] is True


def test_smol_rows_store_class_nll_and_not_token_ids():
    detail = _check("smol_labels_generations_and_class_nll")["detail"]
    assert detail["n_rows"] == 96
    assert detail["token_id_rows"] == 0
    assert detail["n_problems"] == 0


def test_qwen_token_ids_are_a_stable_map_and_are_not_decoded():
    detail = _check("qwen_token_id_map")["detail"]
    assert detail["decoded"] is False
    assert detail["problems"] == []
    assert detail["max_new_tokens"] == 16
    assert detail["observed_final_id_on_non_truncated_rows"] == 151645


def test_row_color_summary_and_gradient_gate_match_the_stored_files():
    assert _check("row_color_summary_matches_predictions")["ok"] is True
    gate = _check("row_color_gradient_gate_is_one_scene")["detail"]
    assert gate["scene_id"] == "dev3-row-00-a"
    assert gate["n_gradients"] == 2
