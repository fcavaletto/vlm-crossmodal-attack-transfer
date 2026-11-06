"""The committed Qwen JSON is audited and left byte-for-byte in place."""

from __future__ import annotations

import hashlib
from pathlib import Path

from study2_audit.qwen_record import (
    QWEN_PREFLIGHT_SHA256,
    QWEN_SUCCESS_SHA256,
    audit_qwen_records,
)
from study2_audit.schema import validate_load_record

ROOT = Path(__file__).resolve().parents[2]


def test_both_files_match_the_pinned_bytes_and_are_not_rewritten():
    success = ROOT / "docs/checkpoint-1.6/qwen_feasibility.json"
    preflight = ROOT / "docs/checkpoint-1.6/qwen_feasibility_preflight.json"
    before = success.read_bytes()
    report = audit_qwen_records(ROOT)
    after = success.read_bytes()
    assert before == after
    assert hashlib.sha256(before).hexdigest() == QWEN_SUCCESS_SHA256
    assert hashlib.sha256(preflight.read_bytes()).hexdigest() == QWEN_PREFLIGHT_SHA256
    assert report["bytes_unchanged"] is True
    assert report["rewritten"] is False
    assert report["successful_load"]["derived_reading_replaces_file"] is False
    assert report["preflight"]["derived_reading_replaces_file"] is False


def test_the_successful_file_has_a_true_summary_and_false_nested_flags():
    report = audit_qwen_records(ROOT)["successful_load"]
    assert report["top_level_from_pretrained_called"] is True
    assert report["from_pretrained_call_count"] == 2
    nested_false = [item["path"] for item in report["flags"] if item["value"] is False]
    assert "$.gates.weight_load_and_memory.from_pretrained_called" in nested_false
    assert "$.qwen_candidate.from_pretrained_called" in nested_false
    assert "$.qwen_dtype_decisions.bfloat16.from_pretrained_called" in nested_false
    assert "$.model_compatibility.from_pretrained_called" in nested_false
    kinds = [item["kind"] for item in report["contradictions"]]
    assert "load_gate_false_while_status_loaded" in kinds
    assert "nested_false_beside_document_true" in kinds
    assert report["stale_candidate_reason"]["load_permitted_by_weight_guard"] is False
    derived = validate_load_record(report["derived_reading"])
    load = next(phase for phase in derived["phases"] if phase["phase_kind"] == "load")
    preflight = next(phase for phase in derived["phases"] if phase["phase_kind"] == "preflight")
    assert load["status"] == "loaded"
    assert load["from_pretrained_called"] is True
    assert load["call_count"] == 2
    cache = report["weight_cache"]
    assert cache["blobs_unchanged"] is True
    assert cache["weights_downloaded"] is False
    assert cache["local_weight_file_present"] is True
    assert cache["from_pretrained_call_count"] == 2
    assert preflight["from_pretrained_called"] is False


def test_the_preflight_file_agrees_with_itself():
    report = audit_qwen_records(ROOT)
    preflight = report["preflight"]
    assert preflight["top_level_from_pretrained_called"] is False
    assert preflight["from_pretrained_call_count"] == 0
    assert preflight["contradictions"] == []
    assert preflight["stale_candidate_reason"] is None
    assert report["same_revision"] is True
    derived = validate_load_record(preflight["derived_reading"])
    load = next(phase for phase in derived["phases"] if phase["phase_kind"] == "load")
    assert load["status"] == "aborted"
    assert load["from_pretrained_called"] is False
