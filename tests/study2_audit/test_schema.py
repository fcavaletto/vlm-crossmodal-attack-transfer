"""The versioned load record keeps the preflight and the load in separate phases."""

from __future__ import annotations

import pytest

from study2_audit.schema import SCHEMA_VERSION, SchemaError, validate_load_record


def _proposal() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "proposal",
        "replaces_official_measurement": False,
        "phases": [
            {
                "phase_id": "preflight",
                "phase_kind": "preflight",
                "from_pretrained_called": False,
                "call_count": 0,
                "status": "budget_cleared",
            },
            {
                "phase_id": "weight_load",
                "phase_kind": "load",
                "from_pretrained_called": True,
                "call_count": 2,
                "status": "loaded",
                "processor_from_pretrained_called": True,
                "model_from_pretrained_called": True,
                "revision": "abc",
            },
        ],
    }


def test_a_preflight_and_a_successful_load_share_one_document():
    document = validate_load_record(_proposal())
    kinds = [phase["phase_kind"] for phase in document["phases"]]
    assert kinds == ["preflight", "load"]
    assert document["phases"][0]["from_pretrained_called"] is False
    assert document["phases"][1]["from_pretrained_called"] is True
    assert document["replaces_official_measurement"] is False


def test_the_document_summary_flag_is_rejected():
    document = _proposal()
    document["from_pretrained_called"] = True
    with pytest.raises(SchemaError, match="belongs inside a phase"):
        validate_load_record(document)


def test_loaded_status_cannot_carry_a_false_flag():
    document = _proposal()
    document["phases"][1]["from_pretrained_called"] = False
    document["phases"][1]["call_count"] = 0
    with pytest.raises(SchemaError, match="status loaded"):
        validate_load_record(document)


def test_a_preflight_phase_cannot_record_a_call():
    document = _proposal()
    document["phases"][0]["from_pretrained_called"] = True
    document["phases"][0]["call_count"] = 1
    with pytest.raises(SchemaError, match="preflight"):
        validate_load_record(document)


def test_call_count_and_the_boolean_must_agree():
    document = _proposal()
    document["phases"][1]["call_count"] = 1
    with pytest.raises(SchemaError, match="model_from_pretrained_called"):
        validate_load_record(document)


def test_a_proposal_does_not_name_a_source_file():
    document = _proposal()
    document["source_sha256"] = "a" * 64
    with pytest.raises(SchemaError, match="proposal"):
        validate_load_record(document)
