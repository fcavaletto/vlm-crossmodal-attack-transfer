"""Versioned load record.

Checkpoint 1.6 stored ``from_pretrained_called`` once at the top of the Qwen
JSON and again inside the survey, the dtype decision, and the load gate.
Those copies answer different questions. A later reader cannot tell them
apart. This schema gives the preflight and the load attempt their own
objects, so a false preflight stamp can sit in the same document as a
successful load without becoming a second answer to the load question.

The schema describes records this package proposes. It does not rewrite
``docs/checkpoint-1.6/qwen_feasibility.json``.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA_VERSION = "study2_audit.load_record.v1"
PHASE_KINDS = frozenset({"preflight", "load"})
RECORD_ROLES = frozenset({"proposal", "derived_reading"})
LOAD_STATUSES = frozenset({"not_started", "aborted", "failed", "loaded"})


class SchemaError(ValueError):
    """The document is not a versioned load record."""


def validate_load_record(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized copy, or raise ``SchemaError``.

    ``from_pretrained_called`` is legal only inside a phase. A preflight
    phase has call count 0. A load phase with status ``loaded`` has a
    positive call count. The two phases may share a document.
    """

    if not isinstance(document, Mapping):
        raise SchemaError("a load record is a JSON object")
    _reject_flag_outside_phases(document)
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError(f"schema_version must be {SCHEMA_VERSION!r}")
    role = document.get("record_role")
    if role not in RECORD_ROLES:
        raise SchemaError("record_role must be proposal or derived_reading")
    if document.get("replaces_official_measurement") is not False:
        raise SchemaError("a load record does not replace the official measurement")
    phases = document.get("phases")
    if not isinstance(phases, list) or not phases:
        raise SchemaError("phases must be a non-empty list")
    normalized_phases = [_validate_phase(phase, index) for index, phase in enumerate(phases)]
    kinds = [phase["phase_kind"] for phase in normalized_phases]
    if kinds.count("preflight") != 1:
        raise SchemaError("a load record has one preflight phase")
    if kinds.count("load") > 1:
        raise SchemaError("a load record has at most one load phase")
    source_sha256 = document.get("source_sha256")
    if role == "derived_reading":
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise SchemaError("a derived reading names the sha256 of the file it read")
        if not isinstance(document.get("source_path"), str) or not document["source_path"]:
            raise SchemaError("a derived reading names the file it read")
    elif source_sha256 is not None:
        raise SchemaError("a proposal has no source file")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": role,
        "replaces_official_measurement": False,
        "source_sha256": source_sha256,
        "source_path": document.get("source_path"),
        "phases": normalized_phases,
    }


def _reject_flag_outside_phases(document: Mapping[str, Any]) -> None:
    for key in (
        "from_pretrained_called",
        "model_from_pretrained_called",
        "processor_from_pretrained_called",
        "from_pretrained_call_count",
    ):
        if key in document:
            raise SchemaError(f"{key} belongs inside a phase, not on the document")


def _validate_phase(phase: Any, index: int) -> dict[str, Any]:
    if not isinstance(phase, Mapping):
        raise SchemaError(f"phase {index} is not an object")
    phase_id = phase.get("phase_id")
    kind = phase.get("phase_kind")
    if not isinstance(phase_id, str) or not phase_id:
        raise SchemaError(f"phase {index} needs a phase_id")
    if kind not in PHASE_KINDS:
        raise SchemaError(f"phase {phase_id!r} has an unknown phase_kind")
    called = phase.get("from_pretrained_called")
    count = phase.get("call_count")
    if not isinstance(called, bool):
        raise SchemaError(f"phase {phase_id!r} needs a boolean from_pretrained_called")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SchemaError(f"phase {phase_id!r} needs a call_count of zero or more")
    if called != (count > 0):
        raise SchemaError(
            f"phase {phase_id!r} has from_pretrained_called {called} and call_count {count}"
        )
    status = phase.get("status")
    if kind == "preflight":
        if called or count != 0:
            raise SchemaError("the preflight phase does not call from_pretrained")
        if status is not None and status not in {"measured", "budget_cleared", "budget_refused"}:
            raise SchemaError(f"preflight status {status!r} is not a preflight status")
        if "model_from_pretrained_called" in phase or "processor_from_pretrained_called" in phase:
            raise SchemaError("processor and model call flags belong on the load phase")
    else:
        if status not in LOAD_STATUSES:
            raise SchemaError(f"load phase {phase_id!r} needs a load status")
        if status == "loaded" and not called:
            raise SchemaError("status loaded requires from_pretrained_called true")
        if status in {"not_started", "aborted"} and called:
            raise SchemaError(f"status {status} has call_count 0")
        _check_party_flags(phase, count)
    note = phase.get("note")
    if note is not None and not isinstance(note, str):
        raise SchemaError(f"phase {phase_id!r} note must be a string")
    normalized = {
        "phase_id": phase_id,
        "phase_kind": kind,
        "from_pretrained_called": called,
        "call_count": count,
        "status": status,
        "note": note,
    }
    if kind == "load":
        for key in ("model_from_pretrained_called", "processor_from_pretrained_called", "revision"):
            if key in phase:
                normalized[key] = phase[key]
    return normalized


def _check_party_flags(phase: Mapping[str, Any], count: int) -> None:
    processor = phase.get("processor_from_pretrained_called")
    model = phase.get("model_from_pretrained_called")
    if processor is not None and processor is not (count >= 1):
        raise SchemaError("processor_from_pretrained_called disagrees with call_count")
    if model is not None and model is not (count >= 2):
        raise SchemaError("model_from_pretrained_called disagrees with call_count")
    revision = phase.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise SchemaError("revision must be a non-empty string")
