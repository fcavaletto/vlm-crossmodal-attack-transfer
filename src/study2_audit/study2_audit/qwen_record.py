"""Read the Checkpoint 1.6 Qwen JSON without rewriting it.

The successful measurement has ``from_pretrained_called`` true at the top
and false on nested objects that were filled in before the load and never
updated. The preflight file is a separate measurement. Both stay on disk
as they were committed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import SCHEMA_VERSION, validate_load_record

# Byte hashes of the committed measurements at commit 004783b.
# A rewrite of either file fails the audit.
QWEN_SUCCESS_SHA256 = "055a133be14cf7f76c20bb59880cf8f4ca0a66321dea71960a78e8ca97b21dd7"
QWEN_PREFLIGHT_SHA256 = "c2d8adc0a830fd58f65e553c69f84889cf37888e6381575d2a50d52976bf999c"
QWEN_SUCCESS_RELATIVE = "docs/checkpoint-1.6/qwen_feasibility.json"
QWEN_PREFLIGHT_RELATIVE = "docs/checkpoint-1.6/qwen_feasibility_preflight.json"
PINNED_QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

FLAG_KEY = "from_pretrained_called"


def audit_qwen_records(root: Path) -> dict[str, Any]:
    """Hash both Qwen files, list every flag, and build derived readings.

    The derived readings are schema objects in memory. They are not written
    back over the measurements.
    """

    success_path = root / QWEN_SUCCESS_RELATIVE
    preflight_path = root / QWEN_PREFLIGHT_RELATIVE
    success_bytes = success_path.read_bytes()
    preflight_bytes = preflight_path.read_bytes()
    success_sha = hashlib.sha256(success_bytes).hexdigest()
    preflight_sha = hashlib.sha256(preflight_bytes).hexdigest()
    success = json.loads(success_bytes)
    preflight = json.loads(preflight_bytes)
    success_reading = _read_legacy(success, success_sha, QWEN_SUCCESS_RELATIVE, "successful_load")
    preflight_reading = _read_legacy(preflight, preflight_sha, QWEN_PREFLIGHT_RELATIVE, "preflight")
    return {
        "rewritten": False,
        "bytes_unchanged": success_sha == QWEN_SUCCESS_SHA256 and preflight_sha == QWEN_PREFLIGHT_SHA256,
        "successful_load": success_reading,
        "preflight": preflight_reading,
        "same_revision": (
            success["qwen_candidate"]["revision"] == preflight["qwen_candidate"]["revision"] == PINNED_QWEN_REVISION
        ),
        "finding": _finding(success_reading, preflight_reading),
    }


def iter_flag_paths(document: Any, prefix: str = "$"):
    """Yield ``(json_pointer, value)`` for every ``from_pretrained_called``."""

    if isinstance(document, dict):
        for key, value in document.items():
            path = f"{prefix}.{key}"
            if key == FLAG_KEY:
                yield path, value
            yield from iter_flag_paths(value, path)
    elif isinstance(document, list):
        for index, value in enumerate(document):
            yield from iter_flag_paths(value, f"{prefix}[{index}]")


def _read_legacy(
    payload: Mapping[str, Any],
    sha256: str,
    relative_path: str,
    measurement: str,
) -> dict[str, Any]:
    flags = [
        {"path": path, "value": value, "role": _role(path)}
        for path, value in iter_flag_paths(payload)
    ]
    top = payload.get(FLAG_KEY)
    contradictions = _contradictions(payload, flags)
    derived = _derived_reading(payload, sha256, relative_path)
    validated = validate_load_record(derived)
    return {
        "measurement": measurement,
        "path": relative_path,
        "sha256": sha256,
        "matches_pinned_sha256": sha256 == (
            QWEN_SUCCESS_SHA256 if measurement == "successful_load" else QWEN_PREFLIGHT_SHA256
        ),
        "legacy_shape": True,
        "schema_version_of_file": None,
        "top_level_from_pretrained_called": top,
        "from_pretrained_call_count": payload.get("from_pretrained_call_count"),
        "model_from_pretrained_called": payload.get("model_from_pretrained_called"),
        "processor_from_pretrained_called": payload.get("processor_from_pretrained_called"),
        "flags": flags,
        "contradictions": contradictions,
        "stale_candidate_reason": _stale_candidate(payload),
        "weight_cache": _weight_cache(payload),
        "derived_reading": validated,
        "derived_reading_replaces_file": False,
    }


def _role(path: str) -> str:
    if path == "$.from_pretrained_called":
        return "document_summary"
    if path == "$.gates.weight_load_and_memory.from_pretrained_called":
        return "load_gate"
    if path.startswith("$.gates."):
        return "gate_stamp"
    if path.startswith(("$.model_compatibility.", "$.qwen_candidate.", "$.qwen_dtype_decisions.")):
        return "preflight_stamp"
    return "other"


def _contradictions(payload: Mapping[str, Any], flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    top = payload.get(FLAG_KEY)
    count = payload.get("from_pretrained_call_count")
    gate = payload.get("gates", {}).get("weight_load_and_memory", {})
    if top is True and gate.get("status") == "loaded" and gate.get(FLAG_KEY) is False:
        found.append(
            {
                "kind": "load_gate_false_while_status_loaded",
                "path": "$.gates.weight_load_and_memory.from_pretrained_called",
                "gate_status": "loaded",
                "gate_flag": False,
                "document_summary": True,
                "detail": (
                    "The load gate was initialized with from_pretrained_called false "
                    "and, after the weights loaded, only its status changed to loaded."
                ),
            }
        )
    if top is True and count == 0:
        found.append(
            {
                "kind": "document_true_with_zero_calls",
                "detail": "The document summary is true and the call count is zero.",
            }
        )
    nested_false = [
        item["path"]
        for item in flags
        if item["path"] != "$.from_pretrained_called" and item["value"] is False and top is True
    ]
    if nested_false:
        found.append(
            {
                "kind": "nested_false_beside_document_true",
                "paths": nested_false,
                "detail": (
                    "The document summary is true. Nested fields still say false. "
                    "Preflight stamps describe steps that did not call from_pretrained. "
                    "The load-gate false is a stale copy of the same word inside the load object."
                ),
            }
        )
    if top is False:
        nested_true = [
            item["path"]
            for item in flags
            if item["path"] != "$.from_pretrained_called" and item["value"] is True
        ]
        if nested_true:
            found.append(
                {
                    "kind": "nested_true_beside_document_false",
                    "paths": nested_true,
                    "detail": "The document summary is false and a nested field says true.",
                }
            )
    return found


def _stale_candidate(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    candidate = payload.get("qwen_candidate") or {}
    reason = candidate.get("reason")
    top = payload.get(FLAG_KEY) is True
    permitted = (payload.get("dtype_selection") or {}).get("permitted") is True
    guard = candidate.get("load_permitted_by_weight_guard")
    if not (top and permitted and isinstance(reason, str) and "must not be called" in reason):
        return None
    return {
        "path": "$.qwen_candidate.reason",
        "reason": reason,
        "load_permitted_by_weight_guard": guard,
        "dtype_selection_permitted": True,
        "detail": (
            "The candidate object still carries the float32 weight-guard sentence. "
            "The later BF16 selection permitted the load. The candidate object was not updated."
        ),
    }


def _weight_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    before = payload.get("large_blobs_before")
    after = payload.get("large_blobs_after")
    return {
        "blobs_unchanged": before == after,
        "weights_downloaded": payload.get("weights_downloaded"),
        "local_weight_file_present": payload.get("qwen_local_weight_file_present"),
        "from_pretrained_call_count": payload.get("from_pretrained_call_count"),
    }


def _derived_reading(payload: Mapping[str, Any], sha256: str, relative_path: str) -> dict[str, Any]:
    """Propose the phase split the legacy document does not have.

    Call counts come from the document summary. The stale load-gate flag
    is recorded as a contradiction and is not copied into the load phase.
    """

    count = int(payload.get("from_pretrained_call_count") or 0)
    loaded = count > 0 and payload.get("gates", {}).get("weight_load_and_memory", {}).get("status") == "loaded"
    if payload.get(FLAG_KEY) is True and not loaded:
        load_status = "failed" if count else "not_started"
    elif loaded:
        load_status = "loaded"
    elif payload.get("stopped_at_gate") == "gate_2_preload_budget":
        load_status = "aborted"
    else:
        load_status = "not_started"
    preflight_status = "budget_cleared" if (payload.get("dtype_selection") or {}).get("permitted") else "budget_refused"
    revision = (payload.get("qwen_candidate") or {}).get("revision")
    load_phase: dict[str, Any] = {
        "phase_id": "weight_load",
        "phase_kind": "load",
        "from_pretrained_called": count > 0,
        "call_count": count,
        "status": load_status,
        "revision": revision,
        "note": (
            "Call count and the two party flags are the document summary. "
            "They are not the nested load-gate boolean."
        ),
    }
    if count > 0:
        load_phase["processor_from_pretrained_called"] = bool(payload.get("processor_from_pretrained_called"))
        load_phase["model_from_pretrained_called"] = bool(payload.get("model_from_pretrained_called"))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_role": "derived_reading",
        "replaces_official_measurement": False,
        "source_sha256": sha256,
        "source_path": relative_path,
        "phases": [
            {
                "phase_id": "preflight",
                "phase_kind": "preflight",
                "from_pretrained_called": False,
                "call_count": 0,
                "status": preflight_status,
                "note": (
                    "Class resolution, the safetensors survey, and the dtype decision "
                    "do not call from_pretrained. A false flag here is the preflight result."
                ),
            },
            load_phase,
        ],
    }


def _finding(success: Mapping[str, Any], preflight: Mapping[str, Any]) -> str:
    success_kinds = [item["kind"] for item in success["contradictions"]]
    if "load_gate_false_while_status_loaded" in success_kinds and preflight["top_level_from_pretrained_called"] is False:
        return (
            "The preflight file says from_pretrained was not called, and its nested flags agree. "
            "The successful-load file says the top-level call happened twice. "
            "Its nested flags still say false, including the load gate whose status is loaded."
        )
    return "The two Qwen files do not match the expected preflight-versus-load split."
