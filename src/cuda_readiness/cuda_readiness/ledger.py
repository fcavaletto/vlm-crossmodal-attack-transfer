"""Checkpoint and resume bookkeeping.

A saved budget or out-of-memory refusal is terminal for that process. This
module does not load a model, does not lower a memory line, and does not
score an image. A later process may keep the finished units and take a new
reading; the new ledger still has ``model_from_pretrained_called`` false.
"""

from __future__ import annotations

import copy
from typing import Any

from .hashes import sha256_bytes

TERMINAL_REFUSAL_STATUSES = frozenset({"stopped_budget", "stopped_oom"})


class RefusalRetryError(RuntimeError):
    """The same process tried to continue after a saved refusal."""


def empty_ledger(process_id: str) -> dict[str, Any]:
    if not process_id:
        raise ValueError("process_id is required")
    return {
        "schema": "cuda_readiness.ledger.v1",
        "process_id": process_id,
        "status": "prepared",
        "attacks_run": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "cuda_numerical_validation": "not_executed",
        "guard_lowered": False,
        "finished_units": {},
        "result_hashes": {},
        "refusals": [],
        "prior_refusal_kept": False,
    }


def _assert_same_process_is_open(ledger: dict[str, Any]) -> None:
    process_id = ledger["process_id"]
    if any(item["process_id"] == process_id for item in ledger["refusals"]):
        raise RefusalRetryError(
            "A saved refusal is not retried in the same process. The model is not loaded."
        )


def record_refusal(
    ledger: dict[str, Any],
    *,
    status: str,
    available_bytes: int,
    reason: str,
) -> dict[str, Any]:
    """Record one terminal refusal and the byte count that entered it."""
    if status not in TERMINAL_REFUSAL_STATUSES:
        raise ValueError(f"refusal status must be one of {sorted(TERMINAL_REFUSAL_STATUSES)}")
    if isinstance(available_bytes, bool) or not isinstance(available_bytes, int):
        raise TypeError("available_bytes must be an int")
    if available_bytes < 0:
        raise ValueError("available_bytes must be non-negative")
    if not reason:
        raise ValueError("reason is required")
    _assert_same_process_is_open(ledger)
    ledger["refusals"].append(
        {
            "status": status,
            "available_bytes": available_bytes,
            "reason": reason,
            "process_id": ledger["process_id"],
            "guard_lowered": False,
            "model_from_pretrained_called": False,
        }
    )
    ledger["status"] = status
    ledger["guard_lowered"] = False
    ledger["model_from_pretrained_called"] = False
    ledger["attacks_run"] = False
    return ledger


def record_unit(ledger: dict[str, Any], unit_id: str, payload: bytes) -> dict[str, Any]:
    """Store the SHA-256 of one finished unit. A second write does not replace it."""
    if not unit_id:
        raise ValueError("unit_id is required")
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    _assert_same_process_is_open(ledger)
    finished = ledger["finished_units"]
    digest = sha256_bytes(payload)
    if unit_id in finished:
        return {"unit_id": unit_id, "sha256": finished[unit_id]["sha256"], "skipped": True}
    finished[unit_id] = {"sha256": digest, "process_id": ledger["process_id"]}
    return {"unit_id": unit_id, "sha256": digest, "skipped": False}


def pending_units(ledger: dict[str, Any], unit_ids: list[str]) -> list[str]:
    """Unit ids that do not yet have a stored hash, in the caller's order."""
    finished = ledger["finished_units"]
    return [unit_id for unit_id in unit_ids if unit_id not in finished]


def attach_result_hashes(ledger: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """Copy result-file digests onto the ledger. Existing digests are not replaced."""
    _assert_same_process_is_open(ledger)
    stored = ledger["result_hashes"]
    for relative, digest in hashes.items():
        if relative not in stored:
            stored[relative] = digest
    return ledger


def continue_in_new_process(ledger: dict[str, Any], process_id: str) -> dict[str, Any]:
    """Start bookkeeping in another process without dropping finished units.

    The returned ledger does not authorize a model load. If ``process_id`` is
    the process that already saved a refusal, this raises.
    """
    if not process_id:
        raise ValueError("process_id is required")
    if process_id == ledger["process_id"] and any(
        item["process_id"] == process_id for item in ledger["refusals"]
    ):
        raise RefusalRetryError(
            "A saved refusal is not retried in the same process. The model is not loaded."
        )
    kept_refusal = ledger["status"] in TERMINAL_REFUSAL_STATUSES or bool(ledger["refusals"])
    return {
        "schema": ledger["schema"],
        "process_id": process_id,
        "status": "prepared",
        "attacks_run": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "cuda_numerical_validation": "not_executed",
        "guard_lowered": False,
        "finished_units": copy.deepcopy(ledger["finished_units"]),
        "result_hashes": dict(ledger["result_hashes"]),
        "refusals": copy.deepcopy(ledger["refusals"]),
        "prior_refusal_kept": kept_refusal,
        "resumed_from_process_id": ledger["process_id"],
    }
