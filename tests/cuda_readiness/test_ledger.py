"""Ledger rules. No model and no image."""

from __future__ import annotations

import pytest

from cuda_readiness.ledger import (
    RefusalRetryError,
    continue_in_new_process,
    empty_ledger,
    pending_units,
    record_refusal,
    record_unit,
)


def test_a_saved_budget_refusal_is_not_retried_in_the_same_process():
    ledger = empty_ledger("process-a")
    record_refusal(
        ledger,
        status="stopped_budget",
        available_bytes=11_093_655_552,
        reason="OS-available memory is below the pre-load budget.",
    )
    assert ledger["status"] == "stopped_budget"
    assert ledger["refusals"][0]["available_bytes"] == 11_093_655_552
    assert ledger["guard_lowered"] is False
    assert ledger["model_from_pretrained_called"] is False
    with pytest.raises(RefusalRetryError, match="same process"):
        record_refusal(
            ledger,
            status="stopped_budget",
            available_bytes=15_000_000_000,
            reason="a second reading",
        )
    with pytest.raises(RefusalRetryError, match="same process"):
        record_unit(ledger, "scene-1", b"result")
    with pytest.raises(RefusalRetryError, match="same process"):
        continue_in_new_process(ledger, "process-a")


def test_a_new_process_keeps_finished_units_and_does_not_load_a_model():
    ledger = empty_ledger("process-a")
    stored = record_unit(ledger, "scene-1", b"result")
    again = record_unit(ledger, "scene-1", b"different")
    assert stored["skipped"] is False
    assert again["skipped"] is True
    assert again["sha256"] == stored["sha256"]
    record_refusal(
        ledger,
        status="stopped_oom",
        available_bytes=7_000_000_000,
        reason="The runtime reported out of memory.",
    )
    resumed = continue_in_new_process(ledger, "process-b")
    assert resumed["process_id"] == "process-b"
    assert resumed["prior_refusal_kept"] is True
    assert resumed["finished_units"]["scene-1"]["sha256"] == stored["sha256"]
    assert resumed["refusals"][0]["available_bytes"] == 7_000_000_000
    assert resumed["status"] == "prepared"
    assert resumed["model_from_pretrained_called"] is False
    assert resumed["attacks_run"] is False
    assert pending_units(resumed, ["scene-1", "scene-2"]) == ["scene-2"]
    record_unit(resumed, "scene-2", b"next")
    assert ledger["finished_units"].keys() == {"scene-1"}
