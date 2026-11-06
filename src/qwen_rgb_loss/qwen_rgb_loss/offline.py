"""Checks that do not load Qwen. They read the locked protocol and the 1.7 rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from choice_eval.models import require_model
from dev_pilot.fixtures import load_manifest, verify_checksums
from dev_pilot.rows import protocol_errors

from .protocol import MODEL_KEY, PREDICTION_RELATIVE, PRIMARY_CONTEXTS, assert_unchanged_lines


def load_predictions(root: Path) -> list[dict[str, Any]]:
    path = root / PREDICTION_RELATIVE
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("model_key") != MODEL_KEY for row in rows):
        raise RuntimeError("a stored prediction is not the Qwen development row")
    return rows


def verify_locked_inputs(root: Path) -> dict[str, Any]:
    """Manifest, checksums, revision, token ids, and the stored decision rule.

    This does not rewrite the Checkpoint 1.7 predictions.
    """
    assert_unchanged_lines()
    checksums = verify_checksums(root)
    records = load_manifest(root)
    rows = load_predictions(root)
    by_id = {record["scene_id"]: record for record in records}
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("scene_id")), str(row.get("context")))
        if key in seen:
            errors.append(f"duplicate {key}")
        seen.add(key)
        record = by_id.get(key[0])
        if record is None:
            errors.append(f"unknown scene {key[0]}")
            continue
        if key[1] in PRIMARY_CONTEXTS or key[1] in {"contradictory", "answer_revealing"}:
            errors.extend(f"{key[0]} {key[1]} {item}" for item in protocol_errors(row, record))
    primary = [row for row in rows if row.get("context") in PRIMARY_CONTEXTS]
    expected = {(record["scene_id"], context) for record in records for context in PRIMARY_CONTEXTS}
    found = {(row["scene_id"], row["context"]) for row in primary}
    if found != expected:
        errors.append("primary prediction coverage does not match the 96 scenes")
    pin = require_model(MODEL_KEY)
    if errors:
        preview = "; ".join(errors[:8])
        raise RuntimeError(f"{len(errors)} locked-input errors: {preview}")
    return {
        "manifest_rows": len(records),
        "checksum_files": checksums["files"],
        "prediction_rows": len(rows),
        "primary_rows": len(primary),
        "revision": pin.revision,
        "model_id": pin.model_id,
        "predictions_rewritten": False,
        "decision_rule": "lowest summed NLL, then color name",
    }
