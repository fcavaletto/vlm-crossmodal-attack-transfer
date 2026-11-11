"""Check development_v4 predictions against the locked protocol. No model load."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .fixtures import load_manifest, verify_checksums
from .lock import EXTRA_CONTEXTS, N_SCENES, OUTPUT_RELATIVE, PRIMARY_CONTEXTS
from .provenance import protocol_provenance
from .rows import protocol_errors
from .run import repo_root
from .summarize import summarize_model

TERMINAL_STOPS = {"stopped_budget", "stopped_memory", "stopped_missing_weights", "out_of_memory"}
MODEL_KEYS = ("smolvlm2", "qwen3-vl-2b")
MAX_PREDICTION_BYTES = 5_000_000


def main(argv: list[str] | None = None) -> int:
    del argv
    root = repo_root()
    errors = verify_tree(root)
    if errors:
        for error in errors:
            print(error, flush=True)
        return 1
    print(json.dumps({"status": "ok", "label": "DEVELOPMENT ONLY"}, sort_keys=True), flush=True)
    return 0


def verify_tree(root: Path) -> list[str]:
    """Return a list of problems. An empty list is a consistent development record."""
    errors: list[str] = []
    try:
        provenance = protocol_provenance(root)
    except RuntimeError as exc:
        return [str(exc)]
    if provenance["predictions_present_at_protocol_commit"] is not False:
        errors.append("predictions exist at the protocol commit")
    if provenance["protocol_commit_is_ancestor"] is not True:
        errors.append("protocol commit is not an ancestor")
    if provenance["checkpoint_1_6_diff_empty"] is not True:
        errors.append("docs/checkpoint-1.6 changed")
    try:
        verify_checksums(root)
        records = load_manifest(root)
    except RuntimeError as exc:
        return errors + [str(exc)]
    errors.extend(_guard_errors(root))
    for model_key in MODEL_KEYS:
        errors.extend(_model_errors(root, records, model_key, provenance))
    return errors


def _model_errors(root: Path, records, model_key: str, provenance: Mapping[str, Any]) -> list[str]:
    directory = root / OUTPUT_RELATIVE / model_key
    meta_path = directory / "run_meta.json"
    if not meta_path.is_file():
        return [f"{model_key}: run_meta.json is missing"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if meta.get("label") != "DEVELOPMENT ONLY":
        errors.append(f"{model_key}: meta label")
    if meta.get("attacks_run") is not False or meta.get("h_star_fitted") is not False:
        errors.append(f"{model_key}: attack or H* flag")
    if meta.get("perturbations_run") is not False or meta.get("gradients_measured_on_development_v4") is not False:
        errors.append(f"{model_key}: perturbation or v4 gradient flag")
    if meta.get("protocol_commit") != provenance["protocol_commit"]:
        errors.append(f"{model_key}: protocol commit does not match the file history")
    if meta.get("predictions_present_at_protocol_commit") is not False:
        errors.append(f"{model_key}: meta says predictions existed at the protocol commit")
    if meta.get("manifest_sha256") != provenance["manifest_sha256"]:
        errors.append(f"{model_key}: manifest hash")
    status = meta.get("status")
    prediction_path = directory / "predictions.jsonl"
    if status == "out_of_memory":
        if not isinstance(meta.get("available_bytes"), int) and "available_bytes" not in meta:
            errors.append(f"{model_key}: out of memory was recorded without a byte count")
        return errors
    if status in TERMINAL_STOPS:
        if meta.get("model_from_pretrained_called") is not False:
            errors.append(f"{model_key}: a budget stop must not have called from_pretrained")
        if status in {"stopped_budget", "stopped_memory"} and not isinstance(meta.get("available_bytes"), int):
            errors.append(f"{model_key}: available_bytes was not recorded")
        if prediction_path.is_file() and prediction_path.stat().st_size > 0:
            errors.append(f"{model_key}: predictions were written after a stop")
        if meta.get("n_groups_eligible") not in (None,):
            errors.append(f"{model_key}: eligible count must stay unset")
        return errors
    if status != "completed":
        return errors + [f"{model_key}: status {status!r} is not a finished record"]
    if not prediction_path.is_file():
        return errors + [f"{model_key}: predictions.jsonl is missing"]
    if prediction_path.stat().st_size > MAX_PREDICTION_BYTES:
        errors.append(f"{model_key}: predictions file is larger than the compact-row cap")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["scene_id"]: record for record in records}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("scene_id"), row.get("context"))
        if key in seen:
            errors.append(f"{model_key}: duplicate {key}")
        seen.add(key)
        record = by_id.get(row.get("scene_id"))
        if record is None:
            errors.append(f"{model_key}: unknown scene {row.get('scene_id')}")
            continue
        for problem in protocol_errors(row, record):
            errors.append(f"{model_key}: {key} {problem}")
    for context in PRIMARY_CONTEXTS:
        count = sum(row["context"] == context for row in rows)
        if count != N_SCENES:
            errors.append(f"{model_key}: {context} has {count} rows")
    extra_done = all(sum(row["context"] == context for row in rows) == N_SCENES for context in EXTRA_CONTEXTS)
    if bool(meta.get("extra_contexts_scored")) != extra_done:
        errors.append(f"{model_key}: extra-context flag does not match the rows")
    if errors:
        return errors
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return errors + [f"{model_key}: summary.json is missing"]
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    fresh = summarize_model(rows, records, model_key=model_key, gradients=None)
    errors.extend(_summary_errors(model_key, stored, fresh))
    return errors


def _summary_errors(model_key: str, stored: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if stored.get("label") != "DEVELOPMENT ONLY" or stored.get("metrics_combined") is not False:
        errors.append(f"{model_key}: summary label or blended metric")
    if stored.get("attacks_run") is not False or stored.get("h_star_fitted") is not False:
        errors.append(f"{model_key}: summary attack or H* flag")
    if stored.get("eligible", {}).get("n_groups_eligible") is not None:
        errors.append(f"{model_key}: eligible count was filled without a v4 gradient pass")
    if stored.get("competence_gate", {}).get("passed") != fresh["competence_gate"]["passed"]:
        errors.append(f"{model_key}: competence gate does not match the rows")
    if stored.get("competence_gate", {}).get("contexts_above_80") != fresh["competence_gate"]["contexts_above_80"]:
        errors.append(f"{model_key}: ceiling flag does not match the rows")
    for name in ("closed_clean_both", "strict_clean_both"):
        if stored.get(name, {}).get("n_scenes") != fresh[name]["n_scenes"]:
            errors.append(f"{model_key}: {name} scene count")
        if stored.get(name, {}).get("n_groups_at_least_one_scene") != fresh[name]["n_groups_at_least_one_scene"]:
            errors.append(f"{model_key}: {name} group count")
    for context, fresh_context in fresh["contexts"].items():
        got = stored.get("contexts", {}).get(context, {})
        for field in (
            "n",
            "closed_vocabulary_correct",
            "free_generation_correct",
            "free_generation_valid_but_wrong",
            "free_generation_invalid",
        ):
            if got.get(field) != fresh_context[field]:
                errors.append(f"{model_key}: {context} {field}")
        pairs = got.get("pairs", {})
        for field, value in fresh_context["pairs"].items():
            if field == "label" or field == "metrics_combined":
                continue
            if pairs.get(field) != value:
                errors.append(f"{model_key}: {context} pairs {field}")
    return errors


def _guard_errors(root: Path) -> list[str]:
    from qwen_rgb.budget import (
        OS_FLOOR_BYTES,
        PROJECTED_REMAINING_FLOOR_BYTES,
        SPARE_REQUIRED_BYTES,
        load_budget_module,
    )
    from vlm_iface.pins import MIN_AVAILABLE_BYTES

    errors: list[str] = []
    if MIN_AVAILABLE_BYTES != 6 * 1024**3 or OS_FLOOR_BYTES != 6 * 1024**3:
        errors.append("the 6 GiB OS floor is not the Checkpoint 1.5 floor")
    if SPARE_REQUIRED_BYTES != 5 * 1024**3:
        errors.append("the 5 GiB spare line changed")
    if PROJECTED_REMAINING_FLOOR_BYTES != 2 * 1024**3:
        errors.append("the 2 GiB projection floor changed")
    try:
        load_budget_module(root)
    except RuntimeError as exc:
        errors.append(str(exc))
    return errors


if __name__ == "__main__":
    sys.exit(main())
