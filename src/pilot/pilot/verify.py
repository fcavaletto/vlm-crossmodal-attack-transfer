"""Rebuild ``summary.json`` from the compact pilot files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.pilot_metrics import build_pilot_summary
from analysis.pilot_schema import validate_pilot_bundle
from analysis.schemas import load_jsonl
from analysis.verify_toy import compare_summaries

from .constants import IMAGE_HEIGHT, IMAGE_WIDTH, N_GROUPS, N_SCENES

VERIFICATION_COMMAND = "python -m pilot.verify"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    names = {
        "predictions": directory / "predictions.jsonl",
        "gradients": directory / "gradient_stats.jsonl",
        "alignments": directory / "alignments.jsonl",
        "run_meta": directory / "run_meta.json",
        "gate": directory / "gate.json",
        "summary": directory / "summary.json",
    }
    missing = [str(path) for path in names.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing pilot files: " + ", ".join(missing))
    predictions = [row for _, row in load_jsonl(names["predictions"])]
    gradients = [row for _, row in load_jsonl(names["gradients"])]
    alignments = [row for _, row in load_jsonl(names["alignments"])]
    run_meta = json.loads(names["run_meta"].read_text(encoding="utf-8"))
    gate = json.loads(names["gate"].read_text(encoding="utf-8"))
    published = json.loads(names["summary"].read_text(encoding="utf-8"))
    return {
        "paths": names,
        "predictions": predictions,
        "gradients": gradients,
        "alignments": alignments,
        "run_meta": run_meta,
        "gate": gate,
        "published": published,
    }


def assemble_summary(
    predictions: list[dict[str, Any]],
    gradients: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    run_meta: dict[str, Any],
    gate: dict[str, Any],
    input_sha256: dict[str, str],
) -> dict[str, Any]:
    bundle = validate_pilot_bundle(predictions, gradients, alignments)
    summary = build_pilot_summary(bundle["predictions"], bundle["gradients"], bundle["alignments"])
    summary["run_meta"] = run_meta
    summary["gate_passed"] = gate.get("passed")
    summary["gate_scene_id"] = gate.get("scene_id")
    summary["input_sha256"] = input_sha256
    return summary


def official_input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "predictions.jsonl": file_sha256(paths["predictions"]),
        "gradient_stats.jsonl": file_sha256(paths["gradients"]),
        "alignments.jsonl": file_sha256(paths["alignments"]),
        "run_meta.json": file_sha256(paths["run_meta"]),
        "gate.json": file_sha256(paths["gate"]),
    }


def development_pilot_problems(summary: dict[str, Any], gate: dict[str, Any], gradients: list[dict[str, Any]]) -> list[str]:
    """Constraints that make this directory the Checkpoint 1 MPS pilot."""
    problems = []
    if summary.get("label") != "DEVELOPMENT ONLY":
        problems.append("summary label is not DEVELOPMENT ONLY")
    if summary.get("attacks_run") is not False:
        problems.append("attacks_run is not false")
    if summary.get("h_star_fitted") is not False:
        problems.append("h_star_fitted is not false")
    if summary.get("n_scenes") != N_SCENES or summary.get("n_groups") != N_GROUPS:
        problems.append("scene or group count is not the committed development sample")
    if summary.get("n_predictions") != N_SCENES * 4:
        problems.append("prediction count is not 24 scenes by 4 contexts")
    if gate.get("passed") is not True or gate.get("device") != "mps":
        problems.append("the full-resolution gate did not pass on MPS")
    if summary.get("gate_passed") is not True:
        problems.append("summary does not record a passed gate")
    for row in gradients:
        if row.get("height") != IMAGE_HEIGHT or row.get("width") != IMAGE_WIDTH:
            problems.append(f"{row.get('scene_id')}: gradient is not 512 by 384")
            break
        if row.get("device") != "mps":
            problems.append("a gradient row is not MPS")
            break
    return problems


def verify_directory(directory: Path) -> dict[str, Any]:
    loaded = load_official(directory)
    hashes = official_input_hashes(loaded["paths"])
    rebuilt = assemble_summary(
        loaded["predictions"],
        loaded["gradients"],
        loaded["alignments"],
        loaded["run_meta"],
        loaded["gate"],
        hashes,
    )
    mismatches = compare_summaries(rebuilt, loaded["published"])
    mismatches.extend(
        development_pilot_problems(rebuilt, loaded["gate"], loaded["gradients"])
    )
    return {
        "ok": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "label": "DEVELOPMENT ONLY",
        "verification_command": VERIFICATION_COMMAND,
    }


def main(argv: list[str] | None = None) -> int:
    from .manifest import repo_root

    parser = argparse.ArgumentParser(description="Rebuild the Checkpoint 1 pilot summary.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Directory containing the compact pilot files. Defaults to docs/checkpoint-1.",
    )
    args = parser.parse_args(argv)
    directory = args.directory if args.directory is not None else repo_root() / "docs" / "checkpoint-1"
    result = verify_directory(directory)
    print(json.dumps({key: result[key] for key in ("ok", "n_mismatches", "label")}, indent=2))
    for mismatch in result["mismatches"]:
        print(mismatch)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
