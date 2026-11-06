"""Rebuild the v2 summary without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.verify_toy import compare_summaries

from .protocol import (
    LABEL,
    MANIFEST_RELATIVE,
    MANIFEST_SHA256,
    OUTPUT_RELATIVE,
    PROTOCOL_RELATIVE,
    repo_root,
)
from .summarize import build_summary

FILE_NAMES = (
    "predictions.jsonl",
    "gradient_stats.jsonl",
    "alignments.jsonl",
    "run_meta.json",
    "gate.json",
    "summary.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_directory(directory: Path, root: Path | None = None) -> dict[str, Any]:
    """Return whether ``directory`` rebuilds its published summary."""
    root = repo_root() if root is None else root
    directory = Path(directory)
    mismatches: list[str] = []
    paths = {name: directory / name for name in FILE_NAMES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {
            "ok": False,
            "n_mismatches": len(missing),
            "mismatches": [f"missing {path}" for path in missing],
            "label": LABEL,
        }
    manifest = root / MANIFEST_RELATIVE
    if file_sha256(manifest) != MANIFEST_SHA256:
        mismatches.append("fixtures/development_v2/manifest.jsonl sha256 does not match the protocol")
    protocol = root / PROTOCOL_RELATIVE
    run_meta = json.loads(paths["run_meta.json"].read_text(encoding="utf-8"))
    gate = json.loads(paths["gate.json"].read_text(encoding="utf-8"))
    published = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    if file_sha256(protocol) != run_meta.get("protocol_sha256"):
        mismatches.append("VALIDATION_PROTOCOL.md changed after the run that wrote run_meta.json")
    if run_meta.get("attacks_run") is not False or published.get("attacks_run") is not False:
        mismatches.append("attacks_run is not false")
    if run_meta.get("h_star_fitted") is not False or published.get("h_star_fitted") is not False:
        mismatches.append("h_star_fitted is not false")
    if run_meta.get("qwen_loaded") is not False or published.get("qwen_loaded") is not False:
        mismatches.append("qwen_loaded is not false")
    if gate.get("passed") is not True or gate.get("device") != "mps":
        mismatches.append("the full-resolution gate did not pass on MPS")
    if run_meta.get("full_gradient_pass") is True:
        if len(gradients) != 144:
            mismatches.append(f"full gradient pass has {len(gradients)} rows, expected 144")
        for row in gradients:
            if row.get("height") != 384 or row.get("width") != 512 or row.get("device") != "mps":
                mismatches.append(f"{row.get('scene_id')} gradient is not a 512 by 384 MPS row")
                break
            if row.get("loss") != "canonical_word":
                mismatches.append(f"{row.get('scene_id')} gradient is not the canonical-word loss")
                break
    predictions = load_jsonl(paths["predictions.jsonl"])
    gradients = load_jsonl(paths["gradient_stats.jsonl"])
    alignments = load_jsonl(paths["alignments.jsonl"])
    expected = [
        json.loads(line)["scene_id"]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rebuilt = build_summary(
        predictions,
        gradients,
        alignments,
        run_meta=run_meta,
        gate=gate,
        expected_scene_ids=expected,
        require_full_sample=True,
    )
    mismatches.extend(compare_summaries(rebuilt, published))
    return {
        "ok": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "label": LABEL,
        "n_groups_eligible": rebuilt["eligible"]["n_groups_eligible"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Checkpoint 1.5 v2 validation summary.")
    parser.add_argument("--directory", type=Path, default=None)
    args = parser.parse_args(argv)
    root = repo_root()
    directory = args.directory if args.directory is not None else root / OUTPUT_RELATIVE
    result = verify_directory(directory, root)
    print(json.dumps({key: result[key] for key in ("ok", "n_mismatches", "label")}, indent=2))
    for mismatch in result["mismatches"]:
        print(mismatch)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
