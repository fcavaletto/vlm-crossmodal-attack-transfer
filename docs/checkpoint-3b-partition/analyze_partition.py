"""Checkpoint 3b analysis (PLAN.md §6). Runs once on the complete 16-scene panel.

    python docs/checkpoint-3b-partition/analyze_partition.py [run_dir]

Reads ``run/records.jsonl`` and ``run/progress.json``; writes ``analysis/partition_analysis.json``.
Refuses an incomplete run (time cap, stop, error, failed drift check) or an existing output: the planned
analysis is defined only for the complete panel. EXPLORATORY: intervals are conditional on this
development panel and the specified randomization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from path_probe import partition as pt

HERE = Path(__file__).resolve().parent


def analyze(run_dir: Path) -> dict:
    progress = json.loads((run_dir / "progress.json").read_text())
    if progress.get("status") != "completed":
        raise SystemExit(f"run status is {progress.get('status')!r}: INCOMPLETE, the planned analysis is not run")
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    planned = progress["planned_scenes"]
    per_scene = [pt.scene_quantities([r for r in rows if r["kind"] == "image" and r["scene_id"] == sid]) for sid in planned]
    structure = {r["scene_id"]: {k: r[k] for k in ("k_retained", "n_pgd_support", "n_excluded_outside_fgsm",
                                                     "sign_disagreements_on_overlap", "pgd_sign_counts", "quarter_sizes")}
                 for r in rows if r["kind"] == "structure"}
    drift = {r["scene_id"]: r["bit_exact_vs_record"] for r in rows if r["kind"] == "pgd_official"}
    return {
        "schema": "checkpoint-3b.partition.analysis.v1",
        "label": "EXPLORATORY; development panel s16; intervals conditional on this panel and the specified randomization",
        "plan_commit": progress["plan_commit"],
        "primary": "fixed clean-competitor contrast, mean Delta = A - Q, paired two-sided 95% percentile scene bootstrap",
        "panel": pt.panel(per_scene, planned),
        "per_scene": per_scene,
        "structure": structure,
        "pgd_official_bit_exact_vs_2.3b": drift,
        "final_drift_check": progress.get("drift_check"),
    }


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run"
    out = HERE / "analysis" / "partition_analysis.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the analysis runs once")
    result = analyze(run_dir)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    p = result["panel"]["fixed"]
    for q in ("A", "Q", "Delta"):
        print(f"{q}: {p[q]['mean']:+.3f} {p[q]['ci95']} ({p[q]['direction']})")


if __name__ == "__main__":
    main()
