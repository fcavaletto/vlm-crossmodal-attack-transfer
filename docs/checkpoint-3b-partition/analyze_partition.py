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
ROOT = HERE.parents[1]

# The frozen s16 panel, in its frozen order (the Checkpoint 2.3 subset used by Checkpoint 3 and this plan).
FROZEN_S16 = (
    "dev4-ref-00-a", "dev4-ref-03-a", "dev4-ref-07-a", "dev4-ref-12-a", "dev4-ref-16-a", "dev4-ref-20-a",
    "dev4-ref-25-a", "dev4-ref-27-a", "dev4-ref-29-a", "dev4-ref-30-a", "dev4-ref-33-a", "dev4-ref-37-a",
    "dev4-ref-38-a", "dev4-ref-39-a", "dev4-ref-42-a", "dev4-ref-47-a",
)


def canonical_s16(root: Path = ROOT) -> list[str]:
    """s16 from the project's canonical selection on the hash-checked development manifest, which must equal
    the frozen list. Independent of any run's progress file."""
    from dev_pilot.fixtures import load_manifest
    from path_probe.run import select_scenes

    s16, _ = select_scenes(load_manifest(root))
    if list(s16) != list(FROZEN_S16):
        raise SystemExit("the canonical s16 selection differs from the frozen s16 list")
    return list(s16)


def check_panel(progress: dict, rows: list[dict], expected: list[str]) -> None:
    """The run must cover exactly the 16 unique planned scenes, in the frozen order."""
    if len(expected) != 16 or len(set(expected)) != 16:
        raise SystemExit("the expected panel is not 16 unique scene IDs")
    if list(progress.get("planned_scenes", [])) != list(expected) or list(progress.get("scene_order", [])) != list(expected):
        raise SystemExit("INCOMPLETE or wrong panel: the run's planned scenes are not the frozen s16 in order")
    seen: list[str] = []
    for r in rows:
        if r.get("kind") == "image" and r["scene_id"] not in seen:
            seen.append(r["scene_id"])
    if seen != list(expected):
        raise SystemExit(f"INCOMPLETE or wrong panel: image records cover {len(seen)} scenes, not exactly the frozen s16 in order")


def analyze(run_dir: Path, expected: list[str] | None = None) -> dict:
    progress = json.loads((run_dir / "progress.json").read_text())
    if progress.get("status") != "completed":
        raise SystemExit(f"run status is {progress.get('status')!r}: INCOMPLETE, the planned analysis is not run")
    rows = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    planned = canonical_s16() if expected is None else list(expected)
    check_panel(progress, rows, planned)
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
