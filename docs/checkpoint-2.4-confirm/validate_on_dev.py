"""Validate analyze_confirm.py on DEVELOPMENT data before the freeze.

Converts the 2.3 (clean, FGSM) and 2.3b (structure, PGD-3 mask, top-k, random) development
records into the confirmation schema, runs the frozen analysis, and checks that every point
estimate equals the 2.3b analysis (masks_analysis.json). Bootstrap CIs differ only by the seed.

Run: python docs/checkpoint-2.4-confirm/validate_on_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = ROOT / "docs/checkpoint-2.3-exploratory"
sys.path.insert(0, str(HERE))
from analyze_confirm import analyze  # noqa: E402


def convert(dest: Path) -> None:
    sys.path[:0] = [str(p) for p in (ROOT / "src").glob("*/")]
    from dev_pilot.fixtures import load_manifest
    labels = {r["scene_id"]: str(r["answer"]) for r in load_manifest(ROOT)}
    out = []
    for line in (DEV / "paths/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("mode") != "B":
            continue
        if r["kind"] == "clean":
            out.append({"kind": "clean", "scene_id": r["scene_id"], "label": labels[r["scene_id"]],
                        "scores": {c: {"official": r["scores"][c]["official"]} for c in r["scores"]}})
        elif r["kind"] == "fgsm":
            out.append({"kind": "fgsm", "scene_id": r["scene_id"], "source": r["source"], "k": None,
                        "scores": {c: r["scores"][c]["official"] for c in r["scores"]}})
    for line in (DEV / "masks/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "structure":
            out.append(r)
        elif r["mask"] == "pgd_mask":
            out.append({**r, "kind": "pgd3"})
        elif r["mask"] == "topk_grad":
            out.append({**r, "kind": "topk_grad"})
        else:
            out.append({**r, "kind": "random"})
    dest.write_text("".join(json.dumps(r) + "\n" for r in out))


def main() -> None:
    tmp = HERE / "analysis" / "dev_as_confirm_records.jsonl"
    tmp.parent.mkdir(exist_ok=True)
    convert(tmp)
    mine, per_group = analyze(tmp)
    tmp.unlink()
    import csv
    (HERE / "analysis" / "dev_analysis_confirm_schema.json").write_text(json.dumps(mine, indent=1) + "\n")
    with (HERE / "analysis" / "dev_per_group.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_group[0]))
        w.writeheader()
        w.writerows(per_group)
    ref = json.loads((DEV / "analysis/masks_analysis.json").read_text())
    pairs = []
    for where in ("source", "target"):
        for src in ("neutral", "helpful_nonrevealing", "pooled"):
            a, b = mine[f"margin_drop_{where}"][src], ref[f"margin_drop_{where}"][src]
            for key in ("D_fgsm", "D_pgd", "D_topk", "D_rand", "selection_pgd_minus_rand", "sparsity_rand_minus_fgsm",
                        "magnitude_topk_minus_rand", "total_pgd_minus_fgsm"):
                pairs.append((f"{where}/{src}/{key}", a[key]["mean"]["estimate"], b[key]["mean"]["estimate"]))
            pairs.append((f"{where}/{src}/share", a["share_of_gap_from_sparsity"]["estimate"],
                          b["share_of_gap_from_sparsity"]["estimate"]))
            if src != "pooled":
                pairs.append((f"{where}/{src}/beats_all", a["groups_pgd_beats_all_seeds"], b["units_pgd_beats_all_seeds"]))
                for m in ("fgsm", "pgd", "topk", "rand"):
                    pairs.append((f"{where}/{src}/flip_{m}", a["flip_rate"][m]["estimate"], b["flip_rate"][m]["estimate"]))
    bad = [p for p in pairs if p[1] is None or p[2] is None or abs(p[1] - p[2]) > 1e-4]
    report = {"compared": len(pairs), "mismatches": bad, "dev_decisions_under_confirm_rules": mine["decisions"],
              "structure": mine["structure"]}
    (HERE / "analysis" / "validation_on_dev.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"compared": len(pairs), "mismatches": bad}, indent=1))
    print(json.dumps({k: v["supported"] for k, v in mine["decisions"].items() if isinstance(v, dict)}, indent=1))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
