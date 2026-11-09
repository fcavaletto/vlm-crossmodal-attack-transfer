"""Write the Track B confirmation split ``study2-trackB-confirm-v1``.

``python -m scene_confirm --output fixtures/trackB_confirm_v1``

No model is run. The writer refuses an existing output directory, every older
fixture directory, and every seed string other than the confirmation one.
Separation from development data is checked three ways: layout signatures
(with mirrors) against ``development_v4``, scene-graph hashes against all four
development exclusion lists, and decoded-pixel hashes against every committed
development image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
from scene_v4.design import _signature, build_scenes
from scene_v4.geometry import CANVAS_H, CANVAS_W
from scene_v4.records import materialize, scene_graph_hash
from scene_v4.render import render_contact_sheet, render_scene

from .draw import CONFIRM_SEED_STRING, MAX_ATTEMPTS, draw_split, refuse_reserved

N_GROUPS = 48
SPLIT = "trackB-confirm"
PRIMARY_ARM = "a"
DEV_FIXTURES = ("development", "development_v2", "development_simple", "development_v4")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_hash(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def development_exclusions(root: Path) -> dict[str, set[str]]:
    graphs: set[str] = set()
    pixels: set[str] = set()
    for name in DEV_FIXTURES:
        d = root / "fixtures" / name
        graphs |= set(json.loads((d / "scene_graph_hashes.json").read_text())["hashes"].values())
        for png in sorted((d / "images").glob("*.png")):
            with Image.open(png) as im:
                pixels.add(pixel_hash(im))
    return {"scene_graph": graphs, "pixels": pixels}


def build(root: Path, seed_string: str = CONFIRM_SEED_STRING, n_groups: int = N_GROUPS):
    """Records (both arms, group order) and generation statistics. Pure, nothing written."""
    refuse_reserved(seed_string)
    dev_sigs = {_signature(s.objects) for s in build_scenes()}
    groups = draw_split(seed_string, n_groups, dev_sigs)
    records = []
    for g in groups:
        for spec in (g.arm_a, g.arm_b):
            rec = materialize(spec)  # v4 leakage, oracle, and ambiguity checks
            rec["split"] = SPLIT
            rec["seed"] = g.seed
            rec["seed_string"] = seed_string
            rec["group_index"] = g.index
            rec["accepted_attempt"] = g.attempt
            rec["primary_arm"] = spec.scene_id.endswith(f"-{PRIMARY_ARM}")
            records.append(rec)
    stats = {
        "attempts_total": sum(g.attempt + 1 for g in groups),
        "rejected_constraint": sum(g.rejections.get("constraint", 0) for g in groups),
        "rejected_novelty": sum(g.rejections.get("novelty", 0) for g in groups),
        "max_attempt_index": max(g.attempt for g in groups),
        "attempt_cap": MAX_ATTEMPTS,
        "development_v4_layouts_excluded": len(dev_sigs),
    }
    return records, stats


def balance(records: list[dict]) -> dict:
    prim = [r for r in records if r["primary_arm"]]
    out: dict[str, dict] = {}
    for name, rows in (("primary_arm", prim), ("both_arms", records)):
        q = [r["question"].split() for r in rows]
        out[name] = {
            "answer": dict(sorted(Counter(r["answer"] for r in rows).items())),
            "size": dict(sorted(Counter(w[4] for w in q).items())),
            "shape": dict(sorted(Counter(w[5] for w in q).items())),
            "row": dict(sorted(Counter(w[8] for w in q).items())),
            "target_column": dict(sorted(Counter(str(r["objects"][0]["position"]["col"]) for r in rows).items())),
            "foil": dict(sorted(Counter(r["contexts"]["contradictory"]["text"].rsplit(" ", 1)[-1].rstrip(".") for r in rows).items())),
        }
    pairs = Counter()
    by_group: dict[str, dict[str, str]] = {}
    for r in records:
        by_group.setdefault(r["counterfactual_group_id"], {})[r["scene_id"][-1]] = r["answer"]
    for v in by_group.values():
        pairs[f"{v['a']}->{v['b']}"] += 1
    out["ordered_answer_pairs"] = dict(sorted(pairs.items()))
    return out


def generate(root: Path, output: Path) -> dict:
    output = Path(output)
    if output.name in DEV_FIXTURES or output.exists():
        raise FileExistsError(f"refusing to write {output}: it exists or is a development fixture")
    records, stats = build(root)
    excl = development_exclusions(root)
    hashes = {r["scene_id"]: scene_graph_hash(r) for r in records}
    clash_graph = sorted(s for s, h in hashes.items() if h in excl["scene_graph"])
    if clash_graph:
        raise RuntimeError(f"scene graphs equal to development scenes: {clash_graph}")
    if len(set(hashes.values())) != len(hashes):
        raise RuntimeError("duplicate scene graph inside the split")

    output.mkdir(parents=True)
    (output / "images").mkdir()
    images, pix = {}, {}
    for r in records:
        im = render_scene(r, canvas_w=CANVAS_W, canvas_h=CANVAS_H)
        im.save(output / r["image_path"], format="PNG", optimize=False, compress_level=9)
        images[r["scene_id"]] = im
        pix[r["scene_id"]] = pixel_hash(im)
    clash_pix = sorted(s for s, h in pix.items() if h in excl["pixels"])
    if clash_pix:
        raise RuntimeError(f"images identical to development images: {clash_pix}")
    if len(set(pix.values())) != len(pix):
        raise RuntimeError("duplicate image inside the split")

    with (output / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    panels = []
    for band in range(4):
        for arm in ("a", "b"):
            for col in range(12):
                k = band * 12 + col
                sid = f"tB-confirm-{k:03d}-{arm}"
                ans = next(r["answer"] for r in records if r["scene_id"] == sid)
                panels.append((f"{k:03d}{arm} {ans}", images[sid]))
    render_contact_sheet(panels).save(output / "contact_sheet.png", format="PNG", optimize=False, compress_level=9)

    metadata = {
        "name": "study2_trackB_confirm_v1",
        "split": SPLIT,
        "seed_string": CONFIRM_SEED_STRING,
        "generator": "scene_confirm.generate",
        "sampling_distribution": "docs/checkpoint-2.2-draft/V5_GENERATOR_SPEC.md section 3, conditioned on the v4 per-group checks",
        "n_groups": N_GROUPS,
        "n_scenes": len(records),
        "primary_arm": PRIMARY_ARM,
        "primary_arm_rule": "arm a of every group, fixed before generation (same rule as the 2.3 development S48 set)",
        "generation": stats,
        "separation_checks": {
            "layout_signature_and_mirrors_vs_development_v4": "enforced by rejection",
            "layout_signature_and_mirrors_within_split": "enforced by rejection",
            "scene_graph_hash_vs_all_development_fixtures": {"compared_against": len(excl["scene_graph"]), "clashes": 0},
            "pixel_hash_vs_all_development_images": {"compared_against": len(excl["pixels"]), "clashes": 0},
            "duplicates_within_split": 0,
        },
        "realized_balance": balance(records),
        "reserved_samples_touched": False,
        "h_dl_seed_strings_refused": ["study2-checkpoint-2.2-dev-v5", "study2-checkpoint-2.2-sealed-v1"],
        "model_was_run": False,
        "model_outputs_used_to_choose_scenes": False,
        "image_size": [CANVAS_W, CANVAS_H],
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "scene_graph_hashes.json").write_text(json.dumps(
        {"algorithm": "sha256", "purpose": "Exclusion list for any later split (including a future sealed draw).",
         "hashes": hashes, "pixel_hashes": pix}, indent=2, sort_keys=True) + "\n")
    files = {}
    for p in [output / "manifest.jsonl", output / "metadata.json", output / "scene_graph_hashes.json",
              output / "contact_sheet.png", *sorted((output / "images").glob("*.png"))]:
        files[str(p.relative_to(output))] = {"bytes": p.stat().st_size, "sha256": _sha(p)}
    (output / "checksums.json").write_text(json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write the Track B confirmation split. No model is run.")
    ap.add_argument("--output", type=Path, default=Path("fixtures/trackB_confirm_v1"))
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    try:
        meta = generate(args.root.resolve(), args.output)
    except (FileExistsError, PermissionError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0
