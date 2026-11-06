"""Write the Checkpoint 1.5 development sample. Refuses every other split.

The output directory is ``fixtures/development_v2``. This module will not write
``fixtures/development``, which is the Checkpoint 1 sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image

from .design import build_scenes
from .geometry import CANVAS_H, CANVAS_W, CELL, COLS, ROWS
from .records import materialize, scene_graph_hash
from .render import contact_sheet_ids, render_contact_sheet, render_scene
from .seeds import DEVELOPMENT_SEED_DOMAIN, SEED_MODULUS
from .version import __version__
from .vocabulary import WORD_TO_COUNT

CHECKPOINT1_GENERATOR = "scene_gen.generate"
CHECKPOINT1_NAME = "study2_development_v0"


class SealedSplitError(RuntimeError):
    """Raised when a caller asks for anything other than the development split."""


def refuse_non_development_split(split: str) -> None:
    """The sealed procedure is documented. It is not implemented."""
    if split != "development":
        raise SealedSplitError(
            f"refusing split {split!r}. This package writes only split 'development'. "
            "The sealed-set procedure is documented in docs/data/DATASET_V2_DESIGN.md "
            "and is not implemented. No held-out scene is generated."
        )


def refuse_checkpoint1_directory(output: Path) -> None:
    """Do not replace the Checkpoint 1 fixtures, even with ``--force``."""
    output = Path(output)
    if output.name == "development" and output.parent.name == "fixtures":
        raise FileExistsError(
            f"{output} is the Checkpoint 1 development sample. "
            "The revised sample writes to fixtures/development_v2."
        )
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if payload.get("generator") == CHECKPOINT1_GENERATOR or payload.get("name") == CHECKPOINT1_NAME:
        raise FileExistsError(
            f"{output} holds the Checkpoint 1 sample ({payload.get('name')}). Refusing to overwrite it."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _answer_counts(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    answers: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record["task_family"])
        answers.setdefault(family, {})
        answer = str(record["answer"])
        answers[family][answer] = answers[family].get(answer, 0) + 1
    return answers


def _object_counts(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """How many scenes of each family have each object count. Keys are strings for JSON."""
    counts: dict[str, Counter[int]] = {}
    for record in records:
        family = str(record["task_family"])
        counts.setdefault(family, Counter())
        counts[family][len(record["objects"])] += 1  # type: ignore[arg-type]
    return {
        family: {str(number): count for number, count in sorted(counter.items())}
        for family, counter in counts.items()
    }


def _counting_transitions(records: list[dict[str, object]]) -> dict[str, int]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        if record["task_family"] != "counting":
            continue
        grouped.setdefault(str(record["counterfactual_group_id"]), []).append(record)
    transitions: Counter[str] = Counter()
    for members in grouped.values():
        members.sort(key=lambda record: str(record["scene_id"]))
        low = WORD_TO_COUNT[str(members[0]["answer"])]
        high = WORD_TO_COUNT[str(members[1]["answer"])]
        transitions[f"{low}-{high}"] += 1
    return dict(sorted(transitions.items()))


def generate_development(output: Path, *, force: bool = False) -> dict[str, object]:
    """Write the development manifest, images, contact sheet, and checksums.

    ``image_path`` is relative to ``output``, the directory that contains
    ``manifest.jsonl``.
    """
    refuse_non_development_split("development")
    output = Path(output)
    refuse_checkpoint1_directory(output)
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists. Refusing to overwrite without --force. "
            "Regenerating changes committed fixtures; do it on purpose."
        )
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = [materialize(spec) for spec in build_scenes()]
    rendered: dict[str, Image.Image] = {}
    expected_names: set[str] = set()
    for record in records:
        image = render_scene(record, canvas_w=CANVAS_W, canvas_h=CANVAS_H)
        relative = str(record["image_path"])
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG", optimize=False, compress_level=9)
        rendered[str(record["scene_id"])] = image
        expected_names.add(target.name)

    for stale in images_dir.glob("*.png"):
        if stale.name not in expected_names:
            stale.unlink()

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    sheet_ids = contact_sheet_ids()
    panels = []
    for scene_id in sheet_ids:
        record = next(item for item in records if item["scene_id"] == scene_id)
        family = str(record["task_family"])
        letter = {"attribute": "a", "counting": "c", "spatial": "s"}[family]
        _prefix, _family, index, arm = str(record["scene_id"]).split("-")
        caption = f"{letter}{index}{arm} {record['answer']}"
        panels.append((caption, rendered[scene_id]))
    sheet = render_contact_sheet(panels)
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)

    families = Counter(str(record["task_family"]) for record in records)
    metadata: dict[str, object] = {
        "name": "study2_development_v2",
        "status": "checkpoint_1_5_development_sample",
        "generator": "scene_v2.generate",
        "generator_version": __version__,
        "scene_count": len(records),
        "split": "development",
        "seed_domain": DEVELOPMENT_SEED_DOMAIN,
        "seed_modulus": SEED_MODULUS,
        "image_size": [CANVAS_W, CANVAS_H],
        "grid": {"cols": COLS, "rows": ROWS, "cell": [CELL, CELL]},
        "pillow": PIL.__version__,
        "task_counts": dict(families),
        "answer_counts": _answer_counts(records),
        "object_counts_by_family": _object_counts(records),
        "counting_transitions": _counting_transitions(records),
        "counterfactual_group_count": len({r["counterfactual_group_id"] for r in records}),
        "normalization": "scene_v2.normalize.normalize_for_leakage",
        "image_path_rule": "relative to the directory that contains manifest.jsonl",
        "contact_sheet": "contact_sheet.png",
        "contact_sheet_scene_ids": list(sheet_ids),
        "contact_sheet_layout": (
            "12 pair columns. Rows are attribute A, attribute B, counting A, "
            "counting B, spatial A, spatial B."
        ),
        "contact_sheet_is_model_input": False,
        "replaces_checkpoint_1_sample": False,
        "checkpoint_1_fixture_dir": "fixtures/development",
        "dataset_origin": (
            "Original Pillow drawings from the scene graph. "
            "No photographs, no CLEVR images, no downloaded assets."
        ),
        "caution": (
            "Development sample only. Not a sealed test set. "
            "Not a measurement of clean accuracy or attack success. "
            "No model was run to produce these files."
        ),
    }
    _write_json(output / "metadata.json", metadata)

    hashes = {str(record["scene_id"]): scene_graph_hash(record) for record in records}
    _write_json(
        output / "scene_graph_hashes.json",
        {
            "algorithm": "sha256",
            "canonicalization": "json sort_keys separators=(',', ':') of question, answer, objects",
            "purpose": (
                "Exclusion list for a future sealed generator, together with "
                "fixtures/development/scene_graph_hashes.json. This file is not a sealed set."
            ),
            "hashes": hashes,
        },
    )

    checksum_paths = [
        manifest_path,
        sheet_path,
        output / "metadata.json",
        output / "scene_graph_hashes.json",
        *sorted(images_dir.glob("*.png")),
    ]
    files = {}
    for path in checksum_paths:
        files[str(path.relative_to(output))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json(
        output / "checksums.json",
        {
            "algorithm": "sha256",
            "pillow": PIL.__version__,
            "note": (
                "Integrity of the committed files. A later machine may encode PNG "
                "bytes differently; compare pixels before replacing these files."
            ),
            "files": files,
        },
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write the Checkpoint 1.5 development scenes. "
            "Will not write a sealed split or the Checkpoint 1 sample."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/development_v2"),
        help="directory that will contain manifest.jsonl (default: fixtures/development_v2)",
    )
    parser.add_argument(
        "--split",
        default="development",
        help="only 'development' is accepted",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing manifest in the output directory",
    )
    args = parser.parse_args(argv)
    try:
        refuse_non_development_split(args.split)
        metadata = generate_development(args.output, force=args.force)
    except SealedSplitError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
