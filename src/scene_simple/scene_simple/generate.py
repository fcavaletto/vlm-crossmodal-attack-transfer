"""Write the Checkpoint 1.6 development sample. Refuses every other split.

The output directory is a new ``development_simple`` sample. This module will
not write ``fixtures/development`` or ``fixtures/development_v2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import PIL
from PIL import Image

from .design import build_scenes
from .geometry import CANVAS_H, CANVAS_W, CELL_H, CELL_W, COLS, ROWS
from .records import materialize, scene_graph_hash
from .render import contact_sheet_ids, render_contact_sheet, render_scene
from .seeds import DEVELOPMENT_SEED_DOMAIN, SEED_MODULUS
from .version import __version__
from .vocabulary import RADIUS

PROTECTED_GENERATORS = {"scene_gen.generate", "scene_v2.generate"}
PROTECTED_NAMES = {"study2_development_v0", "study2_development_v2"}
PROTECTED_DIR_NAMES = {"development", "development_v2"}


class SealedSplitError(RuntimeError):
    """Raised when a caller asks for anything other than the development split."""


def refuse_non_development_split(split: str) -> None:
    """The package has no sealed, test, or held-out generator."""
    if split != "development":
        raise SealedSplitError(
            f"refusing split {split!r}. This package writes only split 'development'. "
            "No sealed, test, or held-out scene is generated."
        )


def refuse_existing_samples(output: Path) -> None:
    """Do not replace the Checkpoint 1 or Checkpoint 1.5 fixtures, even with ``--force``."""
    output = Path(output)
    if output.name in PROTECTED_DIR_NAMES and "fixtures" in output.parts:
        raise FileExistsError(
            f"Refusing to overwrite {output}. "
            "fixtures/development is the Checkpoint 1 sample and "
            "fixtures/development_v2 is the Checkpoint 1.5 sample. "
            "This package writes only a new development_simple directory."
        )
    metadata_path = output / "metadata.json"
    if not metadata_path.is_file():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    generator = payload.get("generator")
    name = payload.get("name")
    if generator in PROTECTED_GENERATORS or name in PROTECTED_NAMES:
        raise FileExistsError(
            f"Refusing to overwrite {output}. It holds {name or generator}."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_name(question: str) -> str:
    match = re.search(r"\b(top|bottom) row\b", question)
    if match is None:
        raise ValueError(f"question has no row name: {question}")
    return match.group(1)


def _shape_name(question: str) -> str:
    match = re.search(r"\b(circle|square|triangle)\b", question)
    if match is None:
        raise ValueError(f"question has no shape: {question}")
    return match.group(1)


def generate_development(output: Path, *, force: bool = False) -> dict[str, object]:
    """Write the development manifest, images, contact sheet, and checksums.

    ``image_path`` is relative to ``output``, the directory that contains
    ``manifest.jsonl``. Every row has ``split`` equal to ``development``.
    """
    refuse_non_development_split("development")
    output = Path(output)
    refuse_existing_samples(output)
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
        if record["split"] != "development":
            raise RuntimeError(f"{record['scene_id']} is not a development row")
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
        _prefix, _family, index, arm = str(record["scene_id"]).split("-")
        caption = f"{index}{arm} {record['answer']}"
        panels.append((caption, rendered[scene_id]))
    sheet = render_contact_sheet(panels)
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)

    families = Counter(str(record["task_family"]) for record in records)
    answers: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record["task_family"])
        answers.setdefault(family, {})
        answer = str(record["answer"])
        answers[family][answer] = answers[family].get(answer, 0) + 1
    object_counts: dict[str, Counter[int]] = {}
    for record in records:
        family = str(record["task_family"])
        object_counts.setdefault(family, Counter())
        object_counts[family][len(record["objects"])] += 1  # type: ignore[arg-type]
    foil_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    row_counts: Counter[str] = Counter()
    for record in records:
        text = str(record["contexts"]["contradictory"]["text"])  # type: ignore[index]
        foil_counts[text.rsplit(" ", 1)[-1].rstrip(".")] += 1
        shape_counts[_shape_name(str(record["question"]))] += 1
        row_counts[_row_name(str(record["question"]))] += 1

    metadata: dict[str, object] = {
        "name": "study2_development_simple",
        "status": "checkpoint_1_6_development_sample",
        "generator": "scene_simple.generate",
        "generator_version": __version__,
        "scene_count": len(records),
        "split": "development",
        "seed_domain": DEVELOPMENT_SEED_DOMAIN,
        "seed_modulus": SEED_MODULUS,
        "image_size": [CANVAS_W, CANVAS_H],
        "grid": {"cols": COLS, "rows": ROWS, "cell": [CELL_W, CELL_H]},
        "radius": RADIUS["large"],
        "pillow": PIL.__version__,
        "task_families": sorted(families),
        "task_counts": dict(families),
        "answer_counts": answers,
        "object_counts_by_family": {
            family: {str(number): count for number, count in sorted(counter.items())}
            for family, counter in object_counts.items()
        },
        "queried_shape_counts": dict(sorted(shape_counts.items())),
        "queried_row_counts": dict(sorted(row_counts.items())),
        "foil_color_counts": dict(sorted(foil_counts.items())),
        "counterfactual_group_count": len({r["counterfactual_group_id"] for r in records}),
        "counting_is_a_graded_family": False,
        "normalization": "scene_simple.normalize.normalize_for_leakage",
        "prompt_join": "context text, a newline, then the question. The answer is not in the question.",
        "image_path_rule": "relative to the directory that contains manifest.jsonl",
        "contact_sheet": "contact_sheet.png",
        "contact_sheet_scene_ids": list(sheet_ids),
        "contact_sheet_layout": "12 pair columns. Rows are arm A, then arm B.",
        "contact_sheet_is_model_input": False,
        "replaces_checkpoint_1_sample": False,
        "replaces_checkpoint_1_5_sample": False,
        "checkpoint_1_fixture_dir": "fixtures/development",
        "checkpoint_1_5_fixture_dir": "fixtures/development_v2",
        "model_was_run": False,
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
                "fixtures/development/scene_graph_hashes.json and "
                "fixtures/development_v2/scene_graph_hashes.json. This file is not a sealed set."
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
            "Write the Checkpoint 1.6 development scenes. "
            "Will not write a sealed split, fixtures/development, or fixtures/development_v2."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/development_simple"),
        help="directory that will contain manifest.jsonl (default: fixtures/development_simple)",
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
